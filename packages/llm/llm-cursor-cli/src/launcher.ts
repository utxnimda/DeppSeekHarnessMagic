/**
 * Locates the local Cursor CLI binary and discovers models available to the
 * signed-in account. On Windows, resolves the native bundled `node.exe` and
 * `index.js` directly to bypass the slow PowerShell wrapper shim and keep
 * stdio completely clean for `stream-json` and `agent acp`.
 */
import { execFile } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import type { LlmModelInfo } from '@deepseek-ai/dsh-llm'

export interface CursorLauncher {
  readonly available: boolean
  readonly command: string
  readonly prefixArgs: readonly string[]
  readonly label: string
  readonly env: Record<string, string>
  readonly error?: string
}

const VERSION_DIR_RE = /^(\d{4})\.(\d{1,2})\.(\d{1,2})(?:-(\d{2})-(\d{2})-(\d{2}))?-[a-f0-9]+$/i
const ANSI_RE = /\x1b\[[0-9;?]*[ -/]*[@-~]/g
const ZERO_WIDTH_RE = /[\u200b-\u200f\ufeff]/g

/** Parse `YYYY.MM.DD[-HH-MM-SS]-hash` version folder names into a sortable tuple. */
export function parseCursorVersionKey(name: string): string | null {
  const match = VERSION_DIR_RE.exec(name)
  if (!match) return null
  const y = match[1] ?? ''
  const m = match[2] ?? ''
  const d = match[3] ?? ''
  const hh = match[4] ?? '00'
  const mm = match[5] ?? '00'
  const ss = match[6] ?? '00'
  if (!y || !m || !d) return null
  return `${y}${m.padStart(2, '0')}${d.padStart(2, '0')}${hh}${mm}${ss}`
}

/**
 * Resolve the cleanest executable + prefix args for Cursor CLI.
 * Explicit `customCommand` takes priority; otherwise probes Windows
 * `%LOCALAPPDATA%\cursor-agent\versions\<latest>\(node.exe, index.js)` before
 * falling back to `agent` on PATH.
 */
export function resolveCursorLauncher(
  customCommand = '',
  env: NodeJS.ProcessEnv = process.env,
): CursorLauncher {
  const extraEnv: Record<string, string> = {}
  const localAppData = env.LOCALAPPDATA ?? ''
  if (localAppData && !env.NODE_COMPILE_CACHE) {
    extraEnv.NODE_COMPILE_CACHE = join(localAppData, 'cursor-compile-cache')
  }
  if (env.CURSOR_API_KEY) {
    extraEnv.CURSOR_API_KEY = env.CURSOR_API_KEY
  }
  if (env.CURSOR_API_ENDPOINT) {
    extraEnv.CURSOR_API_ENDPOINT = env.CURSOR_API_ENDPOINT
  }

  const trimmed = customCommand.trim()
  if (trimmed) {
    return {
      available: true,
      command: trimmed,
      prefixArgs: [],
      label: trimmed,
      env: extraEnv,
    }
  }

  if (process.platform === 'win32' && localAppData) {
    const baseDir = join(localAppData, 'cursor-agent')
    const directNode = join(baseDir, 'node.exe')
    const directEntry = join(baseDir, 'index.js')
    if (existsSync(directNode) && existsSync(directEntry)) {
      return {
        available: true,
        command: directNode,
        prefixArgs: [directEntry],
        label: directEntry,
        env: extraEnv,
      }
    }

    const versionsDir = join(baseDir, 'versions')
    if (existsSync(versionsDir)) {
      try {
        const candidates = readdirSync(versionsDir, { withFileTypes: true })
          .filter(entry => entry.isDirectory())
          .map(entry => ({ name: entry.name, sortKey: parseCursorVersionKey(entry.name) }))
          .filter((item): item is { name: string; sortKey: string } => item.sortKey !== null)
          .sort((a, b) => b.sortKey.localeCompare(a.sortKey))

        for (const item of candidates) {
          const dir = join(versionsDir, item.name)
          const nodeExe = join(dir, 'node.exe')
          const entryJs = join(dir, 'index.js')
          if (existsSync(nodeExe) && existsSync(entryJs)) {
            return {
              available: true,
              command: nodeExe,
              prefixArgs: [entryJs],
              label: `cursor-agent ${item.name}`,
              env: extraEnv,
            }
          }
        }
      } catch {
        // Fall through to PATH fallback
      }
    }
  }

  return {
    available: true,
    command: 'agent',
    prefixArgs: [],
    label: 'agent (PATH)',
    env: extraEnv,
  }
}

export interface ParsedCursorModel {
  readonly id: string
  readonly name: string
}

/** Parse `agent --list-models` stdout into clean model descriptors. */
export function parseModelListOutput(stdout: string): ParsedCursorModel[] {
  const models: ParsedCursorModel[] = []
  const seen = new Set<string>()
  const cleaned = stdout.replace(ANSI_RE, '').replace(ZERO_WIDTH_RE, '')
  for (const rawLine of cleaned.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line === 'Available models' || line.startsWith('Tip:') || line.startsWith('Loading')) continue
    const sep = line.indexOf(' - ')
    if (sep <= 0) continue
    const id = line.slice(0, sep).trim()
    const name = line.slice(sep + 3).replace(/\s+/g, ' ').trim()
    if (!id || /\s/.test(id) || seen.has(id)) continue
    seen.add(id)
    models.push({ id, name: name || id })
  }
  return models
}

const FALLBACK_MODELS: readonly ParsedCursorModel[] = [
  { id: 'auto', name: 'Auto (default)' },
]

/** Estimate context window from Cursor model display name (e.g. `1M`). */
export function inferContextWindow(modelId: string, modelName: string): number {
  const combined = `${modelId} ${modelName}`.toLowerCase()
  if (combined.includes('1m')) return 1_000_000
  if (combined.includes('gemini-4') || combined.includes('muse')) return 300_000
  if (combined.includes('grok-4.7')) return 256_000
  return 200_000
}

export function toLlmModelInfo(provider: string, item: ParsedCursorModel): LlmModelInfo {
  return {
    provider,
    id: item.id,
    name: item.name,
  }
}

export class CursorModelCatalog {
  private cached: readonly ParsedCursorModel[] | null = null
  private cachedAt = 0
  private inflight: Promise<readonly ParsedCursorModel[]> | null = null
  private readonly ttlMs = 5 * 60 * 1000

  constructor(private readonly getLauncher: () => CursorLauncher) {}

  async list(provider: string): Promise<readonly LlmModelInfo[]> {
    const raw = await this.loadRaw()
    return raw.map(item => toLlmModelInfo(provider, item))
  }

  findCached(modelId: string): ParsedCursorModel | undefined {
    return this.cached?.find(item => item.id === modelId)
  }

  private async loadRaw(): Promise<readonly ParsedCursorModel[]> {
    const now = Date.now()
    if (this.cached !== null && now - this.cachedAt < this.ttlMs) {
      return this.cached
    }
    if (this.inflight !== null) {
      return this.inflight
    }
    this.inflight = this.probeOnce().finally(() => {
      this.inflight = null
    })
    return this.inflight
  }

  private probeOnce(): Promise<readonly ParsedCursorModel[]> {
    const launcher = this.getLauncher()
    const env = { ...process.env, ...launcher.env }
    return new Promise((resolve) => {
      execFile(
        launcher.command,
        [...launcher.prefixArgs, '--list-models'],
        {
          env,
          timeout: 15_000,
          maxBuffer: 2 * 1024 * 1024,
          windowsHide: true,
        },
        (error, stdout) => {
          if (error) {
            resolve(this.cached ?? FALLBACK_MODELS)
            return
          }
          const parsed = parseModelListOutput(stdout)
          if (parsed.length > 0) {
            this.cached = parsed
            this.cachedAt = Date.now()
            resolve(parsed)
          } else {
            resolve(this.cached ?? FALLBACK_MODELS)
          }
        },
      )
    })
  }
}
