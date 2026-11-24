/**
 * Persistent JSON-RPC 2.0 ACP (Agent Client Protocol) client for the local
 * Cursor CLI (`agent acp`).
 *
 * Uses `clientCapabilities._meta.parameterizedModelPicker = true` and the
 * `cursor/list_available_models` extension method to expose:
 * - Every selectable context window variant (e.g. `Gemini 4 Neon (300K)` & `Gemini 4 Neon (1M)`)
 *   with accurate DSH `contextWindow` accounting;
 * - Every model's real reasoning effort / Auto router tier (`reasoning.efforts`);
 * - Real-time ACP `session/set_config_option` application for model, context, and effort.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import type { CursorLauncher } from './launcher.ts'

export interface AcpEffortOption {
  readonly id: string
  readonly name: string
}

export interface AcpModelOption {
  /** DSH model route ID (e.g. `gemini-4-neon` for default context, `gemini-4-neon@1m` for 1M context). */
  readonly modelId: string
  /** Underlying Cursor ACP model value passed to `session/set_config_option(configId='model')`. */
  readonly acpModelId: string
  readonly name: string
  readonly contextWindow: number
  /** Cursor ACP `context` config option value (e.g. `300k`, `1m`, `500k`), when applicable. */
  readonly contextOptionValue?: string | undefined
  readonly effortConfigId?: string | undefined
  readonly efforts: readonly AcpEffortOption[]
  readonly defaultEffort?: string | undefined
}

export interface AcpSessionUpdate {
  readonly sessionUpdate: string
  readonly content?: {
    readonly type?: string
    readonly text?: string
  }
  readonly currentModeId?: string
}

interface JsonRpcMessage {
  jsonrpc?: string
  id?: number | string | null
  method?: string
  params?: Record<string, unknown>
  result?: Record<string, unknown>
  error?: {
    code?: number
    message?: string
    data?: unknown
  }
}

interface PendingRpc {
  readonly resolve: (result: Record<string, unknown>) => void
  readonly reject: (error: Error) => void
}

export interface AcpPromptRequest {
  readonly model: string
  readonly reasoningEffort?: string | undefined
  readonly mode: 'ask' | 'plan' | 'agent'
  readonly prompt: string
  readonly cwd: string
  readonly signal?: AbortSignal | undefined
  readonly onUpdate: (update: AcpSessionUpdate) => void
}

export interface AcpPromptResult {
  readonly stopReason: string
}

export class AcpRpcError extends Error {
  constructor(
    message: string,
    readonly rpcCode?: number,
    readonly data?: unknown,
  ) {
    super(message)
    this.name = 'AcpRpcError'
  }
}

const ZERO_WIDTH_RE = /[\u200b-\u200f\ufeff]/g

function cleanLabel(text: string): string {
  return text.replace(ZERO_WIDTH_RE, '').replace(/\s+/g, ' ').trim()
}

function parseContextTokenString(raw: string): number | null {
  const m = /(\d+)\s*(k|m)\b/i.exec(raw)
  if (!m) return null
  const qty = Number(m[1])
  const unit = (m[2] ?? '').toLowerCase()
  if (!Number.isFinite(qty) || qty <= 0) return null
  return unit === 'm' ? qty * 1_000_000 : qty * 1_000
}

/** Extract context window from ACP parameterized model IDs or display names. */
export function inferAcpContextWindow(modelId: string, modelName: string, contextOptionValue?: string): number {
  if (contextOptionValue) {
    const parsed = parseContextTokenString(contextOptionValue)
    if (parsed !== null) return parsed
  }
  const atMatch = /@(\d+(?:k|m))\b/i.exec(modelId)
  if (atMatch?.[1]) {
    const parsed = parseContextTokenString(atMatch[1])
    if (parsed !== null) return parsed
  }
  const paramMatch = /context=(\d+)(k|m)/i.exec(modelId)
  if (paramMatch) {
    const qty = Number(paramMatch[1])
    const unit = (paramMatch[2] ?? '').toLowerCase()
    if (Number.isFinite(qty) && qty > 0) {
      return unit === 'm' ? qty * 1_000_000 : qty * 1_000
    }
  }
  const combined = `${modelId} ${modelName}`.toLowerCase()
  if (combined.includes('1m')) return 1_000_000
  if (combined.includes('500k')) return 500_000
  if (combined.includes('gemini-4') || combined.includes('muse')) return 300_000
  if (combined.includes('gpt-5.6') || combined.includes('gpt-5.5') || combined.includes('gpt-5.4')) return 272_000
  if (combined.includes('grok-4.7')) return 256_000
  return 200_000
}

/** Strip legacy bracket parameters or `@context` suffix to obtain the base ACP model ID. */
export function stripModelParams(modelId: string): string {
  const trimmed = modelId.trim()
  const bracket = trimmed.indexOf('[')
  const withoutBracket = bracket >= 0 ? trimmed.slice(0, bracket).trim() : trimmed
  const at = withoutBracket.indexOf('@')
  return at >= 0 ? withoutBracket.slice(0, at).trim() : withoutBracket
}

/** Extract an explicit context option value from `model@1m` or legacy `model[context=1m,...]`. */
export function extractExplicitContextValue(modelId: string): string | undefined {
  const atMatch = /@([a-z0-9._-]+)$/i.exec(modelId.trim())
  if (atMatch?.[1]) return atMatch[1].toLowerCase()
  const bracketMatch = /context=([a-z0-9._-]+)/i.exec(modelId)
  if (bracketMatch?.[1]) return bracketMatch[1].toLowerCase()
  return undefined
}

export interface ResolvedAcpModelTarget {
  readonly acpModelId: string | null
  readonly contextValue?: string | undefined
}

/**
 * Resolve a requested DSH model ID (including `@1m` context suffixes and legacy aliases
 * like `auto` or `gemini-4-neon[context=300k,...]`) into the exact Cursor ACP model and
 * context option to apply.
 */
export function resolveAcpModelSelection(
  requested: string,
  available: readonly AcpModelOption[],
  currentModelId?: string,
): ResolvedAcpModelTarget {
  const raw = requested.trim()
  const base = stripModelParams(raw)
  const explicitCtx = extractExplicitContextValue(raw)
  const currentBase = currentModelId ? stripModelParams(currentModelId) : undefined

  if (!base || base === 'auto') {
    const autoModel = available.find(m => m.acpModelId.startsWith('auto'))
    if (currentBase && currentBase.startsWith('auto')) {
      return { acpModelId: null, contextValue: explicitCtx }
    }
    return { acpModelId: autoModel ? autoModel.acpModelId : null, contextValue: explicitCtx }
  }

  const exact = available.find(m => m.modelId === raw)
  if (exact) {
    return {
      acpModelId: currentBase === exact.acpModelId ? null : exact.acpModelId,
      contextValue: exact.contextOptionValue ?? explicitCtx,
    }
  }

  const byBase = available.find(m => m.acpModelId === base && (explicitCtx === undefined || m.contextOptionValue === explicitCtx))
    ?? available.find(m => m.acpModelId === base)
  if (byBase) {
    return {
      acpModelId: currentBase === byBase.acpModelId ? null : byBase.acpModelId,
      contextValue: explicitCtx ?? byBase.contextOptionValue,
    }
  }

  const byName = available.find(m => m.name.toLowerCase() === raw.toLowerCase() || m.name.toLowerCase() === base.toLowerCase())
  if (byName) {
    return {
      acpModelId: currentBase === byName.acpModelId ? null : byName.acpModelId,
      contextValue: explicitCtx ?? byName.contextOptionValue,
    }
  }

  const byPrefix = available.find(m => m.acpModelId.toLowerCase().startsWith(base.toLowerCase()))
  if (byPrefix) {
    return {
      acpModelId: currentBase === byPrefix.acpModelId ? null : byPrefix.acpModelId,
      contextValue: explicitCtx ?? byPrefix.contextOptionValue,
    }
  }

  return {
    acpModelId: currentBase === base ? null : base,
    contextValue: explicitCtx,
  }
}

/** Backwards-compatible wrapper returning only the target ACP model ID. */
export function resolveAcpModelId(
  requested: string,
  available: readonly AcpModelOption[],
  currentModelId?: string,
): string | null {
  return resolveAcpModelSelection(requested, available, currentModelId).acpModelId
}

interface RawConfigOption {
  id?: unknown
  name?: unknown
  category?: unknown
  currentValue?: unknown
  options?: unknown
}

interface ParsedContextChoice {
  readonly value: string
  readonly name: string
}

function parseConfigOptionsMeta(configOptions: unknown): {
  readonly effortConfigId?: string | undefined
  readonly efforts: readonly AcpEffortOption[]
  readonly defaultEffort?: string | undefined
  readonly contextChoices: readonly ParsedContextChoice[]
  readonly defaultContextValue?: string | undefined
} {
  if (!Array.isArray(configOptions)) {
    return { efforts: [], contextChoices: [] }
  }
  const items = configOptions as RawConfigOption[]

  // Context window options (e.g. 300k / 1m, 272k / 1m, 256k / 500k)
  const contextOpt = items.find(c => c?.id === 'context')
  const contextChoices: ParsedContextChoice[] = []
  const seenCtx = new Set<string>()
  if (contextOpt && Array.isArray(contextOpt.options)) {
    for (const entry of contextOpt.options) {
      if (typeof entry !== 'object' || entry === null) continue
      const rec = entry as { value?: unknown; name?: unknown }
      if (typeof rec.value !== 'string' || !rec.value || seenCtx.has(rec.value)) continue
      seenCtx.add(rec.value)
      const label = typeof rec.name === 'string' && cleanLabel(rec.name)
        ? cleanLabel(rec.name)
        : rec.value.toUpperCase()
      contextChoices.push({ value: rec.value, name: label })
    }
  }
  const defaultContextValue = typeof contextOpt?.currentValue === 'string' && contextOpt.currentValue
    ? contextOpt.currentValue
    : contextChoices[0]?.value

  // Reasoning effort / Auto router tier options
  const effortOpt = items.find(c => c?.category === 'thought_level')
    ?? items.find(c => c?.id === 'optimize_for')

  if (!effortOpt || typeof effortOpt.id !== 'string' || !Array.isArray(effortOpt.options)) {
    return { efforts: [], contextChoices, defaultContextValue }
  }

  const seen = new Set<string>()
  const efforts: AcpEffortOption[] = []
  for (const entry of effortOpt.options) {
    if (typeof entry !== 'object' || entry === null) continue
    const rec = entry as { value?: unknown; name?: unknown }
    if (typeof rec.value !== 'string' || !rec.value || seen.has(rec.value)) continue
    seen.add(rec.value)
    const label = typeof rec.name === 'string' && cleanLabel(rec.name) ? cleanLabel(rec.name) : rec.value
    efforts.push({ id: rec.value, name: label })
  }

  if (efforts.length === 0) {
    return { efforts: [], contextChoices, defaultContextValue }
  }
  const currentVal = typeof effortOpt.currentValue === 'string' && seen.has(effortOpt.currentValue)
    ? effortOpt.currentValue
    : undefined
  return {
    effortConfigId: effortOpt.id,
    efforts,
    defaultEffort: currentVal,
    contextChoices,
    defaultContextValue,
  }
}

/**
 * Parse `cursor/list_available_models` into DSH model catalog items.
 * Models offering multiple context windows (e.g. 300K & 1M) expand cleanly into one entry
 * per context window (`gemini-4-neon` for default 300K, `gemini-4-neon@1m` for 1M),
 * while preserving the model's full `reasoning.efforts` tier list on every entry.
 */
export function parseCursorParameterizedModels(payload: Record<string, unknown>): AcpModelOption[] {
  const raw = payload.models
  if (!Array.isArray(raw)) return []
  const out: AcpModelOption[] = []
  const seen = new Set<string>()
  for (const item of raw) {
    if (typeof item !== 'object' || item === null) continue
    const rec = item as { value?: unknown; name?: unknown; configOptions?: unknown }
    if (typeof rec.value !== 'string' || !rec.value) continue
    const baseId = rec.value
    const baseName = typeof rec.name === 'string' && cleanLabel(rec.name) ? cleanLabel(rec.name) : baseId
    const {
      effortConfigId,
      efforts,
      defaultEffort,
      contextChoices,
      defaultContextValue,
    } = parseConfigOptionsMeta(rec.configOptions)

    if (contextChoices.length > 1) {
      for (const choice of contextChoices) {
        const isDefaultCtx = choice.value === defaultContextValue
        const modelId = isDefaultCtx ? baseId : `${baseId}@${choice.value}`
        if (seen.has(modelId)) continue
        seen.add(modelId)
        out.push({
          modelId,
          acpModelId: baseId,
          name: `${baseName} (${choice.name})`,
          contextWindow: inferAcpContextWindow(baseId, baseName, choice.value),
          contextOptionValue: choice.value,
          effortConfigId,
          efforts,
          defaultEffort,
        })
      }
      continue
    }

    if (seen.has(baseId)) continue
    seen.add(baseId)
    out.push({
      modelId: baseId,
      acpModelId: baseId,
      name: baseName,
      contextWindow: inferAcpContextWindow(baseId, baseName, defaultContextValue),
      contextOptionValue: defaultContextValue,
      effortConfigId,
      efforts,
      defaultEffort,
    })
  }
  return out
}

function parseFallbackSessionModels(sessionResult: Record<string, unknown>): AcpModelOption[] {
  const modelsObj = sessionResult.models as { availableModels?: unknown } | undefined
  const raw = modelsObj?.availableModels
  if (!Array.isArray(raw)) return []
  const out: AcpModelOption[] = []
  const seen = new Set<string>()
  for (const item of raw) {
    if (typeof item !== 'object' || item === null) continue
    const rec = item as { modelId?: unknown; name?: unknown }
    if (typeof rec.modelId !== 'string' || !rec.modelId || seen.has(rec.modelId)) continue
    seen.add(rec.modelId)
    const name = typeof rec.name === 'string' && cleanLabel(rec.name) ? cleanLabel(rec.name) : rec.modelId
    out.push({
      modelId: rec.modelId,
      acpModelId: stripModelParams(rec.modelId),
      name,
      contextWindow: inferAcpContextWindow(rec.modelId, name),
      efforts: [],
    })
  }
  return out
}

const FALLBACK_ACP_MODELS: readonly AcpModelOption[] = [
  {
    modelId: 'auto-smart',
    acpModelId: 'auto-smart',
    name: 'Auto',
    contextWindow: 200_000,
    effortConfigId: 'optimize_for',
    efforts: [
      { id: 'intelligence', name: 'Intelligence' },
      { id: 'balanced', name: 'Balance' },
      { id: 'cost', name: 'Cost' },
    ],
    defaultEffort: 'balanced',
  },
]

export class PersistentCursorAcpClient {
  private child: ChildProcessWithoutNullStreams | null = null
  private initPromise: Promise<void> | null = null
  private nextId = 1
  private readonly pending = new Map<number, PendingRpc>()
  private readonly sessionListeners = new Map<string, (update: AcpSessionUpdate) => void>()
  private stdoutRemainder = ''
  private stderrTail = ''
  private disposed = false

  private cachedModels: readonly AcpModelOption[] | null = null
  private cachedModelsAt = 0
  private modelsInflight: Promise<readonly AcpModelOption[]> | null = null
  private readonly modelCacheTtlMs = 5 * 60 * 1000
  private configChain: Promise<void> = Promise.resolve()

  constructor(private readonly getLauncher: () => CursorLauncher) {}

  findCachedModel(modelId: string): AcpModelOption | undefined {
    const raw = modelId.trim()
    const base = stripModelParams(raw)
    const explicitCtx = extractExplicitContextValue(raw)
    if (!base || base === 'auto') {
      return this.cachedModels?.find(m => m.acpModelId.startsWith('auto'))
    }
    return this.cachedModels?.find(m => m.modelId === raw)
      ?? this.cachedModels?.find(m => m.acpModelId === base && (explicitCtx === undefined || m.contextOptionValue === explicitCtx))
      ?? this.cachedModels?.find(m => m.acpModelId === base || m.name.toLowerCase() === raw.toLowerCase())
  }

  async listModels(cwd = process.cwd()): Promise<readonly AcpModelOption[]> {
    const now = Date.now()
    if (this.cachedModels !== null && now - this.cachedModelsAt < this.modelCacheTtlMs) {
      return this.cachedModels
    }
    if (this.modelsInflight !== null) {
      return this.modelsInflight
    }
    this.modelsInflight = this.probeModels(cwd).finally(() => {
      this.modelsInflight = null
    })
    return this.modelsInflight
  }

  private async probeModels(cwd: string): Promise<readonly AcpModelOption[]> {
    try {
      await this.ensureConnected()
      try {
        const res = await this.rpc('cursor/list_available_models', {})
        const rich = parseCursorParameterizedModels(res)
        if (rich.length > 0) {
          this.cachedModels = rich
          this.cachedModelsAt = Date.now()
          return rich
        }
      } catch {
        // Fall back to session/new on older Cursor CLI builds.
      }
      const sess = await this.rpc('session/new', { cwd, mcpServers: [] })
      const models = parseFallbackSessionModels(sess)
      if (models.length > 0) {
        this.cachedModels = models
        this.cachedModelsAt = Date.now()
        return models
      }
      return this.cachedModels ?? FALLBACK_ACP_MODELS
    } catch {
      return this.cachedModels ?? FALLBACK_ACP_MODELS
    }
  }

  private withConfigLock<T>(task: () => Promise<T>): Promise<T> {
    const prev = this.configChain
    let release!: () => void
    this.configChain = new Promise<void>((resolve) => {
      release = resolve
    })
    return prev.then(task).finally(() => release())
  }

  async runPrompt(req: AcpPromptRequest): Promise<AcpPromptResult> {
    if (req.signal?.aborted) {
      return { stopReason: 'cancelled' }
    }
    await this.ensureConnected()
    const availableModels = await this.listModels(req.cwd)

    const { sessionId } = await this.withConfigLock(async () => {
      const sess = await this.rpc('session/new', { cwd: req.cwd, mcpServers: [] })
      const sid = typeof sess.sessionId === 'string' ? sess.sessionId : ''
      if (!sid) {
        throw new AcpRpcError('Cursor ACP session/new 未返回有效 sessionId。')
      }

      const modesObj = sess.modes as { currentModeId?: unknown } | undefined
      const currentModeId = typeof modesObj?.currentModeId === 'string' ? modesObj.currentModeId : undefined
      if (currentModeId !== req.mode) {
        await this.rpc('session/set_config_option', {
          sessionId: sid,
          configId: 'mode',
          value: req.mode,
        })
      }

      const modelsObj = sess.models as { currentModelId?: unknown } | undefined
      const currentModelId = typeof modelsObj?.currentModelId === 'string' ? modelsObj.currentModelId : undefined
      const target = resolveAcpModelSelection(req.model, availableModels, currentModelId)
      let latestConfigOptions: unknown = sess.configOptions

      if (target.acpModelId !== null) {
        const modelResp = await this.rpc('session/set_config_option', {
          sessionId: sid,
          configId: 'model',
          value: target.acpModelId,
        })
        latestConfigOptions = modelResp.configOptions ?? latestConfigOptions
      }

      // Apply selected context window size (e.g. 300k / 1m / 500k) when supported by the model.
      if (target.contextValue) {
        const liveMeta = parseConfigOptionsMeta(latestConfigOptions)
        if (
          liveMeta.contextChoices.some(c => c.value === target.contextValue)
          && liveMeta.defaultContextValue !== target.contextValue
        ) {
          const ctxResp = await this.rpc('session/set_config_option', {
            sessionId: sid,
            configId: 'context',
            value: target.contextValue,
          })
          latestConfigOptions = ctxResp.configOptions ?? latestConfigOptions
        }
      }

      // Apply selected reasoning effort / Auto router tier.
      if (req.reasoningEffort) {
        const liveMeta = parseConfigOptionsMeta(latestConfigOptions)
        const cachedMeta = this.findCachedModel(req.model)
        const effortConfigId = liveMeta.effortConfigId ?? cachedMeta?.effortConfigId
        const validEfforts = liveMeta.efforts.length > 0 ? liveMeta.efforts : (cachedMeta?.efforts ?? [])
        if (effortConfigId && validEfforts.some(e => e.id === req.reasoningEffort)) {
          await this.rpc('session/set_config_option', {
            sessionId: sid,
            configId: effortConfigId,
            value: req.reasoningEffort,
          })
        }
      }

      return { sessionId: sid }
    })

    if (req.signal?.aborted) {
      return { stopReason: 'cancelled' }
    }

    this.sessionListeners.set(sessionId, req.onUpdate)
    const onAbort = (): void => {
      this.sendNotification('session/cancel', { sessionId })
    }
    req.signal?.addEventListener('abort', onAbort, { once: true })

    try {
      const result = await this.rpc('session/prompt', {
        sessionId,
        prompt: [{ type: 'text', text: req.prompt }],
      })
      const stopReason = typeof result.stopReason === 'string' ? result.stopReason : 'end_turn'
      return { stopReason }
    } finally {
      req.signal?.removeEventListener('abort', onAbort)
      this.sessionListeners.delete(sessionId)
    }
  }

  private async ensureConnected(): Promise<void> {
    if (this.disposed) {
      throw new AcpRpcError('Cursor ACP 客户端已关闭。')
    }
    if (this.child !== null && this.child.exitCode === null && !this.child.killed && this.initPromise !== null) {
      return this.initPromise
    }
    this.teardownChild(new AcpRpcError('正在重建 Cursor ACP 连接…'))
    this.initPromise = this.spawnAndInitialize().catch((error: unknown) => {
      this.initPromise = null
      throw error
    })
    return this.initPromise
  }

  private async spawnAndInitialize(): Promise<void> {
    const launcher = this.getLauncher()
    const child = spawn(launcher.command, [...launcher.prefixArgs, 'acp'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, ...launcher.env },
      windowsHide: true,
    })

    this.child = child
    this.stdoutRemainder = ''
    this.stderrTail = ''

    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => {
      this.stdoutRemainder += chunk
      let nl = this.stdoutRemainder.indexOf('\n')
      while (nl >= 0) {
        const line = this.stdoutRemainder.slice(0, nl)
        this.stdoutRemainder = this.stdoutRemainder.slice(nl + 1)
        this.handleIncomingLine(line)
        nl = this.stdoutRemainder.indexOf('\n')
      }
    })

    child.stderr.setEncoding('utf8')
    child.stderr.on('data', (chunk: string) => {
      if (this.stderrTail.length < 4096) this.stderrTail += chunk
    })

    child.stdin.on('error', () => {
      // Handled by exit/error listeners.
    })

    child.on('error', (error: Error) => {
      if (this.child === child) {
        this.teardownChild(new AcpRpcError(`无法启动 Cursor ACP (${launcher.command})：${error.message}`))
      }
    })

    child.on('close', (code) => {
      if (this.child === child) {
        const detail = this.stderrTail.trim()
        const msg = detail
          ? `Cursor ACP 进程已退出（code=${String(code)}）：${detail}`
          : `Cursor ACP 进程已退出（code=${String(code)}）。`
        this.teardownChild(new AcpRpcError(msg))
      }
    })

    await this.rpc('initialize', {
      protocolVersion: 1,
      clientCapabilities: {
        _meta: {
          parameterizedModelPicker: true,
        },
      },
      clientInfo: { name: 'dsh-llm-cursor-cli', version: '0.2.3' },
    })
  }

  private handleIncomingLine(rawLine: string): void {
    const line = rawLine.trim()
    if (!line) return
    let msg: JsonRpcMessage
    try {
      msg = JSON.parse(line) as JsonRpcMessage
    } catch {
      return
    }

    if (typeof msg.id === 'number' && msg.method === undefined) {
      const waiter = this.pending.get(msg.id)
      if (waiter !== undefined) {
        this.pending.delete(msg.id)
        if (msg.error) {
          const dataMsg = typeof (msg.error.data as { message?: unknown } | undefined)?.message === 'string'
            ? (msg.error.data as { message: string }).message
            : ''
          const combined = dataMsg ? `${msg.error.message ?? 'ACP Error'}: ${dataMsg}` : (msg.error.message ?? 'ACP Error')
          waiter.reject(new AcpRpcError(combined, msg.error.code, msg.error.data))
        } else {
          waiter.resolve(msg.result ?? {})
        }
        return
      }
    }

    if (msg.method === 'session/update' && msg.params) {
      const sid = typeof msg.params.sessionId === 'string' ? msg.params.sessionId : ''
      const update = msg.params.update as AcpSessionUpdate | undefined
      if (sid && update && typeof update.sessionUpdate === 'string') {
        const listener = this.sessionListeners.get(sid)
        listener?.(update)
      }
      return
    }

    if (msg.method === 'session/request_permission' && msg.id !== undefined && msg.id !== null) {
      this.writeJson({
        jsonrpc: '2.0',
        id: msg.id,
        result: { outcome: { outcome: 'cancelled' } },
      })
      return
    }

    if (typeof msg.method === 'string' && msg.id !== undefined && msg.id !== null) {
      this.writeJson({
        jsonrpc: '2.0',
        id: msg.id,
        error: { code: -32601, message: `Method not implemented by read-only LLM ACP client: ${msg.method}` },
      })
    }
  }

  private rpc(method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    const child = this.child
    if (child === null || child.exitCode !== null || child.killed) {
      return Promise.reject(new AcpRpcError('Cursor ACP 连接未就绪。'))
    }
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      const ok = this.writeJson({ jsonrpc: '2.0', id, method, params })
      if (!ok) {
        this.pending.delete(id)
        reject(new AcpRpcError('写入 Cursor ACP stdin 失败。'))
      }
    })
  }

  private sendNotification(method: string, params: Record<string, unknown>): void {
    if (this.child === null || this.child.exitCode !== null || this.child.killed) return
    this.writeJson({ jsonrpc: '2.0', method, params })
  }

  private writeJson(payload: object): boolean {
    const child = this.child
    if (child === null || child.stdin.destroyed || !child.stdin.writable) return false
    try {
      child.stdin.write(`${JSON.stringify(payload)}\n`, 'utf8')
      return true
    } catch {
      return false
    }
  }

  private teardownChild(reason: Error): void {
    const old = this.child
    this.child = null
    this.initPromise = null
    for (const waiter of this.pending.values()) {
      waiter.reject(reason)
    }
    this.pending.clear()
    this.sessionListeners.clear()
    if (old !== null) {
      try {
        old.stdin.end()
      } catch {
        // Ignore
      }
      try {
        if (old.exitCode === null && !old.killed) old.kill()
      } catch {
        // Ignore
      }
    }
  }

  dispose(): void {
    this.disposed = true
    this.teardownChild(new AcpRpcError('Cursor ACP 客户端已释放。'))
  }
}
