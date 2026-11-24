/**
 * Persistent JSON-RPC 2.0 ACP client for WorkBuddy's bundled CLI
 * (`codebuddy --serve --acp --acp-transport stdio --port 0 --tools "" --no-session-persistence`).
 *
 * Key traits specific to WorkBuddy ACP:
 * - Filters out the initial non-JSON HTTP Server startup banner lines on stdout.
 * - Reads live models + exact `_meta.maxInputTokens` directly from `session/new`.
 * - Detects structured `_meta['codebuddy.ai/errorMessage']` auth errors when `stopReason === 'refusal'`.
 * - Supports optional interactive ACP `authenticate({ methodId })` (iOA / internal / external / selfhosted)
 *   on demand when configured, without ever touching WorkBuddy Desktop's encrypted `~/.workbuddy`.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import type { WorkBuddyLauncher } from './launcher.ts'

export interface WorkBuddyEffortOption {
  readonly id: string
  readonly name: string
}

export interface WorkBuddyAcpModel {
  readonly modelId: string
  readonly name: string
  readonly maxInputTokens: number
  readonly efforts: readonly WorkBuddyEffortOption[]
  readonly defaultEffort?: string | undefined
}

export interface WorkBuddySessionUpdate {
  readonly sessionUpdate: string
  readonly content?: {
    readonly type?: string
    readonly text?: string
  }
}

export interface WorkBuddyPromptRequest {
  readonly model: string
  readonly reasoningEffort?: string | undefined
  readonly prompt: string
  readonly cwd: string
  readonly signal?: AbortSignal | undefined
  readonly onUpdate: (update: WorkBuddySessionUpdate) => void
}

export interface WorkBuddyPromptResult {
  readonly stopReason: string
  readonly errorMessage?: string | undefined
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

export class WorkBuddyAcpError extends Error {
  constructor(
    message: string,
    readonly rpcCode?: number,
    readonly isAuthRequired: boolean = false,
  ) {
    super(message)
    this.name = 'WorkBuddyAcpError'
  }
}

export function parseWorkBuddyModels(sessionResult: Record<string, unknown>): WorkBuddyAcpModel[] {
  const modelsObj = sessionResult.models as { availableModels?: unknown } | undefined
  const raw = modelsObj?.availableModels
  if (!Array.isArray(raw)) return []
  const out: WorkBuddyAcpModel[] = []
  const seen = new Set<string>()
  for (const item of raw) {
    if (typeof item !== 'object' || item === null) continue
    const rec = item as {
      modelId?: unknown
      name?: unknown
      _meta?: { maxInputTokens?: unknown; effortOptions?: unknown }
    }
    if (typeof rec.modelId !== 'string' || !rec.modelId || seen.has(rec.modelId)) continue
    seen.add(rec.modelId)
    const name = typeof rec.name === 'string' && rec.name.trim() ? rec.name.trim() : rec.modelId
    const metaMax = rec._meta?.maxInputTokens
    const maxInputTokens = typeof metaMax === 'number' && Number.isFinite(metaMax) && metaMax > 0
      ? Math.floor(metaMax)
      : 200_000
    const efforts: WorkBuddyEffortOption[] = []
    const seenEff = new Set<string>()
    if (Array.isArray(rec._meta?.effortOptions)) {
      for (const opt of rec._meta.effortOptions) {
        if (typeof opt !== 'object' || opt === null) continue
        const o = opt as { value?: unknown; name?: unknown }
        if (typeof o.value !== 'string' || !o.value || seenEff.has(o.value)) continue
        seenEff.add(o.value)
        efforts.push({
          id: o.value,
          name: typeof o.name === 'string' && o.name.trim() ? o.name.trim() : o.value,
        })
      }
    }
    const defaultEffort = efforts.some(e => e.id === 'high')
      ? 'high'
      : efforts.some(e => e.id === 'medium')
        ? 'medium'
        : efforts[0]?.id
    out.push({ modelId: rec.modelId, name, maxInputTokens, efforts, defaultEffort })
  }
  return out
}

export function extractWorkBuddyPromptError(result: Record<string, unknown>): {
  readonly message: string
  readonly isAuth: boolean
} | null {
  const meta = result._meta as Record<string, unknown> | undefined
  const rawErr = meta?.['codebuddy.ai/errorMessage']
  if (typeof rawErr !== 'string' || !rawErr.trim()) return null
  try {
    const parsed = JSON.parse(rawErr) as {
      code?: unknown
      message?: unknown
      data?: { details?: unknown; category?: unknown }
    }
    const details = typeof parsed.data?.details === 'string' ? parsed.data.details : ''
    const baseMsg = typeof parsed.message === 'string' ? parsed.message : rawErr
    const combined = details || baseMsg
    const isAuth = parsed.data?.category === 'auth'
      || parsed.code === -32000
      || combined.toLowerCase().includes('authentication required')
    return { message: combined, isAuth }
  } catch {
    const isAuth = rawErr.toLowerCase().includes('authentication required')
    return { message: rawErr, isAuth }
  }
}

export function resolveWorkBuddyModelId(
  requested: string,
  available: readonly WorkBuddyAcpModel[],
): string | null {
  const trimmed = requested.trim()
  if (!trimmed || trimmed === 'default') return null
  const exact = available.find(m => m.modelId === trimmed)
  if (exact) return exact.modelId
  const byName = available.find(m => m.name.toLowerCase() === trimmed.toLowerCase())
  if (byName) return byName.modelId
  return trimmed
}

const FALLBACK_WB_MODELS: readonly WorkBuddyAcpModel[] = [
  { modelId: 'hy3-preview-ioa', name: 'Hy3 preview', maxInputTokens: 192_000, efforts: [] },
  { modelId: 'glm-5.1-ioa', name: 'GLM-5.1', maxInputTokens: 200_000, efforts: [] },
  { modelId: 'claude-opus-4.7-1m', name: 'Claude Opus 4.7 1M', maxInputTokens: 1_000_000, efforts: [] },
]

export class PersistentWorkBuddyAcpClient {
  private child: ChildProcessWithoutNullStreams | null = null
  private initPromise: Promise<void> | null = null
  private authPromise: Promise<void> | null = null
  private nextId = 1
  private readonly pending = new Map<number, PendingRpc>()
  private readonly sessionListeners = new Map<string, (update: WorkBuddySessionUpdate) => void>()
  private stdoutRemainder = ''
  private stderrTail = ''
  private disposed = false

  private cachedModels: readonly WorkBuddyAcpModel[] | null = null
  private cachedModelsAt = 0
  private modelsInflight: Promise<readonly WorkBuddyAcpModel[]> | null = null
  private readonly modelCacheTtlMs = 5 * 60 * 1000

  constructor(
    private readonly getLauncher: () => WorkBuddyLauncher,
    private readonly autoAuthMethod: string = '',
  ) {}

  findCachedModel(modelId: string): WorkBuddyAcpModel | undefined {
    return this.cachedModels?.find(m => m.modelId === modelId || m.name === modelId)
  }

  async listModels(cwd = process.cwd()): Promise<readonly WorkBuddyAcpModel[]> {
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

  private async probeModels(cwd: string): Promise<readonly WorkBuddyAcpModel[]> {
    const launcher = this.getLauncher()
    if (!launcher.available) return []
    try {
      await this.ensureConnected()
      const sess = await this.rpc('session/new', { cwd, mcpServers: [] })
      const models = parseWorkBuddyModels(sess)
      if (models.length > 0) {
        this.cachedModels = models
        this.cachedModelsAt = Date.now()
        return models
      }
      return this.cachedModels ?? FALLBACK_WB_MODELS
    } catch {
      return this.cachedModels ?? FALLBACK_WB_MODELS
    }
  }

  async runPrompt(req: WorkBuddyPromptRequest): Promise<WorkBuddyPromptResult> {
    if (req.signal?.aborted) {
      return { stopReason: 'cancelled' }
    }
    const launcher = this.getLauncher()
    if (!launcher.available) {
      throw new WorkBuddyAcpError(launcher.error ?? 'WorkBuddy CLI 未安装。')
    }
    await this.ensureConnected()

    const firstAttempt = await this.executeSingleSession(req)
    if (firstAttempt.authRequired && this.autoAuthMethod) {
      await this.authenticateOnce(this.autoAuthMethod)
      const retry = await this.executeSingleSession(req)
      if (retry.authRequired) {
        throw new WorkBuddyAcpError(retry.errorMessage ?? 'WorkBuddy ACP 认证未完成。', -32000, true)
      }
      return { stopReason: retry.stopReason, errorMessage: retry.errorMessage }
    }
    if (firstAttempt.authRequired) {
      throw new WorkBuddyAcpError(
        firstAttempt.errorMessage ?? 'WorkBuddy ACP 需要登录认证。',
        -32000,
        true,
      )
    }
    return { stopReason: firstAttempt.stopReason, errorMessage: firstAttempt.errorMessage }
  }

  private async authenticateOnce(methodId: string): Promise<void> {
    if (this.authPromise !== null) return this.authPromise
    this.authPromise = this.rpc('authenticate', { methodId }).then(() => undefined).finally(() => {
      this.authPromise = null
    })
    return this.authPromise
  }

  private async executeSingleSession(req: WorkBuddyPromptRequest): Promise<{
    readonly stopReason: string
    readonly errorMessage?: string | undefined
    readonly authRequired: boolean
  }> {
    const sess = await this.rpc('session/new', { cwd: req.cwd, mcpServers: [] })
    const sessionId = typeof sess.sessionId === 'string' ? sess.sessionId : ''
    if (!sessionId) {
      throw new WorkBuddyAcpError('WorkBuddy ACP session/new 未返回有效 sessionId。')
    }

    const availableModels = parseWorkBuddyModels(sess)
    if (availableModels.length > 0) {
      this.cachedModels = availableModels
      this.cachedModelsAt = Date.now()
    }

    const targetModelId = resolveWorkBuddyModelId(req.model, availableModels)
    if (targetModelId !== null) {
      await this.rpc('session/set_config_option', {
        sessionId,
        configId: 'model',
        value: targetModelId,
      })
    }

    if (req.reasoningEffort) {
      await this.rpc('session/set_config_option', {
        sessionId,
        configId: 'thought_level',
        value: req.reasoningEffort,
      })
    }

    if (req.signal?.aborted) {
      return { stopReason: 'cancelled', authRequired: false }
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
      const errInfo = extractWorkBuddyPromptError(result)
      if (errInfo !== null) {
        return {
          stopReason,
          errorMessage: errInfo.message,
          authRequired: errInfo.isAuth,
        }
      }
      return { stopReason, authRequired: false }
    } finally {
      req.signal?.removeEventListener('abort', onAbort)
      this.sessionListeners.delete(sessionId)
    }
  }

  private async ensureConnected(): Promise<void> {
    if (this.disposed) {
      throw new WorkBuddyAcpError('WorkBuddy ACP 客户端已关闭。')
    }
    if (this.child !== null && this.child.exitCode === null && !this.child.killed && this.initPromise !== null) {
      return this.initPromise
    }
    this.teardownChild(new WorkBuddyAcpError('正在重建 WorkBuddy ACP 连接…'))
    this.initPromise = this.spawnAndInitialize().catch((error: unknown) => {
      this.initPromise = null
      throw error
    })
    return this.initPromise
  }

  private async spawnAndInitialize(): Promise<void> {
    const launcher = this.getLauncher()
    if (!launcher.available) {
      throw new WorkBuddyAcpError(launcher.error ?? 'WorkBuddy CLI 未安装。')
    }
    const child = spawn(
      launcher.nodeExecutable,
      [
        launcher.cliScript,
        '--serve',
        '--acp',
        '--acp-transport', 'stdio',
        '--port', '0',
        '--tools', '',
        '--no-session-persistence',
      ],
      {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, ...launcher.env },
        windowsHide: true,
      },
    )

    this.child = child
    this.stdoutRemainder = ''
    this.stderrTail = ''

    let resolveBannerReady: (() => void) | null = null
    const bannerReady = new Promise<void>((resolve) => {
      resolveBannerReady = resolve
    })
    const bannerTimer = setTimeout(() => {
      resolveBannerReady?.()
      resolveBannerReady = null
    }, 3500)

    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => {
      this.stdoutRemainder += chunk
      let nl = this.stdoutRemainder.indexOf('\n')
      while (nl >= 0) {
        const line = this.stdoutRemainder.slice(0, nl)
        this.stdoutRemainder = this.stdoutRemainder.slice(nl + 1)
        const trimmed = line.trim()
        if (!trimmed) {
          nl = this.stdoutRemainder.indexOf('\n')
          continue
        }
        if (!trimmed.startsWith('{')) {
          if (trimmed.includes('Press Ctrl+C') || trimmed.includes('Endpoint')) {
            clearTimeout(bannerTimer)
            resolveBannerReady?.()
            resolveBannerReady = null
          }
          nl = this.stdoutRemainder.indexOf('\n')
          continue
        }
        this.handleIncomingLine(trimmed)
        nl = this.stdoutRemainder.indexOf('\n')
      }
    })

    child.stderr.setEncoding('utf8')
    child.stderr.on('data', (chunk: string) => {
      if (this.stderrTail.length < 4096) this.stderrTail += chunk
    })

    child.stdin.on('error', () => {
      // Handled via exit/error events.
    })

    child.on('error', (error: Error) => {
      clearTimeout(bannerTimer)
      resolveBannerReady?.()
      resolveBannerReady = null
      if (this.child === child) {
        this.teardownChild(new WorkBuddyAcpError(`无法启动 WorkBuddy ACP：${error.message}`))
      }
    })

    child.on('close', (code) => {
      clearTimeout(bannerTimer)
      resolveBannerReady?.()
      resolveBannerReady = null
      if (this.child === child) {
        const detail = this.stderrTail.trim()
        const msg = detail
          ? `WorkBuddy ACP 进程已退出（code=${String(code)}）：${detail}`
          : `WorkBuddy ACP 进程已退出（code=${String(code)}）。`
        this.teardownChild(new WorkBuddyAcpError(msg))
      }
    })

    await bannerReady
    if (this.child !== child || child.exitCode !== null) {
      throw new WorkBuddyAcpError('WorkBuddy ACP 在初始化前已退出。')
    }
    // Brief settle window after HTTP banner prints while acpServer registers stdio transport.
    await new Promise(resolve => setTimeout(resolve, 250))

    await this.rpc('initialize', {
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: 'dsh-llm-workbuddy-acp', version: '0.2.3' },
    })
  }

  private handleIncomingLine(line: string): void {
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
          const dataMsg = typeof (msg.error.data as { details?: unknown; message?: unknown } | undefined)?.details === 'string'
            ? (msg.error.data as { details: string }).details
            : typeof (msg.error.data as { message?: unknown } | undefined)?.message === 'string'
              ? (msg.error.data as { message: string }).message
              : ''
          const combined = dataMsg || msg.error.message || 'WorkBuddy ACP Error'
          const isAuth = msg.error.code === -32000 || combined.toLowerCase().includes('authentication required')
          waiter.reject(new WorkBuddyAcpError(combined, msg.error.code, isAuth))
        } else {
          waiter.resolve(msg.result ?? {})
        }
        return
      }
    }

    if (msg.method === 'session/update' && msg.params) {
      const sid = typeof msg.params.sessionId === 'string' ? msg.params.sessionId : ''
      const update = msg.params.update as WorkBuddySessionUpdate | undefined
      if (sid && update && typeof update.sessionUpdate === 'string') {
        this.sessionListeners.get(sid)?.(update)
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
      return Promise.reject(new WorkBuddyAcpError('WorkBuddy ACP 连接未就绪。'))
    }
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      const ok = this.writeJson({ jsonrpc: '2.0', id, method, params })
      if (!ok) {
        this.pending.delete(id)
        reject(new WorkBuddyAcpError('写入 WorkBuddy ACP stdin 失败。'))
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
    this.teardownChild(new WorkBuddyAcpError('WorkBuddy ACP 客户端已释放。'))
  }
}
