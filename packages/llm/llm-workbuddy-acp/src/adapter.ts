/**
 * LlmAdapter backed by a persistent local WorkBuddy ACP connection.
 */
import { randomUUID } from 'node:crypto'
import { brandString } from '@deepseek-ai/dsh-brand'
import {
  LlmAdapter,
  LlmError,
  type GenerateOptions,
  type LlmModelInfo,
  type LlmProviderInfo,
  type LlmResolvedModelInfo,
  type ReasoningEffortId,
  type StreamChunk,
  type ToolCallId,
} from '@deepseek-ai/dsh-llm'
import {
  PersistentWorkBuddyAcpClient,
  WorkBuddyAcpError,
  type WorkBuddySessionUpdate,
} from './acp-client.ts'
import type { WorkBuddyLauncher } from './launcher.ts'
import { buildWorkBuddyPrompt, ToolShimParser } from './prompt.ts'

export interface WorkBuddyAdapterOptions {
  readonly providerId: string
  readonly displayName: string
  readonly enableToolShim: boolean
  readonly autoAuthMethod: string
  readonly workspaceDir?: string
  readonly getLauncher: () => WorkBuddyLauncher
}

function classifyWorkBuddyError(error: unknown, configDirHint: string): LlmError {
  if (error instanceof WorkBuddyAcpError && error.isAuthRequired) {
    return new LlmError(
      `WorkBuddy ACP 尚未在独立 CLI 目录（${configDirHint}）完成登录，请设置环境变量 CODEBUDDY_AUTH_TOKEN / CODEBUDDY_API_KEY，或在配置中启用 autoAuthMethod（如 iOA）完成一次登录：${error.message}`,
      'AUTH',
      { cause: error },
    )
  }
  const message = error instanceof Error ? error.message : String(error)
  const lower = message.toLowerCase()
  if (lower.includes('authentication required') || lower.includes('unauthorized') || lower.includes('login')) {
    return new LlmError(
      `WorkBuddy ACP 认证失效或未登录（目录 ${configDirHint}）：${message}`,
      'AUTH',
      error instanceof Error ? { cause: error } : undefined,
    )
  }
  if (lower.includes('rate limit') || lower.includes('too many requests')) {
    return new LlmError(`WorkBuddy ACP 触发限流：${message}`, 'RATE_LIMIT', error instanceof Error ? { cause: error } : undefined)
  }
  if (lower.includes('quota') || lower.includes('credit')) {
    return new LlmError(`WorkBuddy ACP 额度不足：${message}`, 'QUOTA', error instanceof Error ? { cause: error } : undefined)
  }
  if (error instanceof WorkBuddyAcpError && (lower.includes('无法启动') || lower.includes('已退出') || lower.includes('未安装'))) {
    return new LlmError(message, 'TRANSPORT', { cause: error })
  }
  return new LlmError(`WorkBuddy ACP 请求失败：${message}`, 'UNKNOWN', error instanceof Error ? { cause: error } : undefined)
}

export class WorkBuddyAcpLlmAdapter extends LlmAdapter {
  private readonly acpClient: PersistentWorkBuddyAcpClient

  constructor(private readonly opts: WorkBuddyAdapterOptions) {
    super()
    this.acpClient = new PersistentWorkBuddyAcpClient(opts.getLauncher, opts.autoAuthMethod)
  }

  dispose(): void {
    this.acpClient.dispose()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: this.opts.displayName }
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const models = await this.acpClient.listModels(this.opts.workspaceDir || process.cwd())
    return models.map(item => ({
      provider,
      id: item.modelId,
      name: item.name,
    }))
  }

  override async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    let known = this.acpClient.findCachedModel(model)
    if (known === undefined) {
      await this.acpClient.listModels(this.opts.workspaceDir || process.cwd())
      known = this.acpClient.findCachedModel(model)
    }
    const efforts = known?.efforts ?? []
    const reasoning = efforts.length > 0
      ? {
        efforts: efforts.map(e => ({
          id: brandString<ReasoningEffortId>(e.id),
          name: e.name,
        })),
        ...known?.defaultEffort !== undefined
          ? { defaultEffort: brandString<ReasoningEffortId>(known.defaultEffort) }
          : {},
      }
      : undefined
    return {
      provider,
      id: model,
      name: known?.name ?? model,
      context: { contextWindow: known?.maxInputTokens ?? 200_000 },
      defaultMaxTokens: 16_384,
      ...reasoning !== undefined ? { reasoning } : {},
    }
  }

  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    if (options.signal?.aborted) {
      yield {
        type: 'finish',
        reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'WorkBuddy ACP request aborted before launch.' } },
      }
      return
    }

    const allowedTools = new Set<string>(
      this.opts.enableToolShim ? (options.tools ?? []).map(t => t.name) : [],
    )
    const prompt = buildWorkBuddyPrompt(options, { enableToolShim: this.opts.enableToolShim })

    const queue: StreamChunk[] = []
    let notify: (() => void) | null = null
    let done = false
    let stopReason = 'end_turn'
    let promptErrorMessage: string | undefined
    let failure: unknown = null

    const wake = (): void => {
      if (notify !== null) {
        const fn = notify
        notify = null
        fn()
      }
    }

    const pushChunk = (chunk: StreamChunk): void => {
      queue.push(chunk)
      wake()
    }

    let nextBlockIndex = 0
    let textBlockIndex: number | null = null
    let textAccum = ''
    let reasoningBlockIndex: number | null = null
    let reasoningAccum = ''
    let emittedToolCalls = 0

    const closeReasoningBlock = (): void => {
      if (reasoningBlockIndex !== null) {
        pushChunk({ type: 'block-end', index: reasoningBlockIndex, block: { type: 'reasoning', text: reasoningAccum } })
        reasoningBlockIndex = null
        reasoningAccum = ''
      }
    }

    const closeTextBlock = (): void => {
      if (textBlockIndex !== null) {
        pushChunk({ type: 'block-end', index: textBlockIndex, block: { type: 'text', text: textAccum } })
        textBlockIndex = null
        textAccum = ''
      }
    }

    const emitText = (delta: string): void => {
      if (!delta) return
      closeReasoningBlock()
      if (textBlockIndex === null) {
        textBlockIndex = nextBlockIndex++
        textAccum = ''
        pushChunk({ type: 'block-start', index: textBlockIndex, blockType: 'text' })
      }
      textAccum += delta
      pushChunk({ type: 'text-delta', index: textBlockIndex, text: delta })
    }

    const emitReasoning = (delta: string): void => {
      if (!delta) return
      closeTextBlock()
      if (reasoningBlockIndex === null) {
        reasoningBlockIndex = nextBlockIndex++
        reasoningAccum = ''
        pushChunk({ type: 'block-start', index: reasoningBlockIndex, blockType: 'reasoning' })
      }
      reasoningAccum += delta
      pushChunk({ type: 'reasoning-delta', index: reasoningBlockIndex, text: delta })
    }

    const parser = new ToolShimParser(allowedTools, {
      onText: emitText,
      onToolCall: (call) => {
        closeReasoningBlock()
        closeTextBlock()
        const idx = nextBlockIndex++
        const id = brandString<ToolCallId>(`wb_tc_${randomUUID().slice(0, 12)}`)
        emittedToolCalls += 1
        pushChunk({ type: 'block-start', index: idx, blockType: 'tool-call' })
        pushChunk({
          type: 'tool-call-delta',
          index: idx,
          id,
          name: call.name,
          argumentsDelta: call.argumentsJson,
        })
        pushChunk({
          type: 'block-end',
          index: idx,
          block: { type: 'tool-call', id, name: call.name, arguments: call.argumentsJson },
        })
      },
    })

    const handleUpdate = (update: WorkBuddySessionUpdate): void => {
      const text = update.content?.type === 'text' && typeof update.content.text === 'string'
        ? update.content.text
        : ''
      if (!text) return
      if (update.sessionUpdate === 'agent_thought_chunk') {
        emitReasoning(text)
      } else if (update.sessionUpdate === 'agent_message_chunk') {
        parser.push(text)
      }
    }

    const runPromise = this.acpClient.runPrompt({
      model: options.model,
      reasoningEffort: options.reasoningEffort === undefined ? undefined : String(options.reasoningEffort),
      prompt,
      cwd: this.opts.workspaceDir || process.cwd(),
      signal: options.signal,
      onUpdate: handleUpdate,
    }).then(
      (res) => {
        stopReason = res.stopReason
        promptErrorMessage = res.errorMessage
      },
      (err: unknown) => {
        failure = err
      },
    ).finally(() => {
      parser.finish()
      closeReasoningBlock()
      closeTextBlock()
      done = true
      wake()
    })

    while (!done || queue.length > 0) {
      const next = queue.shift()
      if (next !== undefined) {
        yield next
        continue
      }
      await new Promise<void>((resolve) => {
        notify = resolve
      })
    }

    await runPromise

    if (options.signal?.aborted || stopReason === 'cancelled') {
      yield {
        type: 'finish',
        reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'WorkBuddy ACP request cancelled.' } },
      }
      return
    }

    const configDirHint = this.opts.getLauncher().env.CODEBUDDY_CONFIG_DIR ?? '~/.codebuddy'
    if (failure !== null) {
      const err = classifyWorkBuddyError(failure, configDirHint)
      yield { type: 'finish', reason: { kind: 'error', failure: err.failure } }
      return
    }

    if (promptErrorMessage) {
      const err = classifyWorkBuddyError(new WorkBuddyAcpError(promptErrorMessage), configDirHint)
      yield { type: 'finish', reason: { kind: 'error', failure: err.failure } }
      return
    }

    if (stopReason === 'max_tokens') {
      yield { type: 'finish', reason: { kind: 'max-tokens' } }
      return
    }

    yield {
      type: 'finish',
      reason: emittedToolCalls > 0 ? { kind: 'tool-calls' } : { kind: 'stop' },
    }
  }
}
