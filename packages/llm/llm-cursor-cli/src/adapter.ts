/**
 * LlmAdapter backed by a persistent local Cursor CLI ACP connection (`agent acp`).
 *
 * Every request creates an isolated ACP session over the warm persistent ACP server,
 * switches to read-only `ask` mode (or the configured ACP mode) and target model via
 * `session/set_config_option`, streams `agent_thought_chunk` -> `reasoning-delta` and
 * `agent_message_chunk` -> `text-delta` (with optional `<dsh_tool_call>` parsing), and
 * supports cooperative cancellation via `session/cancel`.
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
  AcpRpcError,
  inferAcpContextWindow,
  PersistentCursorAcpClient,
  type AcpSessionUpdate,
} from './acp-client.ts'
import type { CursorLauncher } from './launcher.ts'
import { buildCursorAskPrompt } from './prompt.ts'
import { ToolShimParser } from './tool-parser.ts'

export interface CursorCliAdapterOptions {
  readonly providerId: string
  readonly displayName: string
  readonly enableToolShim: boolean
  readonly acpMode: 'ask' | 'plan' | 'agent'
  readonly workspaceDir?: string
  readonly getLauncher: () => CursorLauncher
}

function classifyAcpError(error: unknown): LlmError {
  const message = error instanceof Error ? error.message : String(error)
  const lower = message.toLowerCase()
  if (lower.includes('not logged in') || lower.includes('login') || lower.includes('unauthorized') || lower.includes('auth')) {
    return new LlmError(
      `Cursor ACP 未登录或认证已失效，请先在终端执行 "agent login"：${message}`,
      'AUTH',
      error instanceof Error ? { cause: error } : undefined,
    )
  }
  if (lower.includes('invalid model')) {
    return new LlmError(`Cursor ACP 模型不可用：${message}`, 'UNKNOWN_MODEL', error instanceof Error ? { cause: error } : undefined)
  }
  if (lower.includes('rate limit') || lower.includes('too many requests')) {
    return new LlmError(`Cursor ACP 触发限流：${message}`, 'RATE_LIMIT', error instanceof Error ? { cause: error } : undefined)
  }
  if (lower.includes('quota') || lower.includes('usage limit')) {
    return new LlmError(`Cursor ACP 额度不足：${message}`, 'QUOTA', error instanceof Error ? { cause: error } : undefined)
  }
  if (error instanceof AcpRpcError && (lower.includes('无法启动') || lower.includes('已退出') || lower.includes('未就绪'))) {
    return new LlmError(message, 'TRANSPORT', { cause: error })
  }
  return new LlmError(`Cursor ACP 请求失败：${message}`, 'UNKNOWN', error instanceof Error ? { cause: error } : undefined)
}

export class CursorCliLlmAdapter extends LlmAdapter {
  private readonly acpClient: PersistentCursorAcpClient

  constructor(private readonly opts: CursorCliAdapterOptions) {
    super()
    this.acpClient = new PersistentCursorAcpClient(opts.getLauncher)
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
    const name = known?.name ?? (model === 'auto' ? 'Auto' : model)
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
      name,
      context: { contextWindow: known?.contextWindow ?? inferAcpContextWindow(model, name) },
      defaultMaxTokens: 16_384,
      ...reasoning !== undefined ? { reasoning } : {},
    }
  }

  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    if (options.signal?.aborted) {
      yield {
        type: 'finish',
        reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'Cursor ACP request aborted before launch.' } },
      }
      return
    }

    const useToolShim = this.opts.enableToolShim && this.opts.acpMode === 'ask'
    const allowedTools = new Set<string>(
      useToolShim ? (options.tools ?? []).map(t => t.name) : [],
    )
    const prompt = buildCursorAskPrompt(options, { enableToolShim: useToolShim })

    const queue: StreamChunk[] = []
    let notify: (() => void) | null = null
    let done = false
    let stopReason = 'end_turn'
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
        const id = brandString<ToolCallId>(`cursor_tc_${randomUUID().slice(0, 12)}`)
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

    const handleUpdate = (update: AcpSessionUpdate): void => {
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
      model: options.model || 'auto-smart',
      reasoningEffort: options.reasoningEffort === undefined ? undefined : String(options.reasoningEffort),
      mode: this.opts.acpMode,
      prompt,
      cwd: this.opts.workspaceDir || process.cwd(),
      signal: options.signal,
      onUpdate: handleUpdate,
    }).then(
      (res) => {
        stopReason = res.stopReason
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
        reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'Cursor ACP request cancelled.' } },
      }
      return
    }

    if (failure !== null) {
      const err = classifyAcpError(failure)
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
