/**
 * Prompt serialization and streaming `<dsh_tool_call>` parser for WorkBuddy ACP read-only LLM mode.
 */
import type { ContentBlock, GenerateOptions, RequestMessage, ToolSchema } from '@deepseek-ai/dsh-llm'

export const WB_TOOL_CALL_OPEN = '<dsh_tool_call>'
export const WB_TOOL_CALL_CLOSE = '</dsh_tool_call>'

const MAX_TOOL_SCHEMA_CHARS = 1200

function formatBlocks(content: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of content) {
    switch (block.type) {
      case 'text':
        if (block.text) parts.push(block.text)
        break
      case 'reasoning':
        break
      case 'tool-call':
        parts.push(`[Called tool ${block.name} (id=${block.id}) with arguments: ${block.arguments}]`)
        break
      case 'image':
        parts.push(`[Image attachment: ${block.attachment.mediaType}, ${block.attachment.bytes} bytes]`)
        break
      case 'file':
        parts.push(`[File attachment: ${block.attachment.name} (${block.attachment.bytes} bytes)]`)
        break
      default:
        break
    }
  }
  return parts.join('\n').trim()
}

function formatMessage(message: RequestMessage): string {
  if (message.role === 'tool') {
    const body = formatBlocks(message.content)
    const status = message.isError ? ' ERROR' : ''
    return `[TOOL RESULT id=${message.toolCallId}${status}]\n${body}`
  }
  const body = formatBlocks(message.content)
  if (!body) return ''
  return `[${message.role.toUpperCase()}]\n${body}`
}

function summarizeToolSchema(tool: ToolSchema): string {
  const desc = tool.description.replace(/\s+/g, ' ').trim().slice(0, 280)
  let params = '{}'
  try {
    const serialized = JSON.stringify(tool.parameters ?? {})
    params = serialized.length > MAX_TOOL_SCHEMA_CHARS
      ? `${serialized.slice(0, MAX_TOOL_SCHEMA_CHARS)}...`
      : serialized
  } catch {
    params = '{}'
  }
  return `- ${tool.name}: ${desc}\n  parameters: ${params}`
}

function buildToolProtocolSection(tools: readonly ToolSchema[]): string {
  const hasWbSubagent = tools.some(t => t.name === 'subagent_workbuddy')
  const toolLines = tools.map(summarizeToolSchema).join('\n')
  const delegationHint = hasWbSubagent
    ? '\n- For multi-step coding, file editing, or shell command execution, prefer calling `subagent_workbuddy` (native WorkBuddy ACP agent) or the appropriate workspace tool.'
    : ''
  return [
    '[HARNESS TOOL BRIDGE]',
    'You are acting as the reasoning model for DeepSeek Harness (DSH). Your built-in CLI tools are disabled in this turn; instead, DSH provides these host tools:',
    toolLines,
    '',
    'Tool-calling rules:',
    `- If you can answer directly without calling a tool, reply normally in plain text and NEVER emit ${WB_TOOL_CALL_OPEN}.`,
    `- To invoke one or more DSH tools, wrap each call exactly as: ${WB_TOOL_CALL_OPEN}{"name":"<tool_name>","arguments":{...}}${WB_TOOL_CALL_CLOSE}`,
    '- Use valid JSON with declared tool names and arguments, then stop immediately to await [TOOL RESULT].',
    delegationHint,
  ].filter(Boolean).join('\n')
}

export function buildWorkBuddyPrompt(
  options: GenerateOptions,
  { enableToolShim }: { readonly enableToolShim: boolean },
): string {
  const sections: string[] = []
  if (options.system?.trim()) {
    sections.push(`[SYSTEM]\n${options.system.trim()}`)
  }

  const tools = options.tools ?? []
  if (enableToolShim && tools.length > 0) {
    sections.push(buildToolProtocolSection(tools))
  }

  const transcript: string[] = []
  for (const msg of options.messages) {
    const formatted = formatMessage(msg)
    if (formatted) transcript.push(formatted)
  }

  if (transcript.length === 1 && sections.length === 0 && options.messages[0]?.role === 'user') {
    return formatBlocks(options.messages[0].content)
  }

  if (transcript.length > 0) {
    sections.push('[CONVERSATION]\n' + transcript.join('\n\n'))
  }
  sections.push('[INSTRUCTION]\nRespond as [ASSISTANT] to the latest message above. Do not prefix your response with "[ASSISTANT]".')
  return sections.join('\n\n')
}

export interface ParsedToolCall {
  readonly name: string
  readonly argumentsJson: string
}

export interface ToolShimEvents {
  onText(text: string): void
  onToolCall(call: ParsedToolCall): void
}

function longestPrefixSuffix(text: string, needle: string): number {
  const maxLen = Math.min(text.length, needle.length - 1)
  for (let len = maxLen; len > 0; len -= 1) {
    if (text.endsWith(needle.slice(0, len))) return len
  }
  return 0
}

function tryParseToolCall(raw: string, allowedTools: ReadonlySet<string>): ParsedToolCall | null {
  const trimmed = raw.trim()
  if (!trimmed) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    return null
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null
  const record = parsed as Record<string, unknown>
  if (typeof record.name !== 'string' || !allowedTools.has(record.name)) return null
  const args = record.arguments ?? {}
  if (typeof args !== 'object' || args === null || Array.isArray(args)) return null
  return { name: record.name, argumentsJson: JSON.stringify(args) }
}

export class ToolShimParser {
  private buffer = ''
  private inTag = false

  constructor(
    private readonly allowedTools: ReadonlySet<string>,
    private readonly events: ToolShimEvents,
  ) {}

  push(chunk: string): void {
    if (!chunk) return
    if (this.allowedTools.size === 0) {
      this.events.onText(chunk)
      return
    }
    this.buffer += chunk
    this.drain(false)
  }

  finish(): void {
    this.drain(true)
  }

  private drain(final: boolean): void {
    while (this.buffer.length > 0) {
      if (!this.inTag) {
        const openIdx = this.buffer.indexOf(WB_TOOL_CALL_OPEN)
        if (openIdx >= 0) {
          if (openIdx > 0) this.events.onText(this.buffer.slice(0, openIdx))
          this.buffer = this.buffer.slice(openIdx + WB_TOOL_CALL_OPEN.length)
          this.inTag = true
          continue
        }
        if (final) {
          this.events.onText(this.buffer)
          this.buffer = ''
          return
        }
        const hold = longestPrefixSuffix(this.buffer, WB_TOOL_CALL_OPEN)
        const flushLen = this.buffer.length - hold
        if (flushLen > 0) {
          this.events.onText(this.buffer.slice(0, flushLen))
          this.buffer = this.buffer.slice(flushLen)
        }
        return
      }

      const closeIdx = this.buffer.indexOf(WB_TOOL_CALL_CLOSE)
      if (closeIdx >= 0) {
        const body = this.buffer.slice(0, closeIdx)
        this.buffer = this.buffer.slice(closeIdx + WB_TOOL_CALL_CLOSE.length)
        this.inTag = false
        const parsed = tryParseToolCall(body, this.allowedTools)
        if (parsed !== null) {
          this.events.onToolCall(parsed)
        } else {
          this.events.onText(`${WB_TOOL_CALL_OPEN}${body}${WB_TOOL_CALL_CLOSE}`)
        }
        continue
      }

      if (final) {
        this.events.onText(`${WB_TOOL_CALL_OPEN}${this.buffer}`)
        this.buffer = ''
        this.inTag = false
        return
      }
      return
    }
  }
}
