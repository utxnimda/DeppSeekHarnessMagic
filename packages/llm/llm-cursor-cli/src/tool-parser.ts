/**
 * Incremental streaming parser for `<dsh_tool_call>{"name":...,"arguments":...}</dsh_tool_call>`
 * delimiters. Normal text deltas flush immediately unless the tail is a partial prefix
 * of the opening tag, preserving low-latency streaming in the Web UI.
 */
import { TOOL_CALL_CLOSE, TOOL_CALL_OPEN } from './prompt.ts'

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
  const name = record.name
  if (typeof name !== 'string' || !allowedTools.has(name)) return null
  const args = record.arguments ?? {}
  if (typeof args !== 'object' || args === null || Array.isArray(args)) return null
  return { name, argumentsJson: JSON.stringify(args) }
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
        const openIdx = this.buffer.indexOf(TOOL_CALL_OPEN)
        if (openIdx >= 0) {
          if (openIdx > 0) {
            this.events.onText(this.buffer.slice(0, openIdx))
          }
          this.buffer = this.buffer.slice(openIdx + TOOL_CALL_OPEN.length)
          this.inTag = true
          continue
        }
        if (final) {
          this.events.onText(this.buffer)
          this.buffer = ''
          return
        }
        const hold = longestPrefixSuffix(this.buffer, TOOL_CALL_OPEN)
        const flushLen = this.buffer.length - hold
        if (flushLen > 0) {
          this.events.onText(this.buffer.slice(0, flushLen))
          this.buffer = this.buffer.slice(flushLen)
        }
        return
      }

      const closeIdx = this.buffer.indexOf(TOOL_CALL_CLOSE)
      if (closeIdx >= 0) {
        const body = this.buffer.slice(0, closeIdx)
        this.buffer = this.buffer.slice(closeIdx + TOOL_CALL_CLOSE.length)
        this.inTag = false
        const parsed = tryParseToolCall(body, this.allowedTools)
        if (parsed !== null) {
          this.events.onToolCall(parsed)
        } else {
          // Malformed or unknown tool tag: degrade safely to plain text rather than dropping output.
          this.events.onText(`${TOOL_CALL_OPEN}${body}${TOOL_CALL_CLOSE}`)
        }
        continue
      }

      if (final) {
        this.events.onText(`${TOOL_CALL_OPEN}${this.buffer}`)
        this.buffer = ''
        this.inTag = false
        return
      }
      return
    }
  }
}
