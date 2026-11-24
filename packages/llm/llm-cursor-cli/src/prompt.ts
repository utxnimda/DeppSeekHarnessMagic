/**
 * Prompt serialization for Cursor CLI `--mode ask`.
 * Preserves full DSH conversation history cleanly and only attaches the
 * delimited tool-call shim when the request declares tools.
 */
import type { ContentBlock, GenerateOptions, RequestMessage, ToolSchema } from '@deepseek-ai/dsh-llm'

export const TOOL_CALL_OPEN = '<dsh_tool_call>'
export const TOOL_CALL_CLOSE = '</dsh_tool_call>'

const MAX_TOOL_SCHEMA_CHARS = 1200

function formatBlocks(content: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of content) {
    switch (block.type) {
      case 'text':
        if (block.text) parts.push(block.text)
        break
      case 'reasoning':
        // Omit prior reasoning chains to keep prompt concise and focused.
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
  const tag = message.role.toUpperCase()
  return `[${tag}]\n${body}`
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
  const hasCursorAcp = tools.some(t => t.name === 'subagent_cursor')
  const toolLines = tools.map(summarizeToolSchema).join('\n')
  const delegationHint = hasCursorAcp
    ? '\n- For multi-step coding, editing files, or running workspace commands, prefer calling `subagent_cursor` (which runs a full native Cursor Agent via ACP) or call the specific workspace tool directly.'
    : ''
  return [
    '[HARNESS TOOL BRIDGE]',
    'You are acting as the reasoning model for DeepSeek Harness (DSH). Because you are running in read-only Ask mode, you do NOT have direct built-in write/shell tools; instead, DSH provides the following host tools:',
    toolLines,
    '',
    'Tool-calling rules:',
    `- If you can answer the user directly without calling a tool, reply normally in plain text and NEVER emit ${TOOL_CALL_OPEN}.`,
    `- If you need to invoke one or more DSH tools, output each call wrapped exactly as: ${TOOL_CALL_OPEN}{"name":"<tool_name>","arguments":{...}}${TOOL_CALL_CLOSE}`,
    '- Use valid JSON inside the tags with only declared tool names and arguments, then stop immediately so DSH can execute the tool(s) and return [TOOL RESULT] in the next turn.',
    delegationHint,
  ].filter(Boolean).join('\n')
}

export interface BuildPromptOptions {
  readonly enableToolShim: boolean
}

/** Serialize a full DSH `GenerateOptions` request into a single prompt for stdin piping. */
export function buildCursorAskPrompt(
  options: GenerateOptions,
  { enableToolShim }: BuildPromptOptions,
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
    // Pure single-turn user prompt (e.g. title generation or simple one-shot): pass verbatim.
    return formatBlocks(options.messages[0].content)
  }

  if (transcript.length > 0) {
    sections.push('[CONVERSATION]\n' + transcript.join('\n\n'))
  }

  sections.push('[INSTRUCTION]\nRespond as [ASSISTANT] to the latest message above. Do not prefix your response with "[ASSISTANT]".')
  return sections.join('\n\n')
}
