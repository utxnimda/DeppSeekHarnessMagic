import { describe, expect, it } from 'vitest'
import {
  extractWorkBuddyPromptError,
  parseWorkBuddyModels,
  resolveWorkBuddyModelId,
  ToolShimParser,
  WB_TOOL_CALL_CLOSE,
  WB_TOOL_CALL_OPEN,
} from '../src/index.ts'

describe('@deepseek-ai/dsh-llm-workbuddy-acp', () => {
  it('parses availableModels, exact maxInputTokens, and effortOptions from session/new', () => {
    const parsed = parseWorkBuddyModels({
      models: {
        availableModels: [
          {
            modelId: 'hy3-preview-ioa',
            name: 'Hy3 preview',
            _meta: {
              maxInputTokens: 192000,
              effortOptions: [
                { value: 'low', name: 'Low' },
                { value: 'medium', name: 'Medium' },
                { value: 'high', name: 'High' },
                { value: 'xhigh', name: 'X-High' },
              ],
            },
          },
          { modelId: 'claude-opus-4.7-1m', name: 'Claude Opus 4.7 1M', _meta: { maxInputTokens: 1000000 } },
        ],
      },
    })
    expect(parsed).toEqual([
      {
        modelId: 'hy3-preview-ioa',
        name: 'Hy3 preview',
        maxInputTokens: 192000,
        efforts: [
          { id: 'low', name: 'Low' },
          { id: 'medium', name: 'Medium' },
          { id: 'high', name: 'High' },
          { id: 'xhigh', name: 'X-High' },
        ],
        defaultEffort: 'high',
      },
      {
        modelId: 'claude-opus-4.7-1m',
        name: 'Claude Opus 4.7 1M',
        maxInputTokens: 1000000,
        efforts: [],
        defaultEffort: undefined,
      },
    ])
    expect(resolveWorkBuddyModelId('Hy3 preview', parsed)).toBe('hy3-preview-ioa')
    expect(resolveWorkBuddyModelId('default', parsed)).toBeNull()
  })

  it('extracts structured auth-required errors from WorkBuddy refusal _meta', () => {
    const info = extractWorkBuddyPromptError({
      stopReason: 'refusal',
      _meta: {
        'codebuddy.ai/errorMessage': JSON.stringify({
          code: -32000,
          message: 'Authentication required',
          data: { details: 'Please use /login command', category: 'auth' },
        }),
      },
    })
    expect(info).toEqual({ message: 'Please use /login command', isAuth: true })
    expect(extractWorkBuddyPromptError({ stopReason: 'end_turn' })).toBeNull()
  })

  it('streams text and extracts valid tool calls safely', () => {
    const texts: string[] = []
    const calls: Array<{ name: string; argumentsJson: string }> = []
    const parser = new ToolShimParser(new Set(['subagent_workbuddy']), {
      onText: t => texts.push(t),
      onToolCall: c => calls.push(c),
    })
    parser.push(`Calling WorkBuddy ${WB_TOOL_CALL_OPEN}{"name":"subagent_workbuddy","arguments":{"prompt":"check repo"}}${WB_TOOL_CALL_CLOSE}`)
    parser.finish()
    expect(texts.join('')).toBe('Calling WorkBuddy ')
    expect(calls).toEqual([{ name: 'subagent_workbuddy', argumentsJson: '{"prompt":"check repo"}' }])
  })
})
