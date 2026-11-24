import { describe, expect, it } from 'vitest'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { GenerateOptions, MessageId } from '@deepseek-ai/dsh-llm'
import {
  buildCursorAskPrompt,
  inferAcpContextWindow,
  parseCursorParameterizedModels,
  parseCursorVersionKey,
  parseModelListOutput,
  resolveAcpModelId,
  resolveAcpModelSelection,
  stripModelParams,
  TOOL_CALL_CLOSE,
  TOOL_CALL_OPEN,
  ToolShimParser,
} from '../src/index.ts'

describe('@deepseek-ai/dsh-llm-cursor-cli', () => {
  it('sorts Cursor version directory names chronologically', () => {
    expect(parseCursorVersionKey('2026.11.13-eacade1')).toBe('20261113000000')
    expect(parseCursorVersionKey('2026.11.13-14-30-05-abcdef0')).toBe('20261113143005')
    expect(parseCursorVersionKey('not-a-version')).toBeNull()
  })

  it('parses agent --list-models output cleanly and strips zero-width characters', () => {
    const raw = [
      'Available models',
      '',
      'auto - Auto (default)',
      'grok-4.7-low-fast - Grok 4.7  Low Fast\u200b\u200b',
      'deepseek-v41-flash-1017-high - DeepSeek V4.1 Flash 1017 High',
      '',
      'Tip: use --model <id> to switch.',
    ].join('\n')
    expect(parseModelListOutput(raw)).toEqual([
      { id: 'auto', name: 'Auto (default)' },
      { id: 'grok-4.7-low-fast', name: 'Grok 4.7 Low Fast' },
      { id: 'deepseek-v41-flash-1017-high', name: 'DeepSeek V4.1 Flash 1017 High' },
    ])
  })

  it('keeps pure one-shot prompts clean without tool shim noise', () => {
    const req: GenerateOptions = {
      provider: 'cursor-cli',
      model: 'auto',
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'Summarize this title' }] },
      ],
    }
    expect(buildCursorAskPrompt(req, { enableToolShim: true })).toBe('Summarize this title')
  })

  it('includes tool bridge protocol and ACP delegation hint when tools are provided', () => {
    const req: GenerateOptions = {
      provider: 'cursor-cli',
      model: 'auto',
      system: 'Be concise.',
      tools: [
        {
          name: 'subagent_cursor',
          description: 'Delegate coding tasks to Cursor ACP agent',
          parameters: { type: 'object', properties: { prompt: { type: 'string' } } },
        },
      ],
      messages: [
        {
          id: brandString<MessageId>('m1'),
          role: 'user',
          source: { kind: 'user' },
          content: [{ type: 'text', text: 'Refactor foo.ts' }],
        },
      ],
    }
    const prompt = buildCursorAskPrompt(req, { enableToolShim: true })
    expect(prompt).toContain('[HARNESS TOOL BRIDGE]')
    expect(prompt).toContain('subagent_cursor')
    expect(prompt).toContain(TOOL_CALL_OPEN)
    expect(prompt).toContain('Refactor foo.ts')
  })

  it('streams text immediately and parses chunk-split tool-call tags cleanly', () => {
    const texts: string[] = []
    const calls: Array<{ name: string; argumentsJson: string }> = []
    const parser = new ToolShimParser(new Set(['subagent_cursor']), {
      onText: t => texts.push(t),
      onToolCall: c => calls.push(c),
    })

    parser.push('I will delegate this. <dsh_')
    parser.push('tool_call>{"name":"subagent_cursor","arguments":{"prompt":"fix bug"}}</dsh_tool_call>')
    parser.push(' Done.')
    parser.finish()

    expect(texts.join('')).toBe('I will delegate this.  Done.')
    expect(calls).toEqual([
      { name: 'subagent_cursor', argumentsJson: '{"prompt":"fix bug"}' },
    ])
  })

  it('degrades malformed or unknown tool tags safely to plain text', () => {
    const texts: string[] = []
    const calls: Array<{ name: string; argumentsJson: string }> = []
    const parser = new ToolShimParser(new Set(['subagent_cursor']), {
      onText: t => texts.push(t),
      onToolCall: c => calls.push(c),
    })

    parser.push(`${TOOL_CALL_OPEN}{"name":"unknown_tool","arguments":{}}${TOOL_CALL_CLOSE}`)
    parser.finish()

    expect(calls).toEqual([])
    expect(texts.join('')).toContain('unknown_tool')
  })

  it('parses parameterized Cursor models with efforts and resolves legacy IDs accurately', () => {
    const parsed = parseCursorParameterizedModels({
      models: [
        {
          value: 'auto-smart',
          name: 'Auto',
          configOptions: [
            {
              id: 'optimize_for',
              category: 'model_config',
              currentValue: 'balanced',
              options: [
                { value: 'intelligence', name: 'Intelligence' },
                { value: 'balanced', name: 'Balance' },
                { value: 'cost', name: 'Cost' },
              ],
            },
          ],
        },
        {
          value: 'gemini-4-neon',
          name: 'Gemini 4 Neon',
          configOptions: [
            {
              id: 'context',
              category: 'model_config',
              currentValue: '300k',
              options: [
                { value: '300k', name: '300K' },
                { value: '1m', name: '1M' },
              ],
            },
            {
              id: 'reasoning_effort',
              category: 'thought_level',
              currentValue: 'high',
              options: [
                { value: 'low', name: 'Low' },
                { value: 'medium', name: 'Medium' },
                { value: 'high', name: 'High' },
                { value: 'xhigh', name: 'Extra High' },
              ],
            },
          ],
        },
      ],
    })
    expect(parsed[0]?.effortConfigId).toBe('optimize_for')
    expect(parsed[0]?.defaultEffort).toBe('balanced')
    expect(parsed[1]?.modelId).toBe('gemini-4-neon')
    expect(parsed[1]?.name).toBe('Gemini 4 Neon (300K)')
    expect(parsed[1]?.contextWindow).toBe(300_000)
    expect(parsed[1]?.contextOptionValue).toBe('300k')
    expect(parsed[1]?.effortConfigId).toBe('reasoning_effort')
    expect(parsed[1]?.efforts.map(e => e.id)).toEqual(['low', 'medium', 'high', 'xhigh'])

    expect(parsed[2]?.modelId).toBe('gemini-4-neon@1m')
    expect(parsed[2]?.acpModelId).toBe('gemini-4-neon')
    expect(parsed[2]?.name).toBe('Gemini 4 Neon (1M)')
    expect(parsed[2]?.contextWindow).toBe(1_000_000)
    expect(parsed[2]?.contextOptionValue).toBe('1m')
    expect(parsed[2]?.efforts.map(e => e.id)).toEqual(['low', 'medium', 'high', 'xhigh'])

    expect(stripModelParams('gemini-4-neon@1m')).toBe('gemini-4-neon')
    expect(stripModelParams('gemini-4-neon[context=300k,reasoning_effort=high]')).toBe('gemini-4-neon')
    expect(resolveAcpModelSelection('gemini-4-neon@1m', parsed)).toEqual({
      acpModelId: 'gemini-4-neon',
      contextValue: '1m',
    })
    expect(resolveAcpModelId('auto', parsed, 'auto-smart')).toBeNull()
    expect(resolveAcpModelId('auto-smart[optimize_for=balanced]', parsed, 'gemini-4-neon')).toBe('auto-smart')
    expect(resolveAcpModelId('gemini-4-neon[context=300k,reasoning_effort=high]', parsed)).toBe('gemini-4-neon')
    expect(inferAcpContextWindow('gpt-5.6-sol[context=272k,reasoning=medium,fast=false]', 'gpt-5.6-sol')).toBe(272_000)
  })
})
