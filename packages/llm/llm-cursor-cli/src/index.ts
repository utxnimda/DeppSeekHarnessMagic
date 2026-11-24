/**
 * Cordis plugin mounting:
 * 1. `cursor-cli` LLM provider adapter over a persistent local Cursor CLI ACP server (`agent acp`).
 * 2. Optional `cursor-acp` out-of-process subagent provider (reuses `@deepseek-ai/dsh-subagent-acp`
 *    pointing directly at the native Cursor CLI `acp` server for full coding delegation).
 *
 * Uses named exports only so Cordis loader preserves `name` and `inject`.
 * @module @deepseek-ai/dsh-llm-cursor-cli
 */
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import * as subagentAcp from '@deepseek-ai/dsh-subagent-acp'
import { CursorCliLlmAdapter } from './adapter.ts'
import { resolveCursorLauncher } from './launcher.ts'

export {
  AcpRpcError,
  extractExplicitContextValue,
  inferAcpContextWindow,
  parseCursorParameterizedModels,
  PersistentCursorAcpClient,
  resolveAcpModelId,
  resolveAcpModelSelection,
  stripModelParams,
  type AcpEffortOption,
  type AcpModelOption,
  type AcpPromptRequest,
  type AcpPromptResult,
  type AcpSessionUpdate,
  type ResolvedAcpModelTarget,
} from './acp-client.ts'
export { CursorCliLlmAdapter } from './adapter.ts'
export {
  CursorModelCatalog,
  inferContextWindow,
  parseCursorVersionKey,
  parseModelListOutput,
  resolveCursorLauncher,
  type CursorLauncher,
  type ParsedCursorModel,
} from './launcher.ts'
export { buildCursorAskPrompt, TOOL_CALL_CLOSE, TOOL_CALL_OPEN } from './prompt.ts'
export { ToolShimParser, type ParsedToolCall } from './tool-parser.ts'

export const name = 'llm-cursor-cli'
export const inject = ['llm']

export interface Config {
  /** Provider route key registered on `ctx.llm` (default `cursor-cli`). */
  provider: string
  /** Display name in the model selector UI (default `Cursor CLI (ACP)`). */
  displayName: string
  /** Optional explicit executable path (empty = auto-discover native Cursor CLI). */
  command: string
  /** ACP session mode for the main LLM adapter (`ask` keeps edits/shell under DSH tools). */
  acpMode: 'ask' | 'plan' | 'agent'
  /** Enable delimited `<dsh_tool_call>` shim when DSH passes tools in `ask` mode (default `true`). */
  enableToolShim: boolean
  /** Mount the `cursor-acp` out-of-process ACP subagent provider when `subagents` is present. */
  enableAcpSubagent: boolean
  /** Registered subagent provider name for `agent acp` (default `cursor-acp`). */
  acpProviderName: string
  /** Permission policy for unattended `agent acp` subagent runs (`allow` | `reject`). */
  acpPermission: 'allow' | 'reject'
}

export const Config: z<Config> = z.object({
  provider: z.string().default('cursor-cli'),
  displayName: z.string().default('Cursor CLI (ACP)'),
  command: z.string().default(''),
  acpMode: z.union(['ask', 'plan', 'agent'] as const).default('ask'),
  enableToolShim: z.boolean().default(true),
  enableAcpSubagent: z.boolean().default(true),
  acpProviderName: z.string().default('cursor-acp'),
  acpPermission: z.union(['allow', 'reject'] as const).default('allow'),
})

export function apply(ctx: Context, config: Config): void {
  const getLauncher = () => resolveCursorLauncher(config.command)
  const adapter = new CursorCliLlmAdapter({
    providerId: config.provider,
    displayName: config.displayName,
    enableToolShim: config.enableToolShim,
    acpMode: config.acpMode,
    getLauncher,
  })

  ctx.effect(() => {
    const unregister = ctx.llm.registerAdapter([config.provider], adapter)
    return () => {
      unregister()
      adapter.dispose()
    }
  }, 'llm-cursor-cli: register persistent ACP LLM adapter')

  if (config.enableAcpSubagent) {
    ctx.inject(['subagents', 'subprocess'], (child) => {
      const launcher = getLauncher()
      child.plugin(subagentAcp, {
        providerName: config.acpProviderName,
        command: launcher.command,
        args: [...launcher.prefixArgs, 'acp'],
        permission: config.acpPermission,
        env: { ...launcher.env },
      })
    })
  }
}
