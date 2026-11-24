/**
 * Cordis plugin mounting:
 * 1. `workbuddy-acp` LLM provider adapter over a persistent local WorkBuddy ACP connection.
 * 2. Optional `workbuddy-acp` out-of-process subagent provider via `@deepseek-ai/dsh-subagent-acp`
 *    and our clean NDJSON stdio bridge (`bin/stdio-bridge.mjs`), enabling `subagent_workbuddy`.
 *
 * Uses named exports only so Cordis loader preserves `name` and `inject`.
 * @module @deepseek-ai/dsh-llm-workbuddy-acp
 */
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import * as subagentAcp from '@deepseek-ai/dsh-subagent-acp'
import { WorkBuddyAcpLlmAdapter } from './adapter.ts'
import { resolveWorkBuddyLauncher } from './launcher.ts'

export {
  extractWorkBuddyPromptError,
  parseWorkBuddyModels,
  PersistentWorkBuddyAcpClient,
  resolveWorkBuddyModelId,
  WorkBuddyAcpError,
  type WorkBuddyAcpModel,
  type WorkBuddyEffortOption,
  type WorkBuddyPromptRequest,
  type WorkBuddyPromptResult,
  type WorkBuddySessionUpdate,
} from './acp-client.ts'
export { WorkBuddyAcpLlmAdapter } from './adapter.ts'
export { resolveWorkBuddyLauncher, type WorkBuddyLauncher } from './launcher.ts'
export { buildWorkBuddyPrompt, ToolShimParser, WB_TOOL_CALL_CLOSE, WB_TOOL_CALL_OPEN } from './prompt.ts'

export const name = 'llm-workbuddy-acp'
export const inject = ['llm']

export interface Config {
  /** Provider route key registered on `ctx.llm` (default `workbuddy-acp`). */
  provider: string
  /** Display name in the model selector UI (default `WorkBuddy (ACP)`). */
  displayName: string
  /** Optional explicit WorkBuddy CLI script path (empty = auto-discover bundled CLI). */
  cliPath: string
  /** Optional explicit Node executable path (empty = auto-discover WorkBuddy Node or host Node). */
  nodePath: string
  /** Isolated CLI config dir holding auth state (default `~/.codebuddy`, never modifies `~/.workbuddy`). */
  configDir: string
  /** Optional `CODEBUDDY_INTERNET_ENVIRONMENT` value (e.g. `iOA`). */
  internetEnvironment: string
  /** Optional ACP authenticate method triggered on auth error (`""` | `"iOA"` | `"internal"` | `"external"` | `"selfhosted"`). */
  autoAuthMethod: '' | 'iOA' | 'internal' | 'external' | 'selfhosted'
  /** Enable delimited `<dsh_tool_call>` shim when DSH passes tools (default `true`). */
  enableToolShim: boolean
  /** Mount the `workbuddy-acp` out-of-process ACP subagent provider when `subagents` is present. */
  enableAcpSubagent: boolean
  /** Registered subagent provider name (default `workbuddy-acp`). */
  acpProviderName: string
  /** Permission policy for unattended `subagent_workbuddy` runs (`allow` | `reject`). */
  acpPermission: 'allow' | 'reject'
}

export const Config: z<Config> = z.object({
  provider: z.string().default('workbuddy-acp'),
  displayName: z.string().default('WorkBuddy (ACP)'),
  cliPath: z.string().default(''),
  nodePath: z.string().default(''),
  configDir: z.string().default(''),
  internetEnvironment: z.string().default(''),
  autoAuthMethod: z.union(['', 'iOA', 'internal', 'external', 'selfhosted'] as const).default(''),
  enableToolShim: z.boolean().default(true),
  enableAcpSubagent: z.boolean().default(true),
  acpProviderName: z.string().default('workbuddy-acp'),
  acpPermission: z.union(['allow', 'reject'] as const).default('allow'),
})

function resolveBridgeScriptPath(): string {
  // Works from both `src/index.ts` (tsx) and `lib/index.js` (built bundle).
  return resolve(fileURLToPath(new URL('../bin/stdio-bridge.mjs', import.meta.url)))
}

export function apply(ctx: Context, config: Config): void {
  const getLauncher = () => resolveWorkBuddyLauncher({
    customCliPath: config.cliPath,
    customNodePath: config.nodePath,
    configDir: config.configDir,
    internetEnvironment: config.internetEnvironment,
  })

  const adapter = new WorkBuddyAcpLlmAdapter({
    providerId: config.provider,
    displayName: config.displayName,
    enableToolShim: config.enableToolShim,
    autoAuthMethod: config.autoAuthMethod,
    getLauncher,
  })

  ctx.effect(() => {
    const unregister = ctx.llm.registerAdapter([config.provider], adapter)
    return () => {
      unregister()
      adapter.dispose()
    }
  }, 'llm-workbuddy-acp: register persistent ACP LLM adapter')

  if (config.enableAcpSubagent) {
    ctx.inject(['subagents', 'subprocess'], (child) => {
      const launcher = getLauncher()
      if (!launcher.available) return
      const bridgeScript = resolveBridgeScriptPath()
      const permissionArg = config.acpPermission === 'allow' ? 'acceptEdits' : 'dontAsk'
      child.plugin(subagentAcp, {
        providerName: config.acpProviderName,
        command: launcher.nodeExecutable,
        args: [
          bridgeScript,
          launcher.nodeExecutable,
          launcher.cliScript,
          '--permission-mode', permissionArg,
          '--no-session-persistence',
        ],
        permission: config.acpPermission,
        env: { ...launcher.env },
      })
    })
  }
}
