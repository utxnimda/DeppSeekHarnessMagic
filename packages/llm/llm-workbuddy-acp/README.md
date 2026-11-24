# @deepseek-ai/dsh-llm-workbuddy-acp

Local DeepSeek Harness plugin integrating **WorkBuddy's bundled CLI (`@genie/agent-cli`)** over persistent ACP:

- **Main LLM provider (`workbuddy-acp`, display name `WorkBuddy (ACP)`)**:
  - Auto-discovers `%LOCALAPPDATA%\Programs\WorkBuddy\resources\app.asar.unpacked\cli\bin\codebuddy` with `CODEBUDDY_FORCE_LITE_WB_BUNDLE=1`.
  - Uses a dedicated CLI config directory (default `%USERPROFILE%\.codebuddy`) so WorkBuddy Desktop's encrypted `%USERPROFILE%\.workbuddy` store is never modified.
  - Launches `--serve --acp --acp-transport stdio --port 0 --tools "" --no-session-persistence`, filters initial non-JSON HTTP Server banner lines, discovers live models + `_meta.maxInputTokens` + `_meta.effortOptions`, and applies reasoning effort via `session/set_config_option(configId='thought_level')`.
  - Supports optional on-demand ACP authentication (`autoAuthMethod`: `iOA` | `internal` | `external` | `selfhosted`) or `CODEBUDDY_AUTH_TOKEN` / `CODEBUDDY_API_KEY`.
- **Out-of-process ACP subagent (`workbuddy-acp` / `subagent_workbuddy`)**:
  - Uses `bin/stdio-bridge.mjs` to redact and redirect non-JSON startup banner lines to stderr while forwarding pure JSON-RPC NDJSON to `@deepseek-ai/dsh-subagent-acp`.

## Checks & Rebuild

```powershell
corepack pnpm exec tsc -b packages/llm/llm-workbuddy-acp
corepack pnpm exec vitest run packages/llm/llm-workbuddy-acp/tests/workbuddy-acp.spec.ts
corepack pnpm run build:artifacts
```
