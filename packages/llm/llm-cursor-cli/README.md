# @deepseek-ai/dsh-llm-cursor-cli

Local DeepSeek Harness plugin integrating the local **Cursor CLI (`agent acp`)** over persistent ACP:

- **Main LLM provider (`cursor-cli`, display name `Cursor CLI (ACP)`)**:
  - Auto-discovers the native bundled `node.exe` + `index.js` under `%LOCALAPPDATA%\cursor-agent\versions\<latest>\` (falling back to `agent` on PATH).
  - Negotiates `clientCapabilities._meta.parameterizedModelPicker = true` and queries `cursor/list_available_models` to expose multi-context variants (e.g. `Gemini 4 Neon (300K)` & `Gemini 4 Neon (1M)`, `GPT-5.6 Sol (272K)` & `GPT-5.6 Sol (1M)`, `Grok 4.7 (256K)` & `Grok 4.7 (500K)`) along with each model's native reasoning effort / Auto router tiers.
  - Runs in read-only `ask` mode with optional `<dsh_tool_call>` bridge so workspace mutations remain governed by DSH tools or delegated subagents.
  - Streams `agent_thought_chunk` (`reasoning-delta`) and `agent_message_chunk` (`text-delta`), and cancels cooperatively via `session/cancel`.
- **Out-of-process ACP subagent (`cursor-acp` / `subagent_cursor`)**:
  - Reuses `@deepseek-ai/dsh-subagent-acp` pointing directly at native `node.exe index.js acp`.

## Checks & Rebuild

```powershell
corepack pnpm exec tsc -b packages/llm/llm-cursor-cli
corepack pnpm exec vitest run packages/llm/llm-cursor-cli/tests/cursor-cli.spec.ts
corepack pnpm run build:artifacts
```
