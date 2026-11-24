/**
 * Locates WorkBuddy's bundled CLI (`@genie/agent-cli`) and runtime executable.
 *
 * Uses the unpacked `codebuddy-lite-wb.mjs` fast bundle via `CODEBUDDY_FORCE_LITE_WB_BUNDLE=1`
 * and keeps CLI authentication state in a dedicated config dir (default `~/.codebuddy`)
 * so WorkBuddy Desktop's encrypted `~/.workbuddy` store is never disturbed.
 */
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

export interface WorkBuddyLauncher {
  readonly available: boolean
  readonly nodeExecutable: string
  readonly cliScript: string
  readonly label: string
  readonly env: Record<string, string>
  readonly error?: string
}

const FORWARDED_ENV_KEYS = [
  'CODEBUDDY_AUTH_TOKEN',
  'CODEBUDDY_API_KEY',
  'CODEBUDDY_INTERNET_ENVIRONMENT',
  'CODEBUDDY_BASE_URL',
  'CODEBUDDY_CONFIG_DIR',
] as const

/**
 * Resolve WorkBuddy's bundled CLI entrypoint and Node executable.
 */
export function resolveWorkBuddyLauncher(
  options: {
    readonly customCliPath?: string
    readonly customNodePath?: string
    readonly configDir?: string
    readonly internetEnvironment?: string
  } = {},
  env: NodeJS.ProcessEnv = process.env,
): WorkBuddyLauncher {
  const extraEnv: Record<string, string> = {
    CODEBUDDY_FORCE_LITE_WB_BUNDLE: '1',
  }

  for (const key of FORWARDED_ENV_KEYS) {
    const val = env[key]
    if (typeof val === 'string' && val.trim()) {
      extraEnv[key] = val.trim()
    }
  }

  const explicitConfigDir = options.configDir?.trim()
  if (explicitConfigDir) {
    extraEnv.CODEBUDDY_CONFIG_DIR = explicitConfigDir
  } else if (!extraEnv.CODEBUDDY_CONFIG_DIR) {
    extraEnv.CODEBUDDY_CONFIG_DIR = join(homedir(), '.codebuddy')
  }

  const explicitInternetEnv = options.internetEnvironment?.trim()
  if (explicitInternetEnv) {
    extraEnv.CODEBUDDY_INTERNET_ENVIRONMENT = explicitInternetEnv
  }

  // Node runtime: explicit > WorkBuddy bundled current node > current host node.
  let nodeExecutable = options.customNodePath?.trim() || ''
  if (!nodeExecutable) {
    const home = env.USERPROFILE || homedir()
    const wbNode = join(home, '.workbuddy', 'binaries', 'node', 'versions', 'current', 'node.exe')
    nodeExecutable = existsSync(wbNode) ? wbNode : process.execPath
  }

  // CLI script: explicit > WorkBuddy desktop app unpacked CLI.
  const candidates: string[] = []
  if (options.customCliPath?.trim()) {
    candidates.push(options.customCliPath.trim())
  }
  const localAppData = env.LOCALAPPDATA ?? ''
  if (localAppData) {
    candidates.push(
      join(localAppData, 'Programs', 'WorkBuddy', 'resources', 'app.asar.unpacked', 'cli', 'bin', 'codebuddy'),
    )
  }
  const programFiles = env.ProgramFiles ?? ''
  if (programFiles) {
    candidates.push(
      join(programFiles, 'WorkBuddy', 'resources', 'app.asar.unpacked', 'cli', 'bin', 'codebuddy'),
    )
  }

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return {
        available: true,
        nodeExecutable,
        cliScript: candidate,
        label: candidate,
        env: extraEnv,
      }
    }
  }

  return {
    available: false,
    nodeExecutable,
    cliScript: '',
    label: 'WorkBuddy CLI (not found)',
    env: extraEnv,
    error: '未在本机找到 WorkBuddy 内置 CLI（默认路径 %LOCALAPPDATA%\\Programs\\WorkBuddy\\resources\\app.asar.unpacked\\cli\\bin\\codebuddy）。',
  }
}
