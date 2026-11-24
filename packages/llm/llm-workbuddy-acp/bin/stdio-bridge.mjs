#!/usr/bin/env node
/**
 * Clean stdio NDJSON bridge for WorkBuddy's bundled CLI (`--serve --acp --acp-transport stdio`).
 *
 * WorkBuddy's `--serve --acp` prints a short human-readable HTTP server banner to stdout before
 * ACP JSON-RPC traffic begins. Standard ACP SDK `ndJsonStream` readers expect strictly JSON lines
 * on stdout, so this tiny bridge redirects non-JSON banner lines to stderr and forwards pure
 * JSON-RPC NDJSON lines verbatim on stdout while piping stdin through untouched.
 */
import { spawn } from 'node:child_process'

const [nodeExe, cliScript, ...extraArgs] = process.argv.slice(2)
if (!nodeExe || !cliScript) {
  process.stderr.write('workbuddy-acp stdio-bridge: missing <nodeExe> <cliScript> arguments\n')
  process.exit(2)
}

const child = spawn(
  nodeExe,
  [cliScript, '--serve', '--acp', '--acp-transport', 'stdio', '--port', '0', ...extraArgs],
  {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      CODEBUDDY_FORCE_LITE_WB_BUNDLE: '1',
    },
    windowsHide: true,
  },
)

process.stdin.pipe(child.stdin)
child.stdin.on('error', () => {
  // Ignore EPIPE on shutdown
})

let remainder = ''
child.stdout.setEncoding('utf8')
child.stdout.on('data', (chunk) => {
  remainder += chunk
  let nl = remainder.indexOf('\n')
  while (nl >= 0) {
    const line = remainder.slice(0, nl)
    remainder = remainder.slice(nl + 1)
    const trimmed = line.trim()
    if (trimmed.startsWith('{')) {
      process.stdout.write(`${trimmed}\n`)
    } else if (trimmed) {
      const redacted = trimmed
        .replace(/([?&]password=)([^\s&]+)/gi, '$1***')
        .replace(/(Password\s+)(\S+)/gi, '$1***')
      process.stderr.write(`[workbuddy-acp] ${redacted}\n`)
    }
    nl = remainder.indexOf('\n')
  }
})

child.stderr.pipe(process.stderr)

const forwardSignal = () => {
  try {
    child.kill()
  } catch {
    // Ignore
  }
}

process.on('SIGINT', forwardSignal)
process.on('SIGTERM', forwardSignal)
process.stdin.on('end', () => {
  try {
    child.stdin.end()
  } catch {
    // Ignore
  }
})

child.on('error', (error) => {
  process.stderr.write(`workbuddy-acp stdio-bridge spawn error: ${error.message}\n`)
  process.exit(1)
})

child.on('close', (code) => {
  if (remainder.trim().startsWith('{')) {
    process.stdout.write(`${remainder.trim()}\n`)
  }
  process.exit(code ?? 0)
})
