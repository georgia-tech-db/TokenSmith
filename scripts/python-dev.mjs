import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const venvPython = process.platform === 'win32'
  ? join(root, '.venv', 'Scripts', 'python.exe')
  : join(root, '.venv', 'bin', 'python')

const task = process.argv[2]

function unique(values) {
  return [...new Set(values.filter(Boolean))]
}

function candidates() {
  return unique([
    process.env.TOKENSMITH_TEST_PYTHON,
    process.env.TOKENSMITH_PYTHON,
    existsSync(venvPython) ? venvPython : null,
    process.env.PYTHON,
    'python',
    'python3'
  ])
}

function canRun(python, code) {
  const result = spawnSync(python, ['-c', code], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe']
  })
  return result.status === 0
}

function findPython({ requireCoverage = false } = {}) {
  for (const python of candidates()) {
    if (!canRun(python, 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)')) {
      continue
    }
    if (requireCoverage && !canRun(python, 'import coverage')) {
      continue
    }
    return python
  }

  return null
}

function runPython(python, args, env = {}) {
  const result = spawnSync(python, args, {
    cwd: root,
    stdio: 'inherit',
    env: {
      ...process.env,
      ...env
    }
  })
  return result.status ?? 1
}

function runNode(script, env = {}) {
  const result = spawnSync(process.execPath, [script], {
    cwd: root,
    stdio: 'inherit',
    env: {
      ...process.env,
      ...env
    }
  })
  return result.status ?? 1
}

function setup() {
  const basePython = findPython()
  if (!basePython) {
    console.error('No usable Python 3.10+ runtime was found.')
    process.exit(1)
  }

  const venvResult = spawnSync(basePython, ['-m', 'venv', '.venv'], {
    cwd: root,
    stdio: 'inherit'
  })

  if (venvResult.status !== 0) {
    process.exit(venvResult.status ?? 1)
  }

  let status = runPython(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip'])
  if (status !== 0) {
    process.exit(status)
  }

  status = runPython(venvPython, ['-m', 'pip', 'install', '-r', 'requirements-runtime.txt', '-r', 'requirements-dev.txt'])
  if (status !== 0) {
    process.exit(status)
  }
}

function runIntegration({ requireGguf = false } = {}) {
  const python = findPython()
  if (!python) {
    console.error('No usable Python 3.10+ runtime was found.')
    process.exit(1)
  }

  const env = {
    TOKENSMITH_TEST_PYTHON: python
  }

  if (requireGguf) {
    env.TOKENSMITH_REQUIRE_GGUF_INTEGRATION = '1'
  }

  const tests = requireGguf
    ? [
        'tests/integration/python-engine-pdf.test.mjs',
        'tests/integration/gerard-larcher-chat.test.mjs',
        'tests/integration/annita-demetriou-chat.test.mjs'
      ]
    : ['tests/integration/python-engine-pdf.test.mjs']

  for (const test of tests) {
    const status = runNode(test, env)
    if (status !== 0) {
      process.exit(status)
    }
  }
}

if (task === 'setup') {
  setup()
} else if (task === 'unit') {
  const python = findPython()
  if (!python) {
    console.error('No usable Python 3.10+ runtime was found.')
    process.exit(1)
  }
  process.exit(runPython(python, ['-m', 'unittest', 'discover', '-s', 'tests/python', '-p', 'test_*.py']))
} else if (task === 'coverage') {
  const python = findPython({ requireCoverage: true })
  if (!python) {
    console.error('No Python runtime with coverage.py was found.')
    console.error('Run npm run setup:python-dev, or set TOKENSMITH_TEST_PYTHON=/path/to/python.')
    process.exit(1)
  }
  const coverageStatus = runPython(
    python,
    ['-m', 'coverage', 'run', '-m', 'unittest', 'discover', '-s', 'tests/python', '-p', 'test_*.py']
  )
  if (coverageStatus !== 0) {
    process.exit(coverageStatus)
  }
  process.exit(runPython(python, ['-m', 'coverage', 'report']))
} else if (task === 'integration') {
  runIntegration()
} else if (task === 'integration:gguf') {
  runIntegration({ requireGguf: true })
} else {
  console.error('Usage: node scripts/python-dev.mjs <setup|unit|coverage|integration|integration:gguf>')
  process.exit(1)
}
