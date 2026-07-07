import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs'
import { basename, delimiter, dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const appRuntimeRoot = join(root, 'app_runtime', 'python')
const appRuntimePython = process.platform === 'win32'
  ? join(appRuntimeRoot, 'python.exe')
  : join(appRuntimeRoot, 'bin', 'python')
const task = process.argv[2]

function runtimeCandidates() {
  return existsSync(appRuntimePython) ? [appRuntimePython] : []
}

function pythonEnv(python) {
  const normalizedPython = resolve(python)
  const normalizedRuntime = resolve(appRuntimeRoot)
  if (!normalizedPython.startsWith(normalizedRuntime)) {
    return {}
  }

  if (process.platform === 'win32') {
    return {
      PYTHONIOENCODING: 'utf-8'
    }
  }

  const runtimeRoot = dirname(dirname(normalizedPython))
  const libRoot = join(runtimeRoot, 'lib')
  const pythonLibName = existsSync(libRoot)
    ? readdirSync(libRoot).find((name) => /^python3\.\d+$/.test(name))
    : undefined
  const pythonLib = pythonLibName ? join(libRoot, pythonLibName) : join(libRoot, 'python3.10')
  const sitePackages = join(pythonLib, 'site-packages')
  const pythonPathParts = []
  const hasStdlib = existsSync(join(pythonLib, 'encodings', '__init__.py'))

  if (hasStdlib) {
    pythonPathParts.push(pythonLib)
  }
  if (existsSync(sitePackages)) {
    pythonPathParts.push(sitePackages)
  }

  return {
    ...(hasStdlib ? { PYTHONHOME: runtimeRoot } : {}),
    ...(pythonPathParts.length ? { PYTHONPATH: pythonPathParts.join(delimiter) } : {}),
    DYLD_FALLBACK_LIBRARY_PATH: libRoot,
    LD_LIBRARY_PATH: libRoot,
    PYTHONIOENCODING: 'utf-8'
  }
}

function inspectPython(python) {
  const code = [
    'import json, pathlib, sys',
    'executable = pathlib.Path(sys.executable).resolve()',
    'root = executable.parents[1]',
    'print(json.dumps({',
    '  "executable": str(executable),',
    '  "root": str(root),',
    '  "version": list(sys.version_info[:3])',
    '}))'
  ].join('\n')
  const result = spawnSync(python, ['-c', code], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: {
      ...process.env,
      ...pythonEnv(python)
    }
  })

  if (result.status !== 0) {
    return null
  }

  try {
    return JSON.parse(result.stdout)
  } catch {
    return null
  }
}

function versionIsSupported(info) {
  const [major, minor] = info.version ?? []
  return major > 3 || (major === 3 && minor >= 10)
}

function canRun(python, code) {
  const result = spawnSync(python, ['-c', code], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: {
      ...process.env,
      ...pythonEnv(python)
    }
  })
  return result.status === 0
}

function findRuntimePython({ requireCoverage = false } = {}) {
  for (const python of runtimeCandidates()) {
    const info = inspectPython(python)
    if (!info || !versionIsSupported(info)) {
      continue
    }
    if (requireCoverage && !canRun(info.executable, 'import coverage')) {
      continue
    }
    return info
  }

  return null
}

function runPython(python, args, env = {}) {
  const result = spawnSync(python, args, {
    cwd: root,
    stdio: 'inherit',
    env: {
      ...process.env,
      ...pythonEnv(python),
      ...env
    }
  })
  return result.status ?? 1
}

function runNode(script, args = []) {
  const result = spawnSync(process.execPath, [script, ...args], {
    cwd: root,
    stdio: 'inherit'
  })
  return result.status ?? 1
}

function copyRuntime(sourceRoot) {
  rmSync(appRuntimeRoot, { recursive: true, force: true })
  mkdirSync(dirname(appRuntimeRoot), { recursive: true })
  cpSync(sourceRoot, appRuntimeRoot, {
    recursive: true,
    preserveTimestamps: true,
    filter: (source) => {
      const name = basename(source)
      return name !== '__pycache__' && !source.endsWith('.pyc')
    }
  })
}

function runCommand(command, args, { allowFailure = false } = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe']
  })

  if (result.status !== 0 && !allowFailure) {
    throw new Error(result.stderr || `${command} ${args.join(' ')} failed.`)
  }

  return result
}

function runtimeMachOCandidates(directoryPath, pythonBinPath, candidates = []) {
  for (const entry of readdirSync(directoryPath, { withFileTypes: true })) {
    const entryPath = join(directoryPath, entry.name)
    if (entry.isSymbolicLink()) {
      continue
    }
    if (entry.isDirectory()) {
      runtimeMachOCandidates(entryPath, pythonBinPath, candidates)
      continue
    }
    if (
      entry.name === 'Python' ||
      entry.name.endsWith('.dylib') ||
      entry.name.endsWith('.so') ||
      (directoryPath === pythonBinPath && entry.name.startsWith('python'))
    ) {
      candidates.push(entryPath)
    }
  }

  return candidates
}

function loaderPathReference(fromPath, toPath) {
  const relativePath = relative(dirname(fromPath), toPath).replaceAll('\\', '/')
  return `@loader_path/${relativePath}`
}

function rpathReference(rootPath, toPath) {
  const relativePath = relative(rootPath, toPath).replaceAll('\\', '/')
  return `@rpath/${relativePath}`
}

function patchMacPythonRuntime() {
  const pythonLibrary = join(appRuntimeRoot, 'Python')
  const pythonBinPath = join(appRuntimeRoot, 'bin')

  if (process.platform !== 'darwin' || !existsSync(pythonLibrary) || !existsSync(pythonBinPath)) {
    return
  }

  const installName = runCommand('/usr/bin/otool', ['-D', pythonLibrary])
  const oldPythonLibraryName = installName.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line && !line.endsWith(':'))

  if (!oldPythonLibraryName || oldPythonLibraryName.startsWith('@')) {
    return
  }
  const oldPythonRoot = dirname(oldPythonLibraryName)
  const machOCandidates = runtimeMachOCandidates(appRuntimeRoot, pythonBinPath)

  for (const binaryPath of machOCandidates) {
    const installName = runCommand('/usr/bin/otool', ['-D', binaryPath], { allowFailure: true })
    if (installName.status === 0) {
      const oldInstallName = installName.stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line && !line.endsWith(':'))
      if (oldInstallName?.startsWith(`${oldPythonRoot}/`)) {
        runCommand('/usr/bin/install_name_tool', [
          '-id',
          rpathReference(appRuntimeRoot, binaryPath),
          binaryPath
        ])
      }
    }

    const linkedLibraries = runCommand('/usr/bin/otool', ['-L', binaryPath], { allowFailure: true })
    if (linkedLibraries.status !== 0) {
      continue
    }

    for (const line of linkedLibraries.stdout.split(/\r?\n/)) {
      const linkedLibrary = line.trim().split(/\s+/)[0]
      if (!linkedLibrary?.startsWith(`${oldPythonRoot}/`)) {
        continue
      }

      const bundledLibrary = join(appRuntimeRoot, linkedLibrary.slice(oldPythonRoot.length + 1))
      if (!existsSync(bundledLibrary)) {
        continue
      }

      runCommand('/usr/bin/install_name_tool', [
        '-change',
        linkedLibrary,
        loaderPathReference(binaryPath, bundledLibrary),
        binaryPath
      ])
    }
  }

  for (const binaryPath of machOCandidates) {
    runCommand('/usr/bin/codesign', ['--force', '--sign', '-', binaryPath], { allowFailure: true })
  }

  const pythonAppPath = join(appRuntimeRoot, 'Resources', 'Python.app')
  if (existsSync(pythonAppPath)) {
    runCommand('/usr/bin/codesign', ['--force', '--deep', '--sign', '-', pythonAppPath])
  }

  console.log('[python-runtime] Rewrote and signed bundled Python framework links.')
}

function setup() {
  const buildPythonInfo = inspectPython('python')

  if (!buildPythonInfo || !versionIsSupported(buildPythonInfo)) {
    console.error('No usable Python 3.10+ runtime was found for building app_runtime.')
    console.error('GitHub Actions uses `python`; install or activate the same Python before running this setup.')
    process.exit(1)
  }

  let status = runPython(buildPythonInfo.executable, ['-m', 'pip', 'install', '--upgrade', 'pip'])
  if (status !== 0) {
    process.exit(status)
  }

  status = runPython(buildPythonInfo.executable, ['-m', 'pip', 'install', '-r', 'requirements-runtime.txt'])
  if (status !== 0) {
    process.exit(status)
  }

  console.log(`[python-runtime] Copying ${buildPythonInfo.root} -> ${appRuntimeRoot}`)
  copyRuntime(buildPythonInfo.root)
  patchMacPythonRuntime()

  if (!existsSync(appRuntimePython)) {
    console.error(`Copied runtime did not contain ${appRuntimePython}.`)
    process.exit(1)
  }

  status = runPython(appRuntimePython, [
    '-c',
    'import faiss, numpy, pypdfium2, PIL; print("TokenSmith app_runtime ready:", __import__("sys").executable)'
  ])
  if (status !== 0) {
    process.exit(status)
  }
}

function requireRuntimePython({ requireCoverage = false } = {}) {
  const python = findRuntimePython({ requireCoverage })
  if (!python) {
    console.error('No TokenSmith Python runtime was found.')
    console.error('Run npm run setup:python-runtime to build app_runtime/python.')
    process.exit(1)
  }
  return python
}

function runIntegration({ requireGguf = false } = {}) {
  requireRuntimePython()
  const args = requireGguf ? ['--require-gguf'] : []

  const tests = requireGguf
    ? [
        'tests/integration/python-engine-pdf.test.mjs',
        'tests/integration/gerard-larcher-chat.test.mjs',
        'tests/integration/annita-demetriou-chat.test.mjs'
      ]
    : ['tests/integration/python-engine-pdf.test.mjs']

  for (const test of tests) {
    const status = runNode(test, args)
    if (status !== 0) {
      process.exit(status)
    }
  }
}

if (task === 'setup' || task === 'setup-runtime') {
  setup()
} else if (task === 'unit') {
  const python = requireRuntimePython()
  process.exit(runPython(python.executable, ['-m', 'unittest', 'discover', '-s', 'tests/python', '-p', 'test_*.py']))
} else if (task === 'coverage') {
  const python = requireRuntimePython({ requireCoverage: true })
  const coverageStatus = runPython(
    python.executable,
    ['-m', 'coverage', 'run', '-m', 'unittest', 'discover', '-s', 'tests/python', '-p', 'test_*.py']
  )
  if (coverageStatus !== 0) {
    process.exit(coverageStatus)
  }
  process.exit(runPython(python.executable, ['-m', 'coverage', 'report']))
} else if (task === 'integration') {
  runIntegration()
} else if (task === 'integration:gguf') {
  runIntegration({ requireGguf: true })
} else {
  console.error('Usage: node scripts/python-dev.mjs <setup|setup-runtime|unit|coverage|integration|integration:gguf>')
  process.exit(1)
}
