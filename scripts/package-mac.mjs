import { execFile } from 'node:child_process'
import { copyFile, cp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { existsSync, readdirSync } from 'node:fs'
import { basename, dirname, join, relative, resolve } from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const rootDir = resolve(new URL('..', import.meta.url).pathname)
const packageJson = JSON.parse(await readFile(join(rootDir, 'package.json'), 'utf8'))
const appName = 'TokenSmith'
const appVersion = packageJson.version
const arch = process.arch
const releaseDir = join(rootDir, 'release')
const stagingDir = join(releaseDir, 'mac')
const appBundlePath = join(stagingDir, `${appName}.app`)
const dmgName = `${appName}-${appVersion}-mac-${arch}.dmg`
const dmgPath = join(releaseDir, dmgName)
const volumeName = `${appName} ${appVersion}`
const appIconIcnsFileName = 'tokensmith-icon.icns'
const appIconPngFileName = 'tokensmith-icon.png'
const appIconIcnsPath = join(rootDir, 'build-resources', appIconIcnsFileName)
const appIconPngPath = join(rootDir, 'build-resources', appIconPngFileName)

function log(message) {
  console.log(`[package:mac] ${message}`)
}

async function run(command, args, options = {}) {
  const result = await execFileAsync(command, args, {
    cwd: rootDir,
    maxBuffer: 32 * 1024 * 1024,
    ...options
  })

  return result
}

async function writeJson(filePath, value) {
  await writeFile(`${filePath}.tmp`, `${JSON.stringify(value, null, 2)}\n`, 'utf8')
  await copyFile(`${filePath}.tmp`, filePath)
  await rm(`${filePath}.tmp`)
}

async function patchPlist(plistPath, updates) {
  for (const [key, value] of Object.entries(updates)) {
    try {
      await run('/usr/libexec/PlistBuddy', ['-c', `Set :${key} ${value}`, plistPath])
    } catch {
      await run('/usr/libexec/PlistBuddy', ['-c', `Add :${key} string ${value}`, plistPath])
    }
  }
}

async function renameIfExists(from, to) {
  if (!existsSync(from)) {
    return
  }

  await rm(to, { force: true, recursive: true })
  await cp(from, to, { recursive: true, preserveTimestamps: true })
  await rm(from, { recursive: true, force: true })
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

async function patchMacPythonRuntime(appRuntimePath) {
  const pythonRoot = join(appRuntimePath, 'python')
  const pythonLibrary = join(pythonRoot, 'Python')
  const pythonBinPath = join(pythonRoot, 'bin')

  if (process.platform !== 'darwin' || !existsSync(pythonLibrary) || !existsSync(pythonBinPath)) {
    return
  }

  const installName = await run('/usr/bin/otool', ['-D', pythonLibrary])
  const oldPythonLibraryName = installName.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line && !line.endsWith(':'))

  if (!oldPythonLibraryName || oldPythonLibraryName.startsWith('@')) {
    return
  }
  const oldPythonRoot = dirname(oldPythonLibraryName)
  const machOCandidates = runtimeMachOCandidates(pythonRoot, pythonBinPath)

  for (const binaryPath of machOCandidates) {
    try {
      const installName = await run('/usr/bin/otool', ['-D', binaryPath])
      const oldInstallName = installName.stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line && !line.endsWith(':'))
      if (oldInstallName?.startsWith(`${oldPythonRoot}/`)) {
        await run('/usr/bin/install_name_tool', [
          '-id',
          rpathReference(pythonRoot, binaryPath),
          binaryPath
        ])
      }
    } catch {
      // Executables and extension modules may not have install IDs.
    }

    try {
      const linkedLibraries = await run('/usr/bin/otool', ['-L', binaryPath])
      for (const line of linkedLibraries.stdout.split(/\r?\n/)) {
        const linkedLibrary = line.trim().split(/\s+/)[0]
        if (!linkedLibrary?.startsWith(`${oldPythonRoot}/`)) {
          continue
        }

        const bundledLibrary = join(pythonRoot, linkedLibrary.slice(oldPythonRoot.length + 1))
        if (!existsSync(bundledLibrary)) {
          continue
        }

        await run('/usr/bin/install_name_tool', [
          '-change',
          linkedLibrary,
          loaderPathReference(binaryPath, bundledLibrary),
          binaryPath
        ])
      }
    } catch {
      // Some python* entries are shell scripts or symlinks; only Mach-O binaries need rewriting.
    }
  }

  for (const binaryPath of machOCandidates) {
    try {
      await run('/usr/bin/codesign', ['--force', '--sign', '-', binaryPath])
    } catch {
      // Only successfully parsed Mach-O binaries need refreshed signatures.
    }
  }

  const pythonAppPath = join(pythonRoot, 'Resources', 'Python.app')
  if (existsSync(pythonAppPath)) {
    await run('/usr/bin/codesign', ['--force', '--deep', '--sign', '-', pythonAppPath])
  }

  log('Rewrote and signed bundled Python framework links.')
}

async function prepareAppPayload(resourcesPath) {
  const appPayloadPath = join(resourcesPath, 'app')
  const appRuntimePath = join(rootDir, 'app_runtime')

  await mkdir(appPayloadPath, { recursive: true })
  await cp(join(rootDir, 'out'), join(appPayloadPath, 'out'), {
    recursive: true,
    preserveTimestamps: true
  })
  await cp(join(rootDir, 'python_engine'), join(appPayloadPath, 'python_engine'), {
    recursive: true,
    preserveTimestamps: true,
    filter: (source) => !source.includes('__pycache__') && !source.endsWith('.pyc')
  })
  if (existsSync(appRuntimePath)) {
    const packagedRuntimePath = join(appPayloadPath, 'app_runtime')
    await run('/usr/bin/ditto', [appRuntimePath, packagedRuntimePath])
    await patchMacPythonRuntime(packagedRuntimePath)
  }
  await writeJson(join(appPayloadPath, 'package.json'), {
    name: packageJson.name,
    version: appVersion,
    description: packageJson.description,
    main: packageJson.main,
    private: true
  })
}

async function prepareAppBundle() {
  const electronAppPath = join(rootDir, 'node_modules', 'electron', 'dist', 'Electron.app')
  const plistPath = join(appBundlePath, 'Contents', 'Info.plist')
  const macOsPath = join(appBundlePath, 'Contents', 'MacOS')
  const resourcesPath = join(appBundlePath, 'Contents', 'Resources')

  if (!existsSync(electronAppPath)) {
    throw new Error('Electron.app was not found. Run npm install first.')
  }

  await rm(stagingDir, { recursive: true, force: true })
  await mkdir(stagingDir, { recursive: true })
  await run('/usr/bin/ditto', [electronAppPath, appBundlePath])

  await renameIfExists(join(macOsPath, 'Electron'), join(macOsPath, appName))
  if (!existsSync(appIconIcnsPath) || !existsSync(appIconPngPath)) {
    throw new Error('TokenSmith app icons were not found in build-resources.')
  }

  await copyFile(appIconIcnsPath, join(resourcesPath, appIconIcnsFileName))
  await copyFile(appIconPngPath, join(resourcesPath, appIconPngFileName))
  await rm(join(resourcesPath, 'electron.icns'), { force: true })
  await patchPlist(plistPath, {
    CFBundleDisplayName: appName,
    CFBundleExecutable: appName,
    CFBundleIconFile: appIconIcnsFileName,
    CFBundleIdentifier: 'ai.tokensmith.desktop',
    CFBundleName: appName,
    CFBundleShortVersionString: appVersion,
    CFBundleVersion: appVersion,
    LSApplicationCategoryType: 'public.app-category.education'
  })

  await prepareAppPayload(resourcesPath)
}

async function signAppBundle() {
  if (process.env.TOKENSMITH_SKIP_CODESIGN === '1') {
    log('Skipping ad-hoc codesign because TOKENSMITH_SKIP_CODESIGN=1.')
    return
  }

  try {
    await run('/usr/bin/codesign', ['--force', '--deep', '--sign', '-', appBundlePath])
    log('Ad-hoc signed app bundle.')
  } catch (error) {
    log(`Ad-hoc codesign skipped: ${error instanceof Error ? error.message : String(error)}`)
  }
}

async function createDmg() {
  const readWriteDmgPath = join(stagingDir, `${appName}-${appVersion}-rw.dmg`)
  const mountPath = join(stagingDir, 'dmg-mount')
  const appSize = await run('/usr/bin/du', ['-sk', appBundlePath])
  const appSizeKb = Number.parseInt(appSize.stdout.split(/\s+/)[0] ?? '0', 10)
  const dmgSizeMb = Math.max(320, Math.ceil((appSizeKb / 1024) * 1.35 + 80))

  await rm(readWriteDmgPath, { recursive: true, force: true })
  await rm(dmgPath, { recursive: true, force: true })
  await rm(mountPath, { recursive: true, force: true })
  await mkdir(mountPath, { recursive: true })

  await run('/usr/bin/hdiutil', [
    'create',
    '-size',
    `${dmgSizeMb}m`,
    '-fs',
    'HFS+',
    '-volname',
    volumeName,
    '-ov',
    readWriteDmgPath
  ])

  try {
    await run('/usr/bin/hdiutil', ['attach', readWriteDmgPath, '-mountpoint', mountPath, '-nobrowse'])
    await run('/usr/bin/ditto', [appBundlePath, join(mountPath, basename(appBundlePath))])

    try {
      await symlink('/Applications', join(mountPath, 'Applications'))
    } catch {
      // The link may already exist on repeated local runs.
    }
  } finally {
    try {
      await run('/usr/bin/hdiutil', ['detach', mountPath])
    } catch {
      await run('/usr/bin/hdiutil', ['detach', mountPath, '-force'])
    }
  }

  await run('/usr/bin/hdiutil', ['convert', readWriteDmgPath, '-format', 'UDZO', '-o', dmgPath, '-ov'])
  await rm(readWriteDmgPath, { recursive: true, force: true })
  await rm(mountPath, { recursive: true, force: true })
}

if (process.platform !== 'darwin') {
  throw new Error('DMG packaging must be run on macOS.')
}

log(`Building ${appName} ${appVersion} for macOS ${arch}.`)
await run('npm', ['run', 'build'])
await prepareAppBundle()
await signAppBundle()
await createDmg()
log(`Created ${dmgPath}`)
