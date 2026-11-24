import { macDistribution } from './mac-distribution.mjs'
import { execFile } from 'node:child_process'
import { copyFile, cp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { existsSync, readdirSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'

const execFileAsync = promisify(execFile)
const rootDir = dirname(dirname(fileURLToPath(import.meta.url)))
const packageJson = JSON.parse(await readFile(join(rootDir, 'package.json'), 'utf8'))
const distribution = macDistribution()
let electronAppPath = join(rootDir, 'node_modules', 'electron', 'dist', 'Electron.app')
const appName = 'TokenSmith'
const appVersion = packageJson.version
const arch = process.arch
const releaseDir = join(rootDir, 'release')
const stagingDir = join(releaseDir, distribution.channel)
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
  await cp(from, to, { recursive: true, preserveTimestamps: true, verbatimSymlinks: true })
  await rm(from, { recursive: true, force: true })
}

function requirePythonRuntime(appRuntimePath) {
  const pythonExecutable = join(appRuntimePath, 'python', 'bin', 'python')
  if (!existsSync(pythonExecutable)) {
    throw new Error('TokenSmith Python runtime was not found. Run npm run setup:python-runtime before packaging.')
  }
}

async function verifyPackageArchitecture() {
  const binaryArch = { x64: 'x86_64', arm64: 'arm64' }[arch]
  if (!binaryArch) {
    throw new Error(`macOS packaging does not support the ${arch} architecture.`)
  }

  requirePythonRuntime(join(rootDir, 'app_runtime'))
  const binaries = [
    join(electronAppPath, 'Contents', 'MacOS', 'Electron'),
    join(rootDir, 'app_runtime', 'python', 'bin', 'python')
  ]
  for (const binary of binaries) {
    try {
      await run('/usr/bin/lipo', [binary, '-verify_arch', binaryArch])
    } catch {
      throw new Error(
        `The binary at ${binary} does not support macOS ${arch}. ` +
        `Reinstall Node dependencies and run npm run setup:python-runtime with ${arch} Node.js before packaging.`
      )
    }
  }
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

function bundledLibraryForLink(linkedLibrary, oldPythonRoot, pythonRoot, pythonLibrary) {
  if (linkedLibrary?.startsWith(`${oldPythonRoot}/`)) {
    return join(pythonRoot, linkedLibrary.slice(oldPythonRoot.length + 1))
  }
  if (linkedLibrary?.startsWith('/') && linkedLibrary.endsWith('/Python') && existsSync(pythonLibrary)) {
    return pythonLibrary
  }
  return null
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
        const bundledLibrary = bundledLibraryForLink(linkedLibrary, oldPythonRoot, pythonRoot, pythonLibrary)
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
    preserveTimestamps: true,
    verbatimSymlinks: true
  })
  await cp(join(rootDir, 'python_engine'), join(appPayloadPath, 'python_engine'), {
    recursive: true,
    preserveTimestamps: true,
    verbatimSymlinks: true,
    filter: (source) => !source.includes('__pycache__') && !source.endsWith('.pyc')
  })
  requirePythonRuntime(appRuntimePath)
  const packagedRuntimePath = join(appPayloadPath, 'app_runtime')
  await run('/usr/bin/ditto', [appRuntimePath, packagedRuntimePath])
  await patchMacPythonRuntime(packagedRuntimePath)
  await writeJson(join(appPayloadPath, 'package.json'), {
    name: packageJson.name,
    version: appVersion,
    description: packageJson.description,
    main: packageJson.main,
    private: true
  })
}

async function prepareAppBundle() {
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
    CFBundleIdentifier: distribution.mas ? distribution.bundleId : 'ai.tokensmith.desktop',
    CFBundleName: appName,
    CFBundleShortVersionString: appVersion,
    CFBundleVersion: appVersion,
    LSApplicationCategoryType: 'public.app-category.education',
    ...(arch === 'x64' ? { LSMinimumSystemVersion: '15.0' } : {})
  })

  if (distribution.mas) {
    const frameworks = join(appBundlePath, 'Contents', 'Frameworks')
    for (const entry of readdirSync(frameworks)) {
      if (!entry.endsWith('.app')) continue
      const suffix = entry.replace(/\.app$/, '').replace(/[^a-zA-Z0-9]+/g, '-').toLowerCase()
      await patchPlist(join(frameworks, entry, 'Contents', 'Info.plist'), { CFBundleIdentifier: `${distribution.bundleId}.${suffix}` })
    }
  }
  await prepareAppPayload(resourcesPath)
}

async function signAppBundle() {
  if (distribution.mas) {
    if (distribution.unsigned) {
      log('Unsigned MAS package: packaging checks only; not runnable or ready for submission.')
      return
    }
    const { signAsync } = await import('@electron/osx-sign')
    await signAsync({ app: appBundlePath, platform: 'mas', type: distribution.type,
      identity: distribution.identity, provisioningProfile: distribution.profile,
      optionsForFile: filePath => ({
        entitlements: join(rootDir, 'build-resources', 'mas', filePath === appBundlePath ? 'entitlements.plist' : 'entitlements-child.plist'),
        hardenedRuntime: false
      })
    })
    await run('/usr/bin/codesign', ['--verify', '--deep', '--strict', appBundlePath])
    return
  }
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
  await symlink('/Applications', join(stagingDir, 'Applications'))
  await run('/usr/bin/hdiutil', [
    'create',
    '-srcfolder',
    stagingDir,
    '-fs',
    'HFS+',
    '-volname',
    volumeName,
    '-format',
    'UDZO',
    '-ov',
    dmgPath
  ])
}

async function prepareMasElectron() {
  const { downloadArtifact } = await import('@electron/get')
  const { default: extract } = await import('extract-zip')
  const version = JSON.parse(await readFile(join(rootDir, 'node_modules', 'electron', 'package.json'), 'utf8')).version
  const archive = await downloadArtifact({ version, artifactName: 'electron', platform: 'mas', arch })
  const extracted = join(rootDir, 'node_modules', '.cache', 'tokensmith-mas', version, arch)
  await rm(extracted, { recursive: true, force: true })
  await mkdir(extracted, { recursive: true })
  await extract(archive, { dir: extracted })
  electronAppPath = join(extracted, 'Electron.app')
}

async function createMasArtifact() {
  const base = join(releaseDir, `${appName}-${appVersion}-${distribution.channel}-${arch}`)
  if (!distribution.unsigned && distribution.type === 'distribution') {
    await run('/usr/bin/productbuild', ['--component', appBundlePath, '/Applications', '--sign', distribution.installerIdentity, `${base}.pkg`])
    await run('/usr/sbin/pkgutil', ['--check-signature', `${base}.pkg`])
  } else {
    await rm(`${base}.zip`, { force: true })
    await run('/usr/bin/ditto', ['-c', '-k', '--sequesterRsrc', '--keepParent', appBundlePath, `${base}.zip`])
  }
  await writeJson(`${base}.build-info.json`, {
    channel: distribution.channel, version: appVersion, arch,
    bundleId: distribution.bundleId, signed: !distribution.unsigned,
    validation: distribution.unsigned ? 'Packaging only; Apple signing and sandbox runtime testing are still required.' : 'Signature verified; sandbox runtime and App Store review are still required.'
  })
  log(`Created ${base}`)
}

if (process.platform !== 'darwin') {
  throw new Error('macOS packaging must be run on macOS.')
}
if (distribution.mas) await prepareMasElectron()

log(`Building ${appName} ${appVersion} for macOS ${arch}.`)
await verifyPackageArchitecture()
await run('npm', ['run', 'build'])
await prepareAppBundle()
await signAppBundle()
if (distribution.mas) await createMasArtifact()
else { await createDmg(); log(`Created ${dmgPath}`) }
