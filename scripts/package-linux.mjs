import { execFile } from 'node:child_process'
import { chmod, copyFile, cp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const rootDir = dirname(dirname(fileURLToPath(import.meta.url)))
const packageJson = JSON.parse(await readFile(join(rootDir, 'package.json'), 'utf8'))
const appName = 'TokenSmith'
const packageName = 'tokensmith'
const appVersion = packageJson.version
const arch = process.arch
const debArch = arch === 'x64' ? 'amd64' : arch === 'arm64' ? 'arm64' : arch
const appImageArch = arch === 'x64' ? 'x86_64' : arch === 'arm64' ? 'aarch64' : arch
const releaseDir = join(rootDir, 'release')
const stagingDir = join(releaseDir, 'linux')
const appDir = join(stagingDir, appName)
const debRoot = join(stagingDir, 'deb-root')
const tarPath = join(releaseDir, `${appName}-${appVersion}-linux-${arch}.tar.gz`)
const appImagePath = join(releaseDir, `${appName}-${appVersion}-linux-${arch}.AppImage`)
const debPath = join(releaseDir, `${packageName}_${appVersion}_${debArch}.deb`)
const iconPngPath = join(rootDir, 'build-resources', 'tokensmith-icon.png')
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm'

function log(message) {
  console.log(`[package:linux] ${message}`)
}

async function run(command, args, options = {}) {
  return execFileAsync(command, args, {
    cwd: rootDir,
    maxBuffer: 32 * 1024 * 1024,
    ...options
  })
}

async function commandExists(command) {
  try {
    await run('which', [command])
    return true
  } catch {
    return false
  }
}

async function writeJson(filePath, value) {
  await writeFile(`${filePath}.tmp`, `${JSON.stringify(value, null, 2)}\n`, 'utf8')
  await copyFile(`${filePath}.tmp`, filePath)
  await rm(`${filePath}.tmp`)
}

async function copyIfExists(from, to) {
  if (!existsSync(from)) {
    return
  }

  await cp(from, to, {
    recursive: true,
    preserveTimestamps: true,
    filter: (source) => !source.includes('__pycache__') && !source.endsWith('.pyc')
  })
}

async function prepareAppPayload(resourcesPath) {
  const appPayloadPath = join(resourcesPath, 'app')

  await mkdir(appPayloadPath, { recursive: true })
  await cp(join(rootDir, 'out'), join(appPayloadPath, 'out'), {
    recursive: true,
    preserveTimestamps: true
  })
  await copyIfExists(join(rootDir, 'python_engine'), join(appPayloadPath, 'python_engine'))
  await copyIfExists(join(rootDir, 'app_runtime'), join(appPayloadPath, 'app_runtime'))
  await writeJson(join(appPayloadPath, 'package.json'), {
    name: packageJson.name,
    version: appVersion,
    description: packageJson.description,
    main: packageJson.main,
    private: true
  })
}

async function preparePortableApp() {
  const electronDistPath = join(rootDir, 'node_modules', 'electron', 'dist')
  const electronRuntimePath = join(electronDistPath, 'electron')
  const electronExePath = join(appDir, 'electron')
  const appExePath = join(appDir, appName)
  const resourcesPath = join(appDir, 'resources')

  if (!existsSync(electronRuntimePath)) {
    throw new Error('Electron Linux runtime was not found. Run npm install on Linux first.')
  }

  await rm(stagingDir, { recursive: true, force: true })
  await mkdir(stagingDir, { recursive: true })
  await cp(electronDistPath, appDir, {
    recursive: true,
    preserveTimestamps: true
  })

  await rm(appExePath, { recursive: true, force: true })
  await copyFile(electronExePath, appExePath)
  await chmod(appExePath, 0o755)
  await rm(electronExePath, { force: true })
  await prepareAppPayload(resourcesPath)
}

async function createTarball() {
  await rm(tarPath, { recursive: true, force: true })
  await run('tar', ['-czf', tarPath, '-C', stagingDir, appName])
}

async function createDebPackage() {
  const optAppPath = join(debRoot, 'opt', appName)
  const desktopPath = join(debRoot, 'usr', 'share', 'applications', `${packageName}.desktop`)
  const iconPath = join(debRoot, 'usr', 'share', 'icons', 'hicolor', '512x512', 'apps', `${packageName}.png`)
  const launcherPath = join(debRoot, 'usr', 'bin', packageName)
  const controlPath = join(debRoot, 'DEBIAN', 'control')

  await rm(debRoot, { recursive: true, force: true })
  await mkdir(dirname(controlPath), { recursive: true })
  await mkdir(dirname(desktopPath), { recursive: true })
  await mkdir(dirname(iconPath), { recursive: true })
  await mkdir(dirname(launcherPath), { recursive: true })
  await cp(appDir, optAppPath, { recursive: true, preserveTimestamps: true })

  if (existsSync(iconPngPath)) {
    await copyFile(iconPngPath, iconPath)
  }

  await symlink(`/opt/${appName}/${appName}`, launcherPath)
  await writeFile(
    desktopPath,
    `[Desktop Entry]
Name=${appName}
Comment=${packageJson.description}
Exec=/opt/${appName}/${appName} %U
Terminal=false
Type=Application
Icon=${packageName}
Categories=Education;Utility;
StartupWMClass=${appName}
`,
    'utf8'
  )

  await writeFile(
    controlPath,
    `Package: ${packageName}
Version: ${appVersion}
Section: education
Priority: optional
Architecture: ${debArch}
Maintainer: TokenSmith <support@tokensmith.local>
Description: ${packageJson.description}
Depends: libgtk-3-0, libnss3, libxss1, libasound2, libatk-bridge2.0-0, libdrm2, libgbm1, libxcomposite1, libxdamage1, libxrandr2, libxkbcommon0
`,
    'utf8'
  )

  await rm(debPath, { recursive: true, force: true })
  await run('dpkg-deb', ['--build', debRoot, debPath])
}

async function createAppImage() {
  if (!(await commandExists('appimagetool'))) {
    log('Skipping AppImage because appimagetool was not found.')
    return false
  }

  const appImageRoot = join(stagingDir, `${appName}.AppDir`)
  const optAppPath = join(appImageRoot, 'opt', appName)
  const appRunPath = join(appImageRoot, 'AppRun')
  const desktopPath = join(appImageRoot, `${packageName}.desktop`)
  const iconPath = join(appImageRoot, `${packageName}.png`)

  await rm(appImageRoot, { recursive: true, force: true })
  await mkdir(appImageRoot, { recursive: true })
  await cp(appDir, optAppPath, { recursive: true, preserveTimestamps: true })

  if (existsSync(iconPngPath)) {
    await copyFile(iconPngPath, iconPath)
  }

  await writeFile(
    desktopPath,
    `[Desktop Entry]
Name=${appName}
Comment=${packageJson.description}
Exec=${appName} %U
Terminal=false
Type=Application
Icon=${packageName}
Categories=Education;Utility;
StartupWMClass=${appName}
`,
    'utf8'
  )

  await writeFile(
    appRunPath,
    `#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/opt/${appName}/${appName}" "$@"
`,
    'utf8'
  )
  await chmod(appRunPath, 0o755)

  await rm(appImagePath, { recursive: true, force: true })
  await run('appimagetool', [appImageRoot, appImagePath], {
    env: {
      ...process.env,
      ARCH: appImageArch
    }
  })

  return true
}

if (process.platform !== 'linux') {
  throw new Error('Linux packaging must be run on Linux.')
}

log(`Building ${appName} ${appVersion} for Linux ${arch}.`)
await run(npmCommand, ['run', 'build'])
await preparePortableApp()
await createTarball()
await createDebPackage()
const appImageCreated = await createAppImage()
log(`Created ${tarPath}`)
log(`Created ${debPath}`)
if (appImageCreated) {
  log(`Created ${appImagePath}`)
}
