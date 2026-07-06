import { execFile } from 'node:child_process'
import { copyFile, cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const rootDir = dirname(dirname(fileURLToPath(import.meta.url)))
const packageJson = JSON.parse(await readFile(join(rootDir, 'package.json'), 'utf8'))
const appName = 'TokenSmith'
const appVersion = packageJson.version
const arch = process.arch
const releaseDir = join(rootDir, 'release')
const stagingDir = join(releaseDir, 'win')
const appDir = join(stagingDir, appName)
const zipPath = join(releaseDir, `${appName}-${appVersion}-win-${arch}.zip`)

function log(message) {
  console.log(`[package:win] ${message}`)
}

async function run(command, args, options = {}) {
  return execFileAsync(command, args, {
    cwd: rootDir,
    maxBuffer: 32 * 1024 * 1024,
    ...options
  })
}

async function runNpm(args) {
  if (process.platform === 'win32') {
    return run(process.env.ComSpec ?? 'cmd.exe', ['/d', '/s', '/c', ['npm', ...args].join(' ')])
  }

  return run('npm', args)
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
  const electronExePath = join(appDir, 'electron.exe')
  const appExePath = join(appDir, `${appName}.exe`)
  const resourcesPath = join(appDir, 'resources')

  if (!existsSync(join(electronDistPath, 'electron.exe'))) {
    throw new Error('Electron Windows runtime was not found. Run npm install on Windows first.')
  }

  await rm(stagingDir, { recursive: true, force: true })
  await mkdir(stagingDir, { recursive: true })
  await cp(electronDistPath, appDir, {
    recursive: true,
    preserveTimestamps: true
  })

  await rm(appExePath, { recursive: true, force: true })
  await copyFile(electronExePath, appExePath)
  await rm(electronExePath, { force: true })
  await prepareAppPayload(resourcesPath)
}

async function createZip() {
  await rm(zipPath, { recursive: true, force: true })
  await run('powershell.exe', [
    '-NoProfile',
    '-Command',
    `Compress-Archive -Path '${appDir.replaceAll("'", "''")}' -DestinationPath '${zipPath.replaceAll("'", "''")}' -Force`
  ])
}

if (process.platform !== 'win32') {
  throw new Error('Windows packaging must be run on Windows.')
}

log(`Building ${appName} ${appVersion} for Windows ${arch}.`)
await runNpm(['run', 'build'])
await preparePortableApp()
await createZip()
log(`Created ${zipPath}`)
