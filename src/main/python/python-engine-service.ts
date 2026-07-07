import { app, BrowserWindow } from 'electron'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { appendFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { delimiter, dirname, join } from 'node:path'
import { createInterface } from 'node:readline'
import type { ChatSource, CourseMaterial, LocalModel, MaterialIndexProgress } from '../../shared/app-state'
import type { CleaningPreviewResult, EngineChatRequest, EngineChatResponse } from '../../shared/engine'
import type { CleaningProfileId, CleaningRuleId } from '../../shared/cleaning'

interface PythonRequest {
  id: string
  command:
    | 'health'
    | 'index_material'
    | 'search'
    | 'chat'
    | 'list_materials'
    | 'set_material_enabled'
    | 'remove_material'
    | 'resolve_source_document'
    | 'preview_cleaning'
  payload: Record<string, unknown>
}

interface PythonResponse<T> {
  id: string
  ok: boolean
  result?: T
  error?: string
  progress?: MaterialIndexProgress
}

interface HealthResult {
  ok: boolean
  engine: 'python'
  llamaCppAvailable: boolean
  supports: string[]
}

interface IndexMaterialResult {
  material: CourseMaterial
}

interface SearchResult {
  sources: ChatSource[]
}

interface ListMaterialsResult {
  materials: CourseMaterial[]
}

interface ResolvedSourceDocument {
  chunkId?: number | string
  documentId?: number | string
  path: string
  title?: string
  page?: number
  collectionName?: string
  thumbnailPath?: string
}

interface ResolveSourceDocumentResult {
  source: ResolvedSourceDocument | null
}

let worker: ChildProcessWithoutNullStreams | null = null
let workerBuffer = ''
const pendingRequests = new Map<
  string,
  {
    command: PythonRequest['command']
    materialId?: string
    reject: (reason?: unknown) => void
    onProgress?: (progress: MaterialIndexProgress) => void
    resolve: (value: unknown) => void
    timeout: NodeJS.Timeout
    timeoutMs: number
  }
>()

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function getLogFilePath(): string {
  return join(app.getPath('userData'), 'logs', 'tokensmith.log')
}

function localIsoTimestamp(date = new Date()): string {
  const pad = (value: number, size = 2) => String(value).padStart(size, '0')

  const offsetMinutes = -date.getTimezoneOffset()
  const sign = offsetMinutes >= 0 ? '+' : '-'
  const absOffset = Math.abs(offsetMinutes)
  const offsetHours = Math.floor(absOffset / 60)
  const offsetMins = absOffset % 60

  return [
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`,
    `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}`,
    `${sign}${pad(offsetHours)}:${pad(offsetMins)}`
  ].join('')
}

function writeLog(event: string, detail: Record<string, unknown> = {}): void {
  const logPath = getLogFilePath()
  const payload = {
    time: localIsoTimestamp(),
    event,
    ...detail
  }

  try {
    mkdirSync(dirname(logPath), { recursive: true })
    appendFileSync(logPath, `${JSON.stringify(payload)}\n`, 'utf8')
  } catch {
    // Logging must never block chat or indexing.
  }
}

function getPythonExecutable(): string {
  const runtimeCandidates =
    process.platform === 'win32'
      ? [
          join(app.getAppPath(), 'app_runtime', 'python', 'python.exe'),
          join(app.getAppPath(), 'app_runtime', 'python', 'Scripts', 'python.exe'),
          join(process.resourcesPath, 'app', 'app_runtime', 'python', 'python.exe'),
          join(process.resourcesPath, 'app', 'app_runtime', 'python', 'Scripts', 'python.exe')
        ]
      : [
          join(app.getAppPath(), 'app_runtime', 'python', 'bin', 'python'),
          join(process.resourcesPath, 'app', 'app_runtime', 'python', 'bin', 'python')
        ]

  for (const candidate of runtimeCandidates) {
    if (existsSync(candidate)) {
      return candidate
    }
  }

  throw new Error('The bundled TokenSmith Python runtime was not found. Run npm run setup:python-runtime before starting the app locally.')
}

function getWorkerPath(): string {
  const devPath = join(app.getAppPath(), 'python_engine', 'tokensmith_engine.py')
  if (existsSync(devPath)) {
    return devPath
  }

  return join(process.resourcesPath, 'python_engine', 'tokensmith_engine.py')
}

function resolveModel(model?: LocalModel): LocalModel | undefined {
  if (!model) {
    return undefined
  }

  return { ...model }
}

function modelCanEmbed(model?: LocalModel): boolean {
  return model?.role === 'embedder' || model?.role === 'both'
}

function resolveEmbeddingModel(model?: LocalModel): LocalModel | undefined {
  if (!model || !modelCanEmbed(model)) {
    return undefined
  }

  if (model.engine === 'remote') {
    return model
  }

  const embeddingPath = model.path ?? model.embeddingPath
  if (!embeddingPath) {
    return undefined
  }

  return {
    ...model,
    role: model.role ?? 'embedder',
    embeddingPath
  }
}

function resolveEmbeddingModels(models?: LocalModel[]): LocalModel[] {
  return (models ?? [])
    .map((model) => resolveEmbeddingModel(model))
    .filter((model): model is LocalModel => Boolean(model))
}

function appPythonEnv(pythonExecutable: string): Record<string, string> {
  if (process.platform === 'win32') {
    return {}
  }

  const runtimeRoot = dirname(dirname(pythonExecutable))
  const libRoot = join(runtimeRoot, 'lib')
  const pythonLibName = existsSync(libRoot)
    ? readdirSync(libRoot).find((name) => /^python3\.\d+$/.test(name))
    : undefined
  const pythonLib = pythonLibName ? join(libRoot, pythonLibName) : join(libRoot, 'python3.10')
  const sitePackages = join(pythonLib, 'site-packages')

  if (!pythonExecutable.includes(`${join('app_runtime', 'python', 'bin', 'python')}`) || !existsSync(pythonLib)) {
    return {}
  }

  const pythonPathParts = []
  const hasStdlib = existsSync(join(pythonLib, 'encodings', '__init__.py'))
  if (hasStdlib) {
    pythonPathParts.push(pythonLib)
  }
  if (existsSync(sitePackages)) {
    pythonPathParts.push(sitePackages)
  }
  if (process.env.PYTHONPATH) {
    pythonPathParts.push(process.env.PYTHONPATH)
  }

  const runtimeEnv: Record<string, string> = {
    ...(hasStdlib ? { PYTHONHOME: runtimeRoot } : {}),
    ...(pythonPathParts.length ? { PYTHONPATH: pythonPathParts.join(delimiter) } : {})
  }

  return process.platform === 'darwin'
    ? {
        ...runtimeEnv,
        DYLD_FALLBACK_LIBRARY_PATH: libRoot
      }
    : {
        ...runtimeEnv,
        LD_LIBRARY_PATH: libRoot
      }
}

function rejectPendingRequests(reason: Error): void {
  for (const request of pendingRequests.values()) {
    clearTimeout(request.timeout)
    request.reject(reason)
  }
  pendingRequests.clear()
}

function killWorkerProcess(activeWorker: ChildProcessWithoutNullStreams, signal: NodeJS.Signals): void {
  const pid = activeWorker.pid

  if (pid && process.platform !== 'win32') {
    try {
      process.kill(-pid, signal)
      return
    } catch {
      // Fall back to killing the direct child when process-group termination is unavailable.
    }
  }

  activeWorker.kill(signal)
}

function terminateWorker(reason: string): void {
  const activeWorker = worker
  if (!activeWorker) {
    return
  }

  writeLog('python_worker_terminating', {
    reason,
    pid: activeWorker.pid
  })

  if (worker === activeWorker) {
    worker = null
  }

  killWorkerProcess(activeWorker, 'SIGTERM')

  const forceKill = setTimeout(() => {
    if (activeWorker.exitCode === null && activeWorker.signalCode === null) {
      writeLog('python_worker_force_kill', {
        reason,
        pid: activeWorker.pid
      })
      killWorkerProcess(activeWorker, 'SIGKILL')
    }
  }, 1500)

  forceKill.unref?.()
}

function createRequestTimeout(
  id: string,
  command: PythonRequest['command'],
  timeoutMs: number,
  reject: (reason?: unknown) => void
): NodeJS.Timeout {
  return setTimeout(() => {
    pendingRequests.delete(id)
    writeLog('python_request_timeout', { id, command, timeoutMs })
    reject(new Error(`The Python engine timed out while running ${command}.`))
    if (command === 'index_material') {
      terminateWorker('index_material_timeout')
    }
  }, timeoutMs)
}

function handleWorkerLine(line: string): void {
  if (!line.trim()) {
    return
  }

  let response: PythonResponse<unknown>
  try {
    response = JSON.parse(line) as PythonResponse<unknown>
  } catch {
    return
  }

  const pending = pendingRequests.get(response.id)
  if (!pending) {
    return
  }

  if (response.progress) {
    clearTimeout(pending.timeout)
    pending.timeout = createRequestTimeout(response.id, pending.command, pending.timeoutMs, pending.reject)
    pending.onProgress?.(response.progress)
    return
  }

  clearTimeout(pending.timeout)
  pendingRequests.delete(response.id)

  if (!response.ok) {
    writeLog('python_request_error', {
      id: response.id,
      error: response.error ?? 'The Python engine returned an error.'
    })
    pending.reject(new Error(response.error ?? 'The Python engine returned an error.'))
    return
  }

  writeLog('python_request_success', { id: response.id })
  pending.resolve(response.result)
}

function ensureWorker(): ChildProcessWithoutNullStreams {
  if (worker && !worker.killed) {
    return worker
  }

  const scriptPath = getWorkerPath()
  if (!existsSync(scriptPath)) {
    throw new Error('The Python engine worker script was not found.')
  }

  const pythonExecutable = getPythonExecutable()
  const logFile = getLogFilePath()

  writeLog('python_worker_starting', {
    pythonExecutable,
    scriptPath,
    userDataPath: app.getPath('userData')
  })

  const spawnedWorker = spawn(pythonExecutable, [scriptPath], {
    detached: process.platform !== 'win32',
    env: {
      ...process.env,
      ...appPythonEnv(pythonExecutable),
      TOKENSMITH_LOG_FILE: logFile,
      PYTHONIOENCODING: 'utf-8'
    },
    stdio: 'pipe'
  })
  worker = spawnedWorker

  workerBuffer = ''

  spawnedWorker.stdout.on('data', (chunk: Buffer) => {
    workerBuffer += chunk.toString('utf8')
    const lines = workerBuffer.split(/\r?\n/)
    workerBuffer = lines.pop() ?? ''
    for (const line of lines) {
      handleWorkerLine(line)
    }
  })

  spawnedWorker.stderr.on('data', (chunk: Buffer) => {
    writeLog('python_worker_stderr', {
      text: chunk.toString('utf8').slice(0, 4000)
    })
  })

  spawnedWorker.once('error', (error) => {
    writeLog('python_worker_error', { message: error.message })
    if (worker === spawnedWorker) {
      rejectPendingRequests(error)
      worker = null
    }
  })

  spawnedWorker.once('exit', (code, signal) => {
    writeLog('python_worker_exit', { code, signal })
    if (worker === spawnedWorker) {
      rejectPendingRequests(new Error('The Python engine stopped.'))
      worker = null
    }
  })

  app.once('before-quit', () => {
    terminateWorker('app_before_quit')
  })

  return spawnedWorker
}

async function requestPython<T>(
  command: PythonRequest['command'],
  payload: PythonRequest['payload'],
  timeoutMs = 60_000,
  onProgress?: (progress: MaterialIndexProgress) => void,
  materialId?: string
): Promise<T> {
  const activeWorker = ensureWorker()
  const id = createId('py')
  const request: PythonRequest = {
    id,
    command,
    payload
  }

  return new Promise<T>((resolve, reject) => {
    const timeout = createRequestTimeout(id, command, timeoutMs, reject)

    pendingRequests.set(id, {
      command,
      materialId,
      resolve: resolve as (value: unknown) => void,
      reject,
      onProgress,
      timeout,
      timeoutMs
    })

    writeLog('python_request_start', { id, command })
    activeWorker.stdin.write(`${JSON.stringify(request)}\n`, 'utf8')
  })
}

export async function getPythonEngineHealth(): Promise<HealthResult> {
  return requestPython<HealthResult>('health', {}, 30_000)
}

function sendIndexProgress(progress: MaterialIndexProgress): void {
  for (const window of BrowserWindow.getAllWindows()) {
    window.webContents.send('library:index-progress', progress)
  }
}

export async function indexMaterialWithPython(
  materialPath: string,
  model?: LocalModel,
  materialId?: string,
  options?: {
    resume?: boolean
    title?: string
    cleaningProfileId?: CleaningProfileId
    cleaningRuleIds?: CleaningRuleId[]
  }
): Promise<CourseMaterial> {
  const result = await requestPython<IndexMaterialResult>(
    'index_material',
    {
      path: materialPath,
      materialId,
      resume: options?.resume === true,
      title: options?.title,
      cleaningProfileId: options?.cleaningProfileId,
      cleaningRuleIds: options?.cleaningRuleIds,
      model: resolveEmbeddingModel(model),
      userDataPath: app.getPath('userData')
    },
    180_000,
    sendIndexProgress,
    materialId
  )

  return result.material
}

export async function previewCleaningWithPython(
  materialPath: string,
  options?: {
    cleaningProfileId?: CleaningProfileId
    cleaningRuleIds?: CleaningRuleId[]
  }
): Promise<CleaningPreviewResult> {
  return requestPython<CleaningPreviewResult>(
    'preview_cleaning',
    {
      path: materialPath,
      cleaningProfileId: options?.cleaningProfileId,
      cleaningRuleIds: options?.cleaningRuleIds,
      userDataPath: app.getPath('userData')
    },
    60_000
  )
}

export async function cancelMaterialIndexingWithPython(materialId: string): Promise<void> {
  const cancelledRequestIds: string[] = []

  for (const [id, request] of pendingRequests) {
    if (request.command !== 'index_material' || request.materialId !== materialId) {
      continue
    }

    clearTimeout(request.timeout)
    pendingRequests.delete(id)
    request.reject(new Error('Indexing was cancelled.'))
    cancelledRequestIds.push(id)
  }

  if (cancelledRequestIds.length === 0) {
    writeLog('python_request_cancel_skipped', { materialId })
    return
  }

  writeLog('python_request_cancelled', {
    materialId,
    requestIds: cancelledRequestIds
  })
  terminateWorker('index_material_cancelled')
}

export async function searchLibraryWithPython(
  query: string,
  materials: CourseMaterial[],
  limit: number,
  embeddingModels?: LocalModel[]
): Promise<ChatSource[]> {
  const resolvedEmbeddingModels = resolveEmbeddingModels(embeddingModels)
  const result = await requestPython<SearchResult>(
    'search',
    {
      query,
      materials,
      limit,
      embeddingModels: resolvedEmbeddingModels,
      userDataPath: app.getPath('userData')
    },
    30_000
  )

  return result.sources
}

export async function runPythonStudyEngine(request: EngineChatRequest): Promise<EngineChatResponse> {
  return requestPython<EngineChatResponse>(
    'chat',
    {
      ...request,
      model: resolveModel(request.model),
      userDataPath: app.getPath('userData')
    },
    180_000
  )
}

export async function listIndexedMaterialsWithPython(): Promise<CourseMaterial[]> {
  const result = await requestPython<ListMaterialsResult>(
    'list_materials',
    {
      userDataPath: app.getPath('userData')
    },
    30_000
  )

  return result.materials
}

export async function setMaterialEnabledWithPython(materialId: string, isActive: boolean): Promise<void> {
  await requestPython<{ ok: boolean }>(
    'set_material_enabled',
    {
      materialId,
      isActive,
      userDataPath: app.getPath('userData')
    },
    30_000
  )
}

export async function removeMaterialWithPython(materialId: string, materialPath?: string): Promise<void> {
  await requestPython<{ ok: boolean }>(
    'remove_material',
    {
      materialId,
      path: materialPath,
      userDataPath: app.getPath('userData')
    },
    30_000
  )
}

export async function resolveSourceDocumentWithPython(source: ChatSource): Promise<ResolvedSourceDocument | null> {
  const result = await requestPython<ResolveSourceDocumentResult>(
    'resolve_source_document',
    {
      source,
      userDataPath: app.getPath('userData')
    },
    30_000
  )

  return result.source
}
