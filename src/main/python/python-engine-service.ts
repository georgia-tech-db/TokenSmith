import { localDocuments, qualifyMaterial, qualifyDocument, qualifySource, mergeCollectionSources } from '../collections/collection-routing'
import { CollectionRegistry, type MountedCollection } from '../collections/collection-registry'
import { collectionMaterialId, splitMaterialId } from '../../shared/collections'
import type { IndexMaterialOptions, PreparationReport } from '../../shared/preparation'
import type { StudyDocument, StudyDocumentRef, LibrarySearchOptions } from '../../shared/study-scope'
import { normalizeEmbeddingGpuEnabled } from '../../shared/embedding-settings'
import { createEmbeddingDeviceManager } from '../engine/embedding-device'
import { modelWithRememberedRemoteApiKey } from '../engine/remote-model-secrets'
import { app, BrowserWindow } from 'electron'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { delimiter, dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import type { ChatSource, CourseMaterial, LocalModel, MaterialIndexProgress, SearchMode } from '../../shared/app-state'
import type { CleaningPreviewResult, TokenSmithLogFile } from '../../shared/engine'
import type { CleaningProfileId, CleaningRuleId } from '../../shared/cleaning'

interface PythonRequest {
  id: string
  command:
    | 'health'
    | 'validate_collection'
    | 'create_collection'
    | 'index_material'
    | 'search'
    | 'starter_sources'
    | 'study_documents'
    | 'practice_sources'
    | 'list_materials'
    | 'set_material_enabled'
    | 'remove_material'
    | 'resolve_source_document'
    | 'preview_cleaning'
    | 'preparation_report'
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
  supports: string[]
}

interface IndexMaterialResult {
  material: CourseMaterial
}

interface SearchResult {
  sources: ChatSource[]
  queryTerms?: string[]
  keywordTerms?: string[]
}

interface ListMaterialsResult {
  materials: CourseMaterial[]
}

interface ResolvedSourceDocument {
  chunkId?: number | string
  chunkRowid?: number | string
  chunkKind?: string
  tokensmithChunkId?: string
  tokensmithChapter?: string
  tokensmithChunkKind?: string
  documentId?: number | string
  path: string
  title?: string
  page?: number
  lineFrom?: number
  lineTo?: number
  collectionName?: string
  thumbnailPath?: string
}

interface ResolveSourceDocumentResult {
  source: ResolvedSourceDocument | null
}

const prepareEmbeddingDevice = createEmbeddingDeviceManager()

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

const visibleLogEvents = new Set([
  'chat_request_context',
  'chat_response_context',
  'chat_runtime_context_budget',
  'chat_question_rewrite',
  'follow_up_suggestions',
  'initial_question_suggestions',
  'question_suggestion_request_context',
  'question_suggestion_runtime_context_budget',
  'library_search_request',
  'library_search_result'
])

function shouldWriteLog(event: string): boolean {
  return visibleLogEvents.has(event) || /(?:error|failed|failure|timeout|stderr|force_kill)/i.test(event)
}

function writeLog(event: string, detail: Record<string, unknown> = {}): void {
  if (!shouldWriteLog(event)) {
    return
  }

  const payload = {
    time: localIsoTimestamp(),
    event,
    ...detail
  }

  try {
    const logPath = getLogFilePath()
    mkdirSync(dirname(logPath), { recursive: true })
    appendFileSync(logPath, `${JSON.stringify(payload)}\n`, 'utf8')
  } catch {
    // Logging must never block chat or indexing.
  }
}

export function writeTokenSmithLog(event: string, detail: Record<string, unknown> = {}): void {
  writeLog(event, detail)
}

export function readTokenSmithLogFile(maxBytes = 512_000): TokenSmithLogFile {
  const logPath = getLogFilePath()
  if (!existsSync(logPath)) {
    return {
      path: logPath,
      text: '',
      sizeBytes: 0,
      maxBytes,
      truncated: false
    }
  }

  const sizeBytes = statSync(logPath).size
  const text = readFileSync(logPath, 'utf8')
  const truncated = Buffer.byteLength(text, 'utf8') > maxBytes

  return {
    path: logPath,
    text: truncated ? text.slice(-maxBytes) : text,
    sizeBytes,
    maxBytes,
    truncated
  }
}

function logSource(source: ChatSource): Record<string, unknown> {
  return {
    title: source.title,
    locator: source.locator,
    sourceId: source.sourceId,
    chunkId: source.chunkId,
    chunkRowid: source.chunkRowid,
    chunkKind: source.chunkKind,
    sourceUnitId: source.sourceUnitId,
    sourceUnitComplete: source.sourceUnitComplete,
    sourceChunkIds: source.sourceChunkIds,
    tokensmithChunkId: source.tokensmithChunkId,
    tokensmithChapter: source.tokensmithChapter,
    tokensmithChunkKind: source.tokensmithChunkKind,
    queryTerms: source.queryTerms,
    keywordTerms: source.keywordTerms,
    documentTitle: source.documentTitle,
    collectionName: source.collectionName,
    sectionHeader: source.sectionHeader,
    path: source.path,
    pageStart: source.pageStart,
    pageEnd: source.pageEnd,
    score: source.score,
    retrievalMode: source.retrievalMode,
    embeddingModel: source.embeddingModel,
    chunkEmbeddingModel: source.chunkEmbeddingModel,
    excerpt: source.excerpt
  }
}

function getPythonRuntimeRoot(): string {
  return app.isPackaged
    ? join(process.resourcesPath, 'app', 'app_runtime', 'python')
    : join(app.getAppPath(), 'app_runtime', 'python')
}

function getPythonExecutable(): string {
  const runtimeRoot = getPythonRuntimeRoot()
  const pythonExecutable = process.platform === 'win32'
    ? join(runtimeRoot, 'python.exe')
    : join(runtimeRoot, 'bin', 'python')

  if (existsSync(pythonExecutable)) {
    return pythonExecutable
  }

  throw new Error(
    app.isPackaged
      ? 'The TokenSmith installation is incomplete because its private Python runtime is missing. Reinstall TokenSmith.'
      : 'The TokenSmith Python runtime was not found. Run npm run setup:python-runtime before starting the app locally.'
  )
}

function getWorkerPath(): string {
  const devPath = join(app.getAppPath(), 'python_engine', 'tokensmith_engine.py')
  if (existsSync(devPath)) {
    return devPath
  }

  return join(process.resourcesPath, 'python_engine', 'tokensmith_engine.py')
}

function modelCanEmbed(model?: LocalModel): boolean {
  return model?.role === 'embedder' || model?.role === 'both'
}

function resolveEmbeddingModel(model?: LocalModel): LocalModel | undefined {
  if (!model || !modelCanEmbed(model)) {
    return undefined
  }

  if (model.engine === 'remote' || model.engine === 'ollama') {
    return model
  }

  return undefined
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
  const runtimeEnv: Record<string, string> = {
    ...(hasStdlib ? { PYTHONHOME: runtimeRoot } : {}),
    ...(pythonPathParts.length ? { PYTHONPATH: pythonPathParts.join(delimiter) } : {}),
    PYTHONIOENCODING: 'utf-8',
    // Keep Python caches out of the signed application bundle.
    PYTHONDONTWRITEBYTECODE: '1',
    PYTHONNOUSERSITE: '1'
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

const terminatingWorkers = new WeakMap<ChildProcessWithoutNullStreams, Promise<void>>()
function terminatePythonWorker(child: ChildProcessWithoutNullStreams): Promise<void> {
  const pending = terminatingWorkers.get(child)
  if (pending) return pending
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve()
  const termination = new Promise<void>(resolve => {
    const kill = (signal: NodeJS.Signals) => {
      if (child.pid && process.platform !== 'win32') {
        try { process.kill(-child.pid, signal); return } catch { /* Fall back to the child. */ }
      }
      child.kill(signal)
    }
    const forceKill = setTimeout(() => kill('SIGKILL'), 1500)
    forceKill.unref?.()
    child.once('close', () => { clearTimeout(forceKill); resolve() })
    kill('SIGTERM')
  })
  terminatingWorkers.set(child, termination)
  return termination
}

function createPythonWorker() {
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

  function rejectPendingRequests(reason: Error): void {
    for (const request of pendingRequests.values()) {
      clearTimeout(request.timeout)
      request.reject(reason)
    }
    pendingRequests.clear()
  }

  function terminateWorker(reason: string): Promise<void> {
    const activeWorker = worker
    if (!activeWorker) return Promise.resolve()
    writeLog('python_worker_terminating', { reason, pid: activeWorker.pid })
    worker = null
    return terminatePythonWorker(activeWorker)
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
      if (worker !== spawnedWorker) return
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


    return spawnedWorker
  }

  async function request<T>(
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

  return { request, close() { rejectPendingRequests(new Error('The library was closed.')); return terminateWorker('collection_unloaded') } }
}

const workers = new Map<string, ReturnType<typeof createPythonWorker>>()
async function closeStore(path: string) { const worker = workers.get(path); workers.delete(path); await worker?.close() }
app?.on('before-quit', () => { for (const worker of workers.values()) worker.close(); workers.clear() })
async function requestPython<T>(command: PythonRequest['command'], payload: PythonRequest['payload'], timeoutMs = 60_000,
  onProgress?: (progress: MaterialIndexProgress) => void, materialId?: string): Promise<T> {
  if (command === 'index_material') return isolatedIndexRequest<T>(payload, timeoutMs, onProgress, materialId)
  const path = String(payload.userDataPath ?? 'runtime')
  let worker = workers.get(path)
  if (!worker) { worker = createPythonWorker(); workers.set(path, worker) }
  return worker.request<T>(command, payload, timeoutMs, onProgress, materialId)
}

let registry: CollectionRegistry | undefined
export function collections(): CollectionRegistry {
  return registry ??= new CollectionRegistry(join(app.getPath('userData'), 'collections.json'), {
    validate: async path => { await requestPython('validate_collection', { userDataPath: path }) },
    create: async path => { await requestPython('create_collection', { userDataPath: path }) },
    close: closeStore
  })
}

// Indexing has its own worker so long preparation jobs do not block source search.
const indexingJobs = new Map<string, () => Promise<void>>()
let indexingQueue: Promise<unknown> = Promise.resolve()
app?.on('before-quit', () => { for (const stop of indexingJobs.values()) stop() })

function isolatedIndexRequest<T>(payload: PythonRequest['payload'], timeoutMs: number,
  onProgress?: (progress: MaterialIndexProgress) => void, materialId = createId('index')): Promise<T> {
  let cancelled = false
  let child: ChildProcessWithoutNullStreams | undefined
  let rejectJob: ((error: Error) => void) | undefined
  const stopWorker = () => child ? terminatePythonWorker(child) : Promise.resolve()
  const stop = async () => {
    cancelled = true
    rejectJob?.(new Error('Indexing was cancelled.'))
    await stopWorker()
  }
  indexingJobs.set(materialId, stop)
  const run = async (): Promise<T> => {
    if (cancelled) throw new Error('Indexing was cancelled.')
    if (payload.model) await prepareEmbeddingDevice([payload.model as LocalModel], normalizeEmbeddingGpuEnabled(payload.embeddingGpuEnabled))
    if (cancelled) throw new Error('Indexing was cancelled.')
    return new Promise<T>((resolve, reject) => {
      const python = getPythonExecutable()
      child = spawn(python, [getWorkerPath()], {
        detached: process.platform !== 'win32',
        env: { ...process.env, ...appPythonEnv(python), TOKENSMITH_LOG_FILE: getLogFilePath(), PYTHONIOENCODING: 'utf-8' },
        stdio: 'pipe'
      })
      let buffer = ''
      let settled = false
      const finish = (error?: Error, result?: T) => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        void stopWorker().then(() => {
          if (error) reject(error)
          else resolve(result as T)
        })
      }
      const timer = setTimeout(() => finish(new Error('Preparation stopped responding. Resume to continue from saved work.')), timeoutMs)
      rejectJob = error => finish(error)
      child.stdout.on('data', (data: Buffer) => {
        buffer += data.toString('utf8')
        const lines = buffer.split(/\r?\n/)
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          let message: PythonResponse<T>
          try { message = JSON.parse(line) as PythonResponse<T> } catch { continue }
          if (settled) continue
          if (message.progress) { timer.refresh(); onProgress?.(message.progress); continue }
          if (message.id === materialId) finish(message.ok ? undefined : new Error(message.error || 'Preparation failed.'), message.result)
        }
      })
      child.stderr.on('data', () => { /* Worker logs directly to its configured log file. */ })
      child.once('error', error => finish(error))
      child.once('exit', () => { if (!settled) finish(new Error('The preparation worker stopped. Resume to continue.')) })
      child.stdin.end(JSON.stringify({ id: materialId, command: 'index_material', payload }) + '\n')
    })
  }
  const result = indexingQueue.then(run).finally(() => { if (indexingJobs.get(materialId) === stop) indexingJobs.delete(materialId) })
  indexingQueue = result.catch(() => undefined)
  return result
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
  options?: IndexMaterialOptions
): Promise<CourseMaterial> {
  if (!materialId) throw new Error('Open a library before adding documents.')
  const { collection, localId } = collections().forMaterial(materialId)
  await collections().revise(collection)
  collections().current(collection)
  const result = await requestPython<IndexMaterialResult>(
    'index_material',
    {
      path: materialPath,
      materialId: localId,
      resume: options?.resume === true,
      title: options?.title,
      cleaningProfileId: options?.cleaningProfileId,
      cleaningRuleIds: options?.cleaningRuleIds,
      model: resolveEmbeddingModel(model ? modelWithRememberedRemoteApiKey(model) : undefined),
      preparation: options?.preparation,
      preparationModel: options?.preparationModel ? modelWithRememberedRemoteApiKey(options.preparationModel) : undefined,
      embeddingGpuEnabled: normalizeEmbeddingGpuEnabled(options?.embeddingGpuEnabled),
      userDataPath: collection.path
    },
    180_000,
    progress => { if (collections().loaded().some(item => item.epoch === collection.epoch)) sendIndexProgress({ ...progress, materialId }) },
    materialId
  )

  collections().current(collection)
  await collections().revise(collection)
  return qualifyMaterial(collections(), collection, result.material)
}

export async function preparationReportWithPython(path: string, documentPath?: string, materialId?: string): Promise<PreparationReport> {
  if (!materialId) throw new Error('Choose a document.')
  const { collection, localId } = collections().forMaterial(materialId)
  const report = await storeRequest<PreparationReport>(collection, 'preparation_report', {
    path: collections().originalPath(collection, path, localId),
    documentPath: documentPath ? collections().originalPath(collection, documentPath, localId) : undefined })
  return { ...report, documents: report.documents.map(d => ({ ...d, path: collections().sourcePath(collection, d.path, localId) })) }
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

export async function cancelMaterialIndexingWithPython(materialId: string): Promise<void> { await indexingJobs.get(materialId)?.() }

export async function cancelCollectionIndexing(collectionId: string): Promise<void> {
  await Promise.all([...indexingJobs].filter(([id]) => id.startsWith(`${collectionId}/`)).map(([, stop]) => stop()))
}

async function storeRequest<T>(collection: MountedCollection, command: PythonRequest['command'], payload: Record<string, unknown>, timeout = 30_000): Promise<T> {
  collections().current(collection)
  const revision = collection.indexRevision
  const result = await requestPython<T>(command, { ...payload, userDataPath: collection.path }, timeout)
  collections().current(collection)
  if (collection.indexRevision !== revision) throw new Error('The library changed during this request. Please try again.')
  return result
}

function requestGroups(materials: CourseMaterial[], documents?: StudyDocumentRef[]) {
  const ids = [...new Set((documents ?? materials.filter(m => m.isActive !== false)).map(item =>
    splitMaterialId('materialId' in item ? item.materialId : item.id).collectionId))]
  return ids.map(id => ({ collection: collections().get(id),
    materials: materials.filter(m => splitMaterialId(m.id).collectionId === id).map(m => ({ ...m, id: splitMaterialId(m.id).localId })),
    ...(documents !== undefined ? { documents: localDocuments(collections().get(id), documents.filter(d => splitMaterialId(d.materialId).collectionId === id)) } : {}) }))
}

export async function searchLibraryWithPython(query: string, materials: CourseMaterial[], limit: number,
  embeddingModels?: LocalModel[], searchMode?: SearchMode, options?: LibrarySearchOptions): Promise<ChatSource[]> {
  writeLog('library_search_request', { query, materials: materials.map(m => ({ id: m.id, title: m.title })), documents: options?.documents, searchMode })
  const groups = requestGroups(materials, options?.documents)
  const resolvedEmbeddingModels = resolveEmbeddingModels(embeddingModels)
  const embeddingGpuEnabled = normalizeEmbeddingGpuEnabled(options?.embeddingGpuEnabled)
  if (groups.length && searchMode !== 'keyword') await prepareEmbeddingDevice(resolvedEmbeddingModels, embeddingGpuEnabled)
  const results = await Promise.all(groups.map(async ({ collection, ...scope }) => {
    const result = await storeRequest<SearchResult>(collection, 'search', { ...scope, query, limit,
      embeddingModels: resolvedEmbeddingModels, searchMode, embeddingGpuEnabled })
    return result.sources.map(source => qualifySource(collections(), collection, source))
  }))
  groups.forEach(group => collections().current(group.collection))
  const sources = mergeCollectionSources(results, limit)
  writeLog('library_search_result', { query, sourceCount: sources.length, sources: sources.map(logSource) })
  return sources
}

export async function starterSourcesWithPython(materials: CourseMaterial[], limit = 4, documents?: StudyDocumentRef[]): Promise<ChatSource[]> {
  const groups = requestGroups(materials, documents)
  const results = await Promise.all(groups.map(async ({ collection, ...scope }) => {
    const result = await storeRequest<SearchResult>(collection, 'starter_sources', { ...scope, limit })
    return result.sources.map(source => qualifySource(collections(), collection, source))
  }))
  groups.forEach(group => collections().current(group.collection))
  return mergeCollectionSources(results, limit)
}

export async function listIndexedMaterialsWithPython(): Promise<CourseMaterial[]> {
  const results = await Promise.all(collections().loaded().map(async collection => {
    const result = await storeRequest<ListMaterialsResult>(collection, 'list_materials', {})
    return result.materials.map(material => qualifyMaterial(collections(), collection, material))
  }))
  return results.flat()
}

export async function studyDocumentsWithPython(): Promise<StudyDocument[]> {
  const results = await Promise.all(collections().loaded().map(async collection => {
    const result = await storeRequest<{ documents: StudyDocument[] }>(collection, 'study_documents', {})
    return result.documents.map(document => qualifyDocument(collections(), collection, document))
  }))
  return results.flat()
}

export async function practiceSourcesWithPython(documents: StudyDocumentRef[], usedSourceKeys: string[], questionIndex: number): Promise<ChatSource[]> {
  const groups = requestGroups([], documents)
  if (!groups.length) return []
  for (let offset = 0; offset < groups.length; offset++) {
    const { collection, documents } = groups[(questionIndex + offset) % groups.length]
    const used = usedSourceKeys.filter(key => key.startsWith(`${collection.id}/`)).map(key => key.slice(collection.id.length + 1))
    const result = await storeRequest<SearchResult>(collection, 'practice_sources', { documents, usedSourceKeys: used, questionIndex })
    if (result.sources.length) return result.sources.map(source => qualifySource(collections(), collection, source))
  }
  return []
}

export async function setMaterialEnabledWithPython(materialId: string, isActive: boolean): Promise<void> {
  const { collection, localId } = collections().forMaterial(materialId)
  await storeRequest(collection, 'set_material_enabled', { materialId: localId, isActive })
}

export async function removeMaterialWithPython(materialId: string, materialPath?: string): Promise<void> {
  const { collection, localId } = collections().forMaterial(materialId)
  await collections().revise(collection)
  await storeRequest(collection, 'remove_material', { materialId: localId, path: materialPath })
}

export async function resolveSourceDocumentWithPython(source: ChatSource): Promise<ResolvedSourceDocument | null> {
  const { collection, localId } = collections().forMaterial(source.materialId ?? '')
  const result = await storeRequest<ResolveSourceDocumentResult>(collection, 'resolve_source_document', {
    source: { ...source, materialId: localId }
  })
  if (!result.source) return null
  return { ...result.source, path: collections().sourcePath(collection, result.source.path, localId),
    thumbnailPath: result.source.thumbnailPath ? collections().sourcePath(collection, result.source.thumbnailPath) : undefined }
}

export async function locateCollectionFolder(materialId: string, path: string): Promise<void> {
  const { collection, localId } = collections().forMaterial(materialId)
  const result = await storeRequest<ListMaterialsResult>(collection, 'list_materials', {})
  const material = result.materials.find(m => m.id === localId)
  if (!material?.path) throw new Error('This document is no longer in the library.')
  await collections().locateFolder(materialId, resolve(collection.path, material.kind === 'folder' ? material.path : dirname(material.path)), path)
}
