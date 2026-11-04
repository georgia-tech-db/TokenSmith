import { createStateWriter } from './atomic-state'
import { materialCollectionId } from '../shared/collections'
import type { IndexMaterialOptions } from '../shared/preparation'
import { preparationReportWithPython } from './python/python-engine-service'
import { app, BrowserWindow, dialog, ipcMain, nativeImage, net, safeStorage, shell } from 'electron'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, extname, isAbsolute, join, parse, relative, resolve } from 'node:path'
import { listEngines, resolveChatQuestion, scoreChatAnswer, sendChatMessage, suggestChatQuestions } from './engine/engine-service'
import {
  collections,
  cancelCollectionIndexing,
  locateCollectionFolder,
  cancelMaterialIndexingWithPython,
  indexMaterialWithPython,
  listIndexedMaterialsWithPython,
  previewCleaningWithPython,
  studyDocumentsWithPython,
  practiceSourcesWithPython,
  removeMaterialWithPython,
  readTokenSmithLogFile,
  resolveSourceDocumentWithPython,
  searchLibraryWithPython,
  setMaterialEnabledWithPython,
  starterSourcesWithPython
} from './python/python-engine-service'
import { getModelContextMetadata } from './engine/model-context-metadata'
import { CloudGeneratorService, cloudResult } from './engine/cloud-generator-service'
import { remoteGeneratorFetch, setRemoteGeneratorTransport } from './engine/remote-generator-network'
import type { CloudConnectionInput, CloudGeneratorInput } from '../shared/cloud-generators'
import {
  cancelOllamaPullModel,
  deleteOllamaModel,
  getOllamaStatus,
  openOllamaApp,
  openOllamaDownloadPage,
  pullOllamaModel,
  startOllamaService
} from './engine/ollama-service'
import { searchOllamaLibrary } from './engine/ollama-library-search'
import { detectDeviceCapabilities } from './system/detect-device-capabilities'
import {
  rememberRemoteModelApiKeys,
  setCloudCredentialResolver,
  modelWithRememberedRemoteApiKey,
  sanitizeAppStateSecrets
} from './engine/remote-model-secrets'
import type { AppStateSnapshot, ChatSource, CourseMaterial, LocalModel, LocalModelRole, SearchMode } from '../shared/app-state'
import type { StudyDocumentRef, LibrarySearchOptions } from '../shared/study-scope'
import { removeRetiredModelState } from '../shared/retired-model-state'
import type { CleaningProfileId, CleaningRuleId } from '../shared/cleaning'
import type {
  EngineChatRequest,
  EngineChatResponse,
  EngineQuestionSuggestionRequest,
  EngineQuestionRewriteRequest,
  MarkdownSourceDocument,
  PdfSourceDocument,
  PdfSourceThumbnail,
  PickMaterialFolderResult,
  PickMaterialsResult
} from '../shared/engine'

const stateFileName = 'tokensmith-state.json'
const appName = 'TokenSmith'
const appIconFileName = 'tokensmith-icon.png'
let cloudGenerators: CloudGeneratorService
const chatRequests = new Map<string, AbortController>()
const questionSuggestionRequests = new Map<string, AbortController>()
const collectionRequests = new Set<{ ids: Set<string>; controller: AbortController }>()
function trackCollectionRequest(request: EngineChatRequest | EngineQuestionSuggestionRequest, controller: AbortController) {
  const ids = new Set([...request.materials.map(m => materialCollectionId(m.id)),
    ...(request.retrievedSources ?? []).map(s => materialCollectionId(s.materialId))].filter((id): id is string => Boolean(id)))
  const mounted = [...ids].map(id => { const item = collections().get(id); return { item, revision: item.indexRevision } })
  for (const source of request.retrievedSources ?? []) {
    const id = materialCollectionId(source.materialId)
    if (id && source.collectionRevision && collections().get(id).indexRevision !== source.collectionRevision) {
      throw new Error('This collection has changed. Start a new question or practice topic to use its current passages.')
    }
  }
  const tracked = { ids, controller }; collectionRequests.add(tracked)
  return { finish: () => {
    if (controller.signal.aborted) throw new DOMException('Request cancelled', 'AbortError')
    for (const { item, revision } of mounted) {
      collections().current(item)
      if (item.indexRevision !== revision) throw new Error('The collection changed during this request. Please try again.')
    }
  }, close: () => collectionRequests.delete(tracked) }
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function withCloudStatus(state: AppStateSnapshot): AppStateSnapshot {
  return { ...state, models: state.models.map(model => {
    if (model.connectionId) return cloudGenerators.describeModel(model)
    if (model.engine === 'remote' && (model.role === 'generator' || !model.role)) {
      return { ...model, cloudCredentialStatus: modelWithRememberedRemoteApiKey(model).apiKey ? 'connected' : 'reconnect' }
    }
    return model
  }) }
}

function getAppIconPath(): string {
  const candidates = app.isPackaged
    ? [join(process.resourcesPath, appIconFileName)]
    : [join(app.getAppPath(), 'build-resources', appIconFileName), join(__dirname, '../../build-resources', appIconFileName)]

  return candidates.find((candidate) => existsSync(candidate)) ?? candidates[0]
}

function applyDockIcon(): void {
  if (process.platform !== 'darwin') {
    return
  }

  const icon = nativeImage.createFromPath(getAppIconPath())

  if (!icon.isEmpty()) {
    app.dock?.setIcon(icon)
  }
}

function getStatePath(): string {
  return join(app.getPath('userData'), stateFileName)
}

async function loadAppState(): Promise<AppStateSnapshot | null> {
  try {
    const stateJson = await readFile(getStatePath(), 'utf8')
    let state = JSON.parse(stateJson) as AppStateSnapshot
    if (state.collectionFormatVersion !== 1) state = { ...state, collectionFormatVersion: 1,
      materials: [], conversations: [], activeConversationId: '', practice: undefined, studyScope: [] }
    rememberRemoteModelApiKeys(state.models)

    const safeState = sanitizeAppStateSecrets(removeRetiredModelState({ ...state, collectionFormatVersion: 1 }))
    const safeStateJson = JSON.stringify(safeState, null, 2)
    if (safeStateJson !== stateJson) {
      await writeFile(getStatePath(), safeStateJson, 'utf8')
    }

    return withCloudStatus(safeState)
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return null
    }

    throw error
  }
}

const writeStateSnapshot = createStateWriter()

async function saveAppState(state: AppStateSnapshot): Promise<AppStateSnapshot> {
  const statePath = getStatePath()
  rememberRemoteModelApiKeys(state.models)
  const safeState = sanitizeAppStateSecrets(removeRetiredModelState({ ...state, collectionFormatVersion: 1 }))

  await writeStateSnapshot(statePath, JSON.stringify(safeState, null, 2))

  return safeState
}

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

async function pickMaterials(): Promise<PickMaterialsResult> {
  const result = await dialog.showOpenDialog({
    title: 'Add documents',
    buttonLabel: 'Add documents',
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'Documents', extensions: ['pdf', 'md', 'markdown', 'txt'] }]
  })

  if (result.canceled) {
    return {
      canceled: true,
      materials: []
    }
  }

  const materials = await Promise.all(result.filePaths.map((filePath) => createIndexingMaterial(filePath)))

  return {
    canceled: false,
    materials
  }
}

async function pickMaterialFolder(library = false): Promise<PickMaterialFolderResult> {
  const result = await dialog.showOpenDialog({
    title: library ? 'Open library' : 'Locate original documents',
    buttonLabel: library ? 'Open library' : 'Choose',
    properties: ['openDirectory']
  })

  if (result.canceled || result.filePaths.length === 0) {
    return {
      canceled: true
    }
  }

  const folderPath = result.filePaths[0]

  return {
    canceled: false,
    path: folderPath,
    title: parse(folderPath).name
  }
}

async function createIndexingMaterial(materialPath: string): Promise<CourseMaterial> {
  const materialStat = await stat(materialPath)
  const title = parse(materialPath).name
  const materialId = createId('material')

  return {
    id: materialId,
    title,
    detail: materialStat.isDirectory() ? 'Parsing folder' : 'Parsing file',
    status: 'indexing',
    kind: materialStat.isDirectory() ? 'folder' : 'document',
    path: materialPath,
    addedAt: new Date().toISOString(),
    fileCount: materialStat.isDirectory() ? 0 : 1,
    wordCount: 0,
    chunkCount: 0,
    isActive: false,
    indexing: {
      materialId,
      phase: 'parsing',
      percent: 1,
      processedFiles: 0,
      totalFiles: materialStat.isDirectory() ? 0 : 1,
      processedEmbeddings: 0,
      totalEmbeddings: 0,
      message: 'Parsing'
    }
  }
}

function normalizeSourcePath(path?: string): string | null {
  if (!path) {
    return null
  }

  try {
    if (path.startsWith('file://')) {
      return resolve(decodeURIComponent(new URL(path).pathname))
    }
  } catch {
    return null
  }

  return resolve(path)
}

function isSameOrChildPath(parentPath: string, childPath: string): boolean {
  const relativePath = relative(parentPath, childPath)
  return relativePath === '' || (!!relativePath && !relativePath.startsWith('..') && !isAbsolute(relativePath))
}

function isMarkdownSourcePath(path: string): boolean {
  const extension = extname(path).toLowerCase()
  return extension === '.md' || extension === '.markdown' || extension === '.txt'
}

interface IndexedPdfSourceResolution {
  path: string
  title: string
  page?: number
  thumbnailPath?: string
}

function normalizedPageNumber(page: unknown): number | undefined {
  const numericPage = Number(page)
  return Number.isFinite(numericPage) && numericPage > 0 ? Math.round(numericPage) : undefined
}

async function resolveIndexedPdfSource(source: ChatSource): Promise<IndexedPdfSourceResolution> {
  const { collection } = collections().forMaterial(source.materialId ?? '')
  const resolvedSource = await resolveSourceDocumentWithPython(source)
  collections().current(collection)

  if (resolvedSource?.path) {
    const pdfPath = normalizeSourcePath(resolvedSource.path)
    if (!pdfPath || extname(pdfPath).toLowerCase() !== '.pdf') {
      throw new Error('This source is not backed by a PDF file.')
    }

    const pdfStat = existsSync(pdfPath) ? statSync(pdfPath) : undefined
    if (!pdfStat?.isFile()) {
      throw new Error('The original PDF is unavailable. Use Locate files in Library to find it.')
    }

    return {
      path: pdfPath,
      title: resolvedSource.title || source.documentTitle || source.title || parse(pdfPath).name,
      page: sourcePageNumber(source) ?? normalizedPageNumber(resolvedSource.page),
      thumbnailPath: resolvedSource.thumbnailPath || source.thumbnailPath
    }
  }

  throw new Error('This passage is no longer available in the loaded collection.')
}

function pdfThumbnailRootPath(source: ChatSource): string {
  return join(collections().forMaterial(source.materialId ?? '').collection.path, 'tokensmith-pdf-thumbnails')
}

function sourcePageNumber(source: ChatSource): number | undefined {
  return normalizedPageNumber(source.pageStart)
}

function cachedThumbnailPathForPdfPage(source: ChatSource, pdfPath: string, page: number): string {
  const digest = createHash('sha256').update(pdfPath).digest('hex').slice(0, 20)
  return join(pdfThumbnailRootPath(source), digest, `page-${String(page).padStart(4, '0')}.png`)
}

function resolveCachedThumbnailPath(
  source: ChatSource,
  pdfPath: string,
  page: number | undefined,
  resolvedThumbnailPath?: string
): string {
  const candidates = [
    normalizeSourcePath(resolvedThumbnailPath),
    normalizeSourcePath(source.thumbnailPath),
    page ? cachedThumbnailPathForPdfPage(source, pdfPath, page) : undefined
  ].filter((candidate): candidate is string => Boolean(candidate && extname(candidate).toLowerCase() === '.png'))

  for (const thumbnailPath of candidates) {
    if (!isSameOrChildPath(pdfThumbnailRootPath(source), thumbnailPath)) {
      continue
    }

    if (!existsSync(thumbnailPath)) {
      continue
    }

    const thumbnailStat = statSync(thumbnailPath)
    if (thumbnailStat.isFile()) {
      return thumbnailPath
    }
  }

  throw new Error('This source does not have a cached thumbnail yet.')
}

async function getPdfForSource(source: ChatSource): Promise<PdfSourceDocument> {
  const resolvedSource = await resolveIndexedPdfSource(source)
  const dataUrl = `data:application/pdf;base64,${readFileSync(resolvedSource.path).toString('base64')}`

  return {
    title: resolvedSource.title,
    dataUrl,
    path: resolvedSource.path,
    page: resolvedSource.page
  }
}

async function getPdfThumbnailForSource(source: ChatSource): Promise<PdfSourceThumbnail> {
  collections().forMaterial(source.materialId ?? '')
  const resolvedSource = await resolveIndexedPdfSource(source)
  const thumbnailPath = resolveCachedThumbnailPath(
    source,
    resolvedSource.path,
    resolvedSource.page,
    resolvedSource.thumbnailPath
  )
  const dataUrl = `data:image/png;base64,${readFileSync(thumbnailPath).toString('base64')}`

  return {
    title: resolvedSource.title,
    dataUrl,
    path: thumbnailPath,
    page: resolvedSource.page
  }
}

async function getMarkdownForSource(source: ChatSource): Promise<MarkdownSourceDocument> {
  const { collection } = collections().forMaterial(source.materialId ?? '')
  const resolvedSource = await resolveSourceDocumentWithPython(source)
  collections().current(collection)
  const markdownPath = normalizeSourcePath(resolvedSource?.path)
  if (!markdownPath || !isMarkdownSourcePath(markdownPath)) {
    throw new Error('This source is not backed by a Markdown file.')
  }

  const markdownStat = existsSync(markdownPath) ? statSync(markdownPath) : undefined
  if (!markdownStat?.isFile()) {
    throw new Error('The original document is unavailable. Use Locate files in Library to find it.')
  }

  return {
    title: resolvedSource?.title || source.documentTitle || source.title || parse(markdownPath).name,
    path: markdownPath,
    text: readFileSync(markdownPath, 'utf8'),
    chunkText: source.context || source.excerpt,
    locator: source.locator,
    sectionHeader: source.sectionHeader,
    lineFrom: source.lineFrom ?? resolvedSource?.lineFrom,
    lineTo: source.lineTo ?? resolvedSource?.lineTo
  }
}

function createMainWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 780,
    minWidth: 960,
    minHeight: 640,
    title: 'TokenSmith',
    backgroundColor: '#f7f4ef',
    icon: getAppIconPath(),
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })

  mainWindow.maximize()

  if (process.env.ELECTRON_RENDERER_URL) {
    void mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    void mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.setName(appName)

app.whenReady().then(async () => {
  setRemoteGeneratorTransport((input, init) => net.fetch(input instanceof URL ? input.toString() : input, { ...init, credentials: 'omit' }))
  cloudGenerators = new CloudGeneratorService(join(app.getPath('userData'), 'cloud-connections.json'), safeStorage, remoteGeneratorFetch)
  await cloudGenerators.initialize()
  await collections().initialize()
  setCloudCredentialResolver(model => cloudGenerators.credentialFor(model))
  applyDockIcon()

  ipcMain.handle('app:get-version', () => app.getVersion())
  ipcMain.handle('app:get-log-file', () => readTokenSmithLogFile())
  ipcMain.handle('device:capabilities', () =>
    detectDeviceCapabilities(
      process.env.OLLAMA_MODELS?.trim() || app.getPath('home'),
      () => app.getGPUInfo('complete')
    )
  )
  ipcMain.handle('state:load', () => loadAppState())
  ipcMain.handle('state:save', (_event, state: AppStateSnapshot) => saveAppState(state))
  ipcMain.handle('cloud:connections', () => cloudGenerators.status())
  ipcMain.handle('cloud:discover', (_event, input: CloudConnectionInput) => cloudResult(() => cloudGenerators.discover(input)))
  ipcMain.handle('cloud:connect-generator', (_event, input: CloudGeneratorInput) => cloudResult(() => cloudGenerators.connect(input)))
  ipcMain.handle('cloud:forget', (_event, connectionId: string) => cloudResult(() => cloudGenerators.forget(connectionId)))
  ipcMain.handle('cloud:cancel', (_event, requestId: string) => cloudGenerators.cancel(requestId))
  ipcMain.handle('engine:list', () => listEngines())
  ipcMain.handle('engine:chat', async (event, request: EngineChatRequest) => {
    const key = `${event.sender.id}:${request.requestId}`
    const controller = new AbortController()
    const tracked = trackCollectionRequest(request, controller)
    if (request.requestId) chatRequests.set(key, controller)
    try {
      const result = await sendChatMessage(request, {
        signal: controller.signal,
        onAnswer: (answer, hasFollowUps) => {
          tracked.finish()
          if (request.requestId && !controller.signal.aborted && !event.sender.isDestroyed()) {
            event.sender.send('engine:answer-ready', request.requestId, answer, hasFollowUps)
          }
        }
      })
      tracked.finish()
      return result
    } finally {
      tracked.close()
      if (chatRequests.get(key) === controller) chatRequests.delete(key)
    }
  })
  ipcMain.handle('engine:score-answer', (_event, request: EngineChatRequest, response: EngineChatResponse) =>
    scoreChatAnswer(request, response)
  )
  ipcMain.handle('engine:resolve-question', async (event, request: EngineQuestionRewriteRequest) => {
    const key = `${event.sender.id}:${request.requestId}`
    const controller = new AbortController()
    if (request.requestId) chatRequests.set(key, controller)
    try { return await resolveChatQuestion(request, controller.signal) }
    finally { if (chatRequests.get(key) === controller) chatRequests.delete(key) }
  })
  ipcMain.handle('engine:cancel-chat', (event, requestId: string) => {
    chatRequests.get(`${event.sender.id}:${requestId}`)?.abort()
  })
  ipcMain.handle('engine:suggest-questions', async (_event, requestId: string, request: EngineQuestionSuggestionRequest) => {
    const controller = new AbortController()
    const tracked = trackCollectionRequest(request, controller)
    questionSuggestionRequests.set(requestId, controller)
    try {
      const result = await suggestChatQuestions(request, controller.signal)
      tracked.finish()
      return result
    } catch (error) {
      if (isAbortError(error)) return { suggestions: [] }
      throw error
    } finally {
      tracked.close()
      questionSuggestionRequests.delete(requestId)
    }
  })
  ipcMain.handle('engine:cancel-suggest-questions', (_event, requestId: string) => {
    questionSuggestionRequests.get(requestId)?.abort()
  })
  ipcMain.handle('collections:list', () => collections().list())
  ipcMain.handle('collections:create', (_event, path: string, name: string) => collections().create(path, name))
  ipcMain.handle('libraries:create', (_event, name: string) => collections().createManaged(name))
  ipcMain.handle('libraries:pick', () => pickMaterialFolder(true))
  ipcMain.handle('collections:load', (_event, path: string) => collections().load(path))
  ipcMain.handle('collections:unload', (_event, id: string) => collections().unload(id, async () => {
    for (const request of collectionRequests) if (request.ids.has(id)) request.controller.abort()
    await cancelCollectionIndexing(id)
  }))
  ipcMain.handle('collections:locate-folder', (_event, materialId: string, path: string) => locateCollectionFolder(materialId, path))
  ipcMain.handle('collections:topics', (_event, documents: StudyDocumentRef[]) => collections().topics(documents))
  ipcMain.handle('collections:save-topics', (_event, documents: StudyDocumentRef[], topics: import('../shared/practice').PracticeTopic[], revision: string) => collections().saveTopics(documents, topics, revision))
  ipcMain.handle('library:pick-materials', () => pickMaterials())
  ipcMain.handle('library:pick-material-folder', () => pickMaterialFolder())
  ipcMain.handle('library:starter-sources', (_event, materials: CourseMaterial[], limit?: number, documents?: StudyDocumentRef[]) =>
    starterSourcesWithPython(materials, limit, documents)
  )
  ipcMain.handle('library:study-documents', () => studyDocumentsWithPython())
  ipcMain.handle('library:practice-sources', (_event, documents: StudyDocumentRef[], usedSourceKeys: string[], questionIndex: number) =>
    practiceSourcesWithPython(documents, usedSourceKeys, questionIndex)
  )
  ipcMain.handle('library:search', (_event, query: string, materials: CourseMaterial[], limit: number, embeddingModels?: LocalModel[], searchMode?: SearchMode, options?: LibrarySearchOptions) =>
    searchLibraryWithPython(query, materials, limit, embeddingModels, searchMode, options)
  )
  ipcMain.handle('library:get-pdf-for-source', (_event, source: ChatSource) => getPdfForSource(source))
  ipcMain.handle('library:get-pdf-thumbnail-for-source', (_event, source: ChatSource) =>
    getPdfThumbnailForSource(source)
  )
  ipcMain.handle('library:get-markdown-for-source', (_event, source: ChatSource) => getMarkdownForSource(source))
  ipcMain.handle('library:cancel-index-material', (_event, materialId: string) =>
    cancelMaterialIndexingWithPython(materialId)
  )
  ipcMain.handle(
    'library:preview-cleaning',
    (
      _event,
      materialPath: string,
      options?: {
        cleaningProfileId?: CleaningProfileId
        cleaningRuleIds?: CleaningRuleId[]
      }
    ) => previewCleaningWithPython(materialPath, options)
  )
  ipcMain.handle(
    'library:index-material',
    (
      _event,
      materialId: string,
      materialPath: string,
      embeddingModel?: LocalModel,
      options?: IndexMaterialOptions
    ) => indexMaterialWithPython(materialPath, embeddingModel, materialId, options)
  )
  ipcMain.handle('library:preparation-report', (_event, path: string, documentPath?: string, materialId?: string) => preparationReportWithPython(path, documentPath, materialId))
  ipcMain.handle('library:list-materials', () => listIndexedMaterialsWithPython())
  ipcMain.handle('library:set-material-enabled', (_event, materialId: string, isActive: boolean) =>
    setMaterialEnabledWithPython(materialId, isActive)
  )
  ipcMain.handle('library:remove-material', (_event, materialId: string, materialPath?: string) =>
    removeMaterialWithPython(materialId, materialPath)
  )
  ipcMain.handle('ollama:status', () => getOllamaStatus())
  ipcMain.handle('models:context-metadata', (_event, model: LocalModel) =>
    getModelContextMetadata(model.engine === 'remote' ? modelWithRememberedRemoteApiKey(model) : model)
  )
  ipcMain.handle('ollama:open-download-page', () => openOllamaDownloadPage())
  ipcMain.handle('ollama:open-app', () => openOllamaApp())
  ipcMain.handle('ollama:start-service', () => startOllamaService())
  ipcMain.handle('ollama:search-models', (_event, query: string, role?: LocalModelRole, limit?: number) =>
    searchOllamaLibrary(query, role, limit)
  )
  ipcMain.handle('ollama:pull-model', (_event, modelName: string, baseUrl?: string) => pullOllamaModel(modelName, baseUrl))
  ipcMain.handle('ollama:cancel-pull', (_event, modelName: string, baseUrl?: string) => cancelOllamaPullModel(modelName, baseUrl))
  ipcMain.handle('ollama:delete-model', (_event, modelName: string, baseUrl?: string) => deleteOllamaModel(modelName, baseUrl))
  ipcMain.handle('models:remove-model', (_event, model: LocalModel) => {
    if (model.engine === 'ollama' || model.source === 'ollama') {
      return deleteOllamaModel(model.ollamaModelName ?? model.name, model.ollamaBaseUrl)
    }
  })

  createMainWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
