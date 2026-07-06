import { app, BrowserWindow, dialog, ipcMain, nativeImage, shell } from 'electron'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, extname, isAbsolute, join, parse, relative, resolve } from 'node:path'
import { listEngines, sendChatMessage } from './engine/engine-service'
import {
  cancelMaterialIndexingWithPython,
  indexMaterialWithPython,
  listIndexedMaterialsWithPython,
  previewCleaningWithPython,
  removeMaterialWithPython,
  resolveSourceDocumentWithPython,
  searchLibraryWithPython,
  setMaterialEnabledWithPython
} from './python/python-engine-service'
import {
  cancelModelDownload,
  downloadModelFromCatalog,
  removeDownloadedModel
} from './models/model-download-service'
import { searchHuggingFaceModels } from './models/huggingface-service'
import { listOpenAiCompatibleModels } from './engine/remote-chat-service'
import {
  rememberRemoteModelApiKeys,
  sanitizeAppStateSecrets
} from './engine/remote-model-secrets'
import type { AppStateSnapshot, ChatSource, CourseMaterial, LocalModel, LocalModelRole } from '../shared/app-state'
import type { CleaningProfileId, CleaningRuleId } from '../shared/cleaning'
import type {
  EngineChatRequest,
  PdfSourceDocument,
  PdfSourceThumbnail,
  PickMaterialFolderResult,
  PickMaterialsResult,
  PickModelResult
} from '../shared/engine'
import type { ModelCatalogItem } from '../shared/model-catalog'
import type { HuggingFaceSearchOptions } from '../shared/model-providers'

const stateFileName = 'tokensmith-state.json'
const appName = 'TokenSmith'
const appIconFileName = 'tokensmith-icon.png'

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
    const state = JSON.parse(stateJson) as AppStateSnapshot
    rememberRemoteModelApiKeys(state.models)

    const safeState = sanitizeAppStateSecrets(state)
    const safeStateJson = JSON.stringify(safeState, null, 2)
    if (safeStateJson !== stateJson) {
      await writeFile(getStatePath(), safeStateJson, 'utf8')
    }

    return safeState
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return null
    }

    throw error
  }
}

async function saveAppState(state: AppStateSnapshot): Promise<AppStateSnapshot> {
  const statePath = getStatePath()
  rememberRemoteModelApiKeys(state.models)
  const safeState = sanitizeAppStateSecrets(state)

  await mkdir(dirname(statePath), { recursive: true })
  await writeFile(statePath, JSON.stringify(safeState, null, 2), 'utf8')

  return safeState
}

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

async function pickMaterials(): Promise<PickMaterialsResult> {
  const result = await dialog.showOpenDialog({
    title: 'Add document collection',
    buttonLabel: 'Choose Folder',
    properties: ['openDirectory']
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

async function pickMaterialFolder(): Promise<PickMaterialFolderResult> {
  const result = await dialog.showOpenDialog({
    title: 'Choose document collection folder',
    buttonLabel: 'Choose Folder',
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

async function pickModel(role: LocalModelRole = 'generator'): Promise<PickModelResult> {
  const isEmbedder = role === 'embedder'
  const result = await dialog.showOpenDialog({
    title: isEmbedder ? 'Add local embedder model' : 'Add local chat model',
    buttonLabel: isEmbedder ? 'Add Embedder Model' : 'Add Chat Model',
    properties: ['openFile'],
    filters: [{ name: 'GGUF Models', extensions: ['gguf'] }]
  })

  if (result.canceled || result.filePaths.length === 0) {
    return {
      canceled: true
    }
  }

  const modelPath = result.filePaths[0]
  const modelStat = await stat(modelPath)
  const model: LocalModel = {
    id: createId('model'),
    name: parse(modelPath).name,
    engine: 'python',
    role,
    status: isEmbedder ? 'ready' : 'needsRuntime',
    source: 'local',
    path: modelPath,
    sizeBytes: modelStat.size,
    addedAt: new Date().toISOString()
  }

  return {
    canceled: false,
    model
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

function indexedMaterialAllowsPdf(material: CourseMaterial, pdfPath: string): boolean {
  const materialPath = normalizeSourcePath(material.path)
  if (!materialPath) {
    return false
  }

  if (material.kind === 'folder') {
    return isSameOrChildPath(materialPath, pdfPath)
  }

  return materialPath === pdfPath
}

interface IndexedPdfSourceResolution {
  path: string
  title: string
  page?: number
  thumbnailPath?: string
}

async function resolveIndexedPdfSourcePathFromMaterials(source: ChatSource): Promise<string> {
  const pdfPath = normalizeSourcePath(source.path)
  if (!pdfPath || extname(pdfPath).toLowerCase() !== '.pdf') {
    throw new Error('This source is not backed by a PDF file.')
  }

  const indexedMaterials = await listIndexedMaterialsWithPython()
  const isIndexed = indexedMaterials.some((material) => indexedMaterialAllowsPdf(material, pdfPath))
  if (!isIndexed) {
    throw new Error('This PDF is not part of the indexed library.')
  }

  const pdfStat = statSync(pdfPath)
  if (!pdfStat.isFile()) {
    throw new Error('The source PDF is no longer available.')
  }

  return pdfPath
}

function normalizedPageNumber(page: unknown): number | undefined {
  const numericPage = Number(page)
  return Number.isFinite(numericPage) && numericPage > 0 ? Math.round(numericPage) : undefined
}

async function resolveIndexedPdfSource(source: ChatSource): Promise<IndexedPdfSourceResolution> {
  const resolvedSource = await resolveSourceDocumentWithPython(source).catch(() => null)

  if (resolvedSource?.path) {
    const pdfPath = normalizeSourcePath(resolvedSource.path)
    if (!pdfPath || extname(pdfPath).toLowerCase() !== '.pdf') {
      throw new Error('This source is not backed by a PDF file.')
    }

    const pdfStat = statSync(pdfPath)
    if (!pdfStat.isFile()) {
      throw new Error('The source PDF is no longer available.')
    }

    return {
      path: pdfPath,
      title: resolvedSource.title || source.documentTitle || source.title || parse(pdfPath).name,
      page: normalizedPageNumber(resolvedSource.page) ?? sourcePageNumber(source),
      thumbnailPath: resolvedSource.thumbnailPath || source.thumbnailPath
    }
  }

  const pdfPath = await resolveIndexedPdfSourcePathFromMaterials(source)
  return {
    path: pdfPath,
    title: source.documentTitle || source.title || parse(pdfPath).name,
    page: sourcePageNumber(source),
    thumbnailPath: source.thumbnailPath
  }
}

function pdfThumbnailRootPath(): string {
  return join(app.getPath('userData'), 'tokensmith-pdf-thumbnails')
}

function sourcePageNumber(source: ChatSource): number | undefined {
  return normalizedPageNumber(source.pageStart)
}

function cachedThumbnailPathForPdfPage(pdfPath: string, page: number): string {
  const digest = createHash('sha256').update(pdfPath).digest('hex').slice(0, 20)
  return join(pdfThumbnailRootPath(), digest, `page-${String(page).padStart(4, '0')}.png`)
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
    page ? cachedThumbnailPathForPdfPage(pdfPath, page) : undefined
  ].filter((candidate): candidate is string => Boolean(candidate && extname(candidate).toLowerCase() === '.png'))

  for (const thumbnailPath of candidates) {
    if (!isSameOrChildPath(pdfThumbnailRootPath(), thumbnailPath)) {
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

app.whenReady().then(() => {
  applyDockIcon()

  ipcMain.handle('app:get-version', () => app.getVersion())
  ipcMain.handle('state:load', () => loadAppState())
  ipcMain.handle('state:save', (_event, state: AppStateSnapshot) => saveAppState(state))
  ipcMain.handle('engine:list', () => listEngines())
  ipcMain.handle('engine:chat', (_event, request: EngineChatRequest) => sendChatMessage(request))
  ipcMain.handle('library:pick-materials', () => pickMaterials())
  ipcMain.handle('library:pick-material-folder', () => pickMaterialFolder())
  ipcMain.handle('library:search', (_event, query: string, materials: CourseMaterial[], limit: number, embeddingModels?: LocalModel[]) =>
    searchLibraryWithPython(query, materials, limit, embeddingModels)
  )
  ipcMain.handle('library:get-pdf-for-source', (_event, source: ChatSource) => getPdfForSource(source))
  ipcMain.handle('library:get-pdf-thumbnail-for-source', (_event, source: ChatSource) =>
    getPdfThumbnailForSource(source)
  )
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
      options?: {
        resume?: boolean
        title?: string
        cleaningProfileId?: CleaningProfileId
        cleaningRuleIds?: CleaningRuleId[]
      }
    ) => indexMaterialWithPython(materialPath, embeddingModel, materialId, options)
  )
  ipcMain.handle('library:list-materials', () => listIndexedMaterialsWithPython())
  ipcMain.handle('library:set-material-enabled', (_event, materialId: string, isActive: boolean) =>
    setMaterialEnabledWithPython(materialId, isActive)
  )
  ipcMain.handle('library:remove-material', (_event, materialId: string, materialPath?: string) =>
    removeMaterialWithPython(materialId, materialPath)
  )
  ipcMain.handle('models:pick-model', (_event, role?: LocalModelRole) => pickModel(role))
  ipcMain.handle('models:list-remote-provider-models', (_event, apiKey: string, baseUrl: string, role?: LocalModelRole) =>
    listOpenAiCompatibleModels(apiKey, baseUrl, role)
  )
  ipcMain.handle('models:search-huggingface', (_event, query: string, options: HuggingFaceSearchOptions) =>
    searchHuggingFaceModels(query, options)
  )
  ipcMain.handle('models:download-model', (_event, model: ModelCatalogItem, modelId: string) =>
    downloadModelFromCatalog(model, modelId)
  )
  ipcMain.handle('models:cancel-download', (_event, filename: string) => cancelModelDownload(filename))
  ipcMain.handle('models:remove-model', (_event, model: LocalModel) => removeDownloadedModel(model))

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
