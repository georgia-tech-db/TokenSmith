import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'
import type { TokenSmithBridge } from '../shared/bridge'

const tokenSmithBridge: TokenSmithBridge = {
  platform: process.platform,
  getAppVersion: () => ipcRenderer.invoke('app:get-version') as Promise<string>,
  loadAppState: () => ipcRenderer.invoke('state:load') as Promise<Awaited<ReturnType<TokenSmithBridge['loadAppState']>>>,
  saveAppState: (state) =>
    ipcRenderer.invoke('state:save', state) as Promise<Awaited<ReturnType<TokenSmithBridge['saveAppState']>>>,
  listEngines: () => ipcRenderer.invoke('engine:list') as Promise<Awaited<ReturnType<TokenSmithBridge['listEngines']>>>,
  sendChatMessage: (request) =>
    ipcRenderer.invoke('engine:chat', request) as Promise<
      Awaited<ReturnType<TokenSmithBridge['sendChatMessage']>>
    >,
  searchLibrary: (query, materials, limit, embeddingModels) =>
    ipcRenderer.invoke('library:search', query, materials, limit, embeddingModels) as Promise<
      Awaited<ReturnType<TokenSmithBridge['searchLibrary']>>
    >,
  getPdfForSource: (source) =>
    ipcRenderer.invoke('library:get-pdf-for-source', source) as Promise<
      Awaited<ReturnType<TokenSmithBridge['getPdfForSource']>>
    >,
  getPdfThumbnailForSource: (source) =>
    ipcRenderer.invoke('library:get-pdf-thumbnail-for-source', source) as Promise<
      Awaited<ReturnType<TokenSmithBridge['getPdfThumbnailForSource']>>
    >,
  pickMaterials: () =>
    ipcRenderer.invoke('library:pick-materials') as Promise<
      Awaited<ReturnType<TokenSmithBridge['pickMaterials']>>
    >,
  pickMaterialFolder: () =>
    ipcRenderer.invoke('library:pick-material-folder') as Promise<
      Awaited<ReturnType<TokenSmithBridge['pickMaterialFolder']>>
    >,
  cancelMaterialIndexing: (materialId) =>
    ipcRenderer.invoke('library:cancel-index-material', materialId) as Promise<
      Awaited<ReturnType<TokenSmithBridge['cancelMaterialIndexing']>>
    >,
  previewCleaning: (materialPath, options) =>
    ipcRenderer.invoke('library:preview-cleaning', materialPath, options) as Promise<
      Awaited<ReturnType<TokenSmithBridge['previewCleaning']>>
    >,
  indexMaterial: (materialId, materialPath, embeddingModel, options) =>
    ipcRenderer.invoke('library:index-material', materialId, materialPath, embeddingModel, options) as Promise<
      Awaited<ReturnType<TokenSmithBridge['indexMaterial']>>
    >,
  onMaterialIndexProgress: (callback) => {
    const listener = (_event: IpcRendererEvent, progress: Parameters<typeof callback>[0]) => {
      callback(progress)
    }

    ipcRenderer.on('library:index-progress', listener)
    return () => {
      ipcRenderer.off('library:index-progress', listener)
    }
  },
  listMaterials: () =>
    ipcRenderer.invoke('library:list-materials') as Promise<
      Awaited<ReturnType<TokenSmithBridge['listMaterials']>>
    >,
  setMaterialEnabled: (materialId, isActive) =>
    ipcRenderer.invoke('library:set-material-enabled', materialId, isActive) as Promise<
      Awaited<ReturnType<TokenSmithBridge['setMaterialEnabled']>>
    >,
  removeMaterial: (materialId, materialPath) =>
    ipcRenderer.invoke('library:remove-material', materialId, materialPath) as Promise<
      Awaited<ReturnType<TokenSmithBridge['removeMaterial']>>
    >,
  getOllamaStatus: () =>
    ipcRenderer.invoke('ollama:status') as Promise<Awaited<ReturnType<TokenSmithBridge['getOllamaStatus']>>>,
  openOllamaDownloadPage: () =>
    ipcRenderer.invoke('ollama:open-download-page') as Promise<
      Awaited<ReturnType<TokenSmithBridge['openOllamaDownloadPage']>>
    >,
  openOllamaApp: () =>
    ipcRenderer.invoke('ollama:open-app') as Promise<Awaited<ReturnType<TokenSmithBridge['openOllamaApp']>>>,
  startOllamaService: () =>
    ipcRenderer.invoke('ollama:start-service') as Promise<
      Awaited<ReturnType<TokenSmithBridge['startOllamaService']>>
    >,
  pullOllamaModel: (modelName) =>
    ipcRenderer.invoke('ollama:pull-model', modelName) as Promise<
      Awaited<ReturnType<TokenSmithBridge['pullOllamaModel']>>
    >,
  onOllamaPullProgress: (callback) => {
    const listener = (_event: IpcRendererEvent, progress: Parameters<typeof callback>[0]) => {
      callback(progress)
    }

    ipcRenderer.on('ollama:pull-progress', listener)
    return () => {
      ipcRenderer.off('ollama:pull-progress', listener)
    }
  },
  pickModel: (role) =>
    ipcRenderer.invoke('models:pick-model', role) as Promise<Awaited<ReturnType<TokenSmithBridge['pickModel']>>>,
  listRemoteProviderModels: (apiKey, baseUrl, role) =>
    ipcRenderer.invoke('models:list-remote-provider-models', apiKey, baseUrl, role) as Promise<
      Awaited<ReturnType<TokenSmithBridge['listRemoteProviderModels']>>
    >,
  searchHuggingFaceModels: (query, options) =>
    ipcRenderer.invoke('models:search-huggingface', query, options) as Promise<
      Awaited<ReturnType<TokenSmithBridge['searchHuggingFaceModels']>>
    >,
  downloadModel: (model, modelId) =>
    ipcRenderer.invoke('models:download-model', model, modelId) as Promise<
      Awaited<ReturnType<TokenSmithBridge['downloadModel']>>
    >,
  cancelModelDownload: (filename) =>
    ipcRenderer.invoke('models:cancel-download', filename) as Promise<
      Awaited<ReturnType<TokenSmithBridge['cancelModelDownload']>>
    >,
  removeModel: (model) =>
    ipcRenderer.invoke('models:remove-model', model) as Promise<Awaited<ReturnType<TokenSmithBridge['removeModel']>>>,
  onModelDownloadProgress: (callback) => {
    const listener = (_event: IpcRendererEvent, progress: Parameters<typeof callback>[0]) => {
      callback(progress)
    }

    ipcRenderer.on('models:download-progress', listener)
    return () => {
      ipcRenderer.off('models:download-progress', listener)
    }
  }
}

contextBridge.exposeInMainWorld('tokensmith', tokenSmithBridge)
