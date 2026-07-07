import type {
  AppStateSnapshot,
  ChatSource,
  CourseMaterial,
  LocalModel,
  LocalModelRole,
  MaterialIndexProgress,
  ModelDownloadProgress
} from './app-state'
import type {
  CleaningPreviewResult,
  EngineChatRequest,
  EngineChatResponse,
  EngineInfo,
  PdfSourceDocument,
  PdfSourceThumbnail,
  PickMaterialFolderResult,
  PickMaterialsResult,
  PickModelResult
} from './engine'
import type { CleaningProfileId, CleaningRuleId } from './cleaning'
import type { ModelCatalogItem } from './model-catalog'
import type { HuggingFaceSearchOptions } from './model-providers'
import type { OllamaDeleteResult, OllamaOpenResult, OllamaPullProgress, OllamaPullResult, OllamaStatus } from './ollama'

export interface TokenSmithBridge {
  platform: string
  getAppVersion: () => Promise<string>
  loadAppState: () => Promise<AppStateSnapshot | null>
  saveAppState: (state: AppStateSnapshot) => Promise<AppStateSnapshot>
  listEngines: () => Promise<EngineInfo[]>
  sendChatMessage: (request: EngineChatRequest) => Promise<EngineChatResponse>
  searchLibrary: (query: string, materials: CourseMaterial[], limit: number, embeddingModels?: LocalModel[]) => Promise<ChatSource[]>
  getPdfForSource: (source: ChatSource) => Promise<PdfSourceDocument>
  getPdfThumbnailForSource: (source: ChatSource) => Promise<PdfSourceThumbnail>
  pickMaterials: () => Promise<PickMaterialsResult>
  pickMaterialFolder: () => Promise<PickMaterialFolderResult>
  cancelMaterialIndexing: (materialId: string) => Promise<void>
  previewCleaning: (
    materialPath: string,
    options?: {
      cleaningProfileId?: CleaningProfileId
      cleaningRuleIds?: CleaningRuleId[]
    }
  ) => Promise<CleaningPreviewResult>
  indexMaterial: (
    materialId: string,
    materialPath: string,
    embeddingModel?: LocalModel,
    options?: {
      resume?: boolean
      title?: string
      cleaningProfileId?: CleaningProfileId
      cleaningRuleIds?: CleaningRuleId[]
    }
  ) => Promise<CourseMaterial>
  onMaterialIndexProgress: (callback: (progress: MaterialIndexProgress) => void) => () => void
  listMaterials: () => Promise<CourseMaterial[]>
  setMaterialEnabled: (materialId: string, isActive: boolean) => Promise<void>
  removeMaterial: (materialId: string, materialPath?: string) => Promise<void>
  getOllamaStatus: () => Promise<OllamaStatus>
  openOllamaDownloadPage: () => Promise<void>
  openOllamaApp: () => Promise<OllamaOpenResult>
  startOllamaService: () => Promise<OllamaOpenResult>
  pullOllamaModel: (modelName: string, baseUrl?: string) => Promise<OllamaPullResult>
  cancelOllamaPull: (modelName: string, baseUrl?: string) => Promise<void>
  deleteOllamaModel: (modelName: string, baseUrl?: string) => Promise<OllamaDeleteResult>
  onOllamaPullProgress: (callback: (progress: OllamaPullProgress) => void) => () => void
  pickModel: (role?: LocalModelRole) => Promise<PickModelResult>
  listRemoteProviderModels: (apiKey: string, baseUrl: string, role?: LocalModelRole) => Promise<string[]>
  searchHuggingFaceModels: (query: string, options: HuggingFaceSearchOptions) => Promise<ModelCatalogItem[]>
  downloadModel: (model: ModelCatalogItem, modelId: string) => Promise<LocalModel>
  cancelModelDownload: (filename: string) => Promise<void>
  removeModel: (model: LocalModel) => Promise<void>
  onModelDownloadProgress: (callback: (progress: ModelDownloadProgress) => void) => () => void
}
