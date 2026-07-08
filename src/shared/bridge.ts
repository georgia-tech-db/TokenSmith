import type {
  AppStateSnapshot,
  ChatSource,
  CourseMaterial,
  LocalModel,
  LocalModelRole,
  MaterialIndexProgress
} from './app-state'
import type {
  CleaningPreviewResult,
  EngineChatRequest,
  EngineChatResponse,
  EngineInfo,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse,
  PdfSourceDocument,
  PdfSourceThumbnail,
  PickMaterialFolderResult,
  PickMaterialsResult
} from './engine'
import type { CleaningProfileId, CleaningRuleId } from './cleaning'
import type {
  OllamaDeleteResult,
  OllamaOpenResult,
  OllamaPullProgress,
  OllamaPullResult,
  OllamaSearchResult,
  OllamaStatus
} from './ollama'

export interface TokenSmithBridge {
  platform: string
  getAppVersion: () => Promise<string>
  loadAppState: () => Promise<AppStateSnapshot | null>
  saveAppState: (state: AppStateSnapshot) => Promise<AppStateSnapshot>
  listEngines: () => Promise<EngineInfo[]>
  sendChatMessage: (request: EngineChatRequest) => Promise<EngineChatResponse>
  suggestChatQuestions: (request: EngineQuestionSuggestionRequest) => Promise<EngineQuestionSuggestionResponse>
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
  searchOllamaModels: (query: string, role?: LocalModelRole, limit?: number) => Promise<OllamaSearchResult[]>
  pullOllamaModel: (modelName: string, baseUrl?: string) => Promise<OllamaPullResult>
  cancelOllamaPull: (modelName: string, baseUrl?: string) => Promise<void>
  deleteOllamaModel: (modelName: string, baseUrl?: string) => Promise<OllamaDeleteResult>
  onOllamaPullProgress: (callback: (progress: OllamaPullProgress) => void) => () => void
  listRemoteProviderModels: (apiKey: string, baseUrl: string, role?: LocalModelRole) => Promise<string[]>
  removeModel: (model: LocalModel) => Promise<void>
}
