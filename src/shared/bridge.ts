import type { IndexMaterialOptions, PreparationReport } from './preparation'
import type {
  AppStateSnapshot,
  ChatSource,
  CourseMaterial,
  LocalModel,
  LocalModelRole,
  MaterialIndexProgress,
  SearchMode
} from './app-state'
import type {
  CleaningPreviewResult,
  EngineChatRequest,
  EngineChatResponse,
  EngineInfo,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse,
  EngineQuestionRewriteRequest,
  QuestionRewrite,
  MarkdownSourceDocument,
  PdfSourceDocument,
  PdfSourceThumbnail,
  PickMaterialFolderResult,
  PickMaterialsResult,
  TokenSmithLogFile
} from './engine'
import type { CleaningProfileId, CleaningRuleId } from './cleaning'
import type { CloudConnectionInput, CloudConnectionStatus, CloudGeneratorInput, CloudResult } from './cloud-generators'
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
  getLogFile: () => Promise<TokenSmithLogFile>
  loadAppState: () => Promise<AppStateSnapshot | null>
  saveAppState: (state: AppStateSnapshot) => Promise<AppStateSnapshot>
  listEngines: () => Promise<EngineInfo[]>
  sendChatMessage: (request: EngineChatRequest) => Promise<EngineChatResponse>
  resolveChatQuestion: (request: EngineQuestionRewriteRequest) => Promise<QuestionRewrite>
  suggestChatQuestions: (request: EngineQuestionSuggestionRequest) => Promise<EngineQuestionSuggestionResponse>
  starterSources: (materials: CourseMaterial[], limit?: number) => Promise<ChatSource[]>
  searchLibrary: (
    query: string,
    materials: CourseMaterial[],
    limit: number,
    embeddingModels?: LocalModel[],
    searchMode?: SearchMode,
    typoCorrectionEnabled?: boolean
  ) => Promise<ChatSource[]>
  getPdfForSource: (source: ChatSource) => Promise<PdfSourceDocument>
  getPdfThumbnailForSource: (source: ChatSource) => Promise<PdfSourceThumbnail>
  getMarkdownForSource: (source: ChatSource) => Promise<MarkdownSourceDocument>
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
    options?: IndexMaterialOptions
  ) => Promise<CourseMaterial>
  preparationReport: (path: string, documentPath?: string) => Promise<PreparationReport>
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
  getCloudConnections: () => Promise<CloudConnectionStatus>
  discoverCloudModels: (input: CloudConnectionInput) => Promise<CloudResult<string[]>>
  connectCloudGenerator: (input: CloudGeneratorInput) => Promise<CloudResult<LocalModel>>
  cancelCloudSetup: (requestId: string) => Promise<void>
  removeModel: (model: LocalModel) => Promise<void>
}
