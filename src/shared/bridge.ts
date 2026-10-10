import type { IndexMaterialOptions, PreparationReport } from './preparation'
import type { StudyDocument, StudyDocumentRef, LibrarySearchOptions } from './study-scope'
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
import type { DeviceCapabilities } from './device-capabilities'
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
  getModelContextMetadata: (model: LocalModel) => Promise<import('./model-context').ModelContextMetadata>
  platform: string
  getAppVersion: () => Promise<string>
  getDeviceCapabilities: () => Promise<DeviceCapabilities>
  getLogFile: () => Promise<TokenSmithLogFile>
  loadAppState: () => Promise<AppStateSnapshot | null>
  saveAppState: (state: AppStateSnapshot) => Promise<AppStateSnapshot>
  listEngines: () => Promise<EngineInfo[]>
  sendChatMessage: (request: EngineChatRequest, onAnswer?: (answer: EngineChatResponse, hasFollowUps: boolean) => void) => Promise<EngineChatResponse>
  cancelChatRequest: (requestId: string) => Promise<void>
  resolveChatQuestion: (request: EngineQuestionRewriteRequest) => Promise<QuestionRewrite>
  suggestChatQuestions: (requestId: string, request: EngineQuestionSuggestionRequest) => Promise<EngineQuestionSuggestionResponse>
  cancelChatQuestionSuggestions: (requestId: string) => Promise<void>
  starterSources: (materials: CourseMaterial[], limit?: number, documents?: StudyDocumentRef[]) => Promise<ChatSource[]>
  studyDocuments: () => Promise<StudyDocument[]>
  practiceSources: (documents: StudyDocumentRef[], usedSourceKeys: string[], questionIndex: number) => Promise<ChatSource[]>
  searchLibrary: (query: string, materials: CourseMaterial[], limit: number, embeddingModels?: LocalModel[], searchMode?: SearchMode, options?: LibrarySearchOptions) => Promise<ChatSource[]>
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
