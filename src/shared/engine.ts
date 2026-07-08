import type {
  ApplicationSettings,
  ChatMessage,
  ChatSource,
  CourseMaterial,
  LocalModel,
  ModelRuntimeSettings,
  TokenSmithSettings
} from './app-state'
import type { CleaningProfileId, CleaningRuleId } from './cleaning'

export interface EngineInfo {
  id: 'tokensmith'
  name: string
  status: 'ready' | 'needsSetup' | 'unavailable'
  detail: string
}

export interface EngineChatRequest {
  prompt: string
  messages: ChatMessage[]
  materials: CourseMaterial[]
  model: LocalModel
  settings: TokenSmithSettings
  applicationSettings?: ApplicationSettings
  modelSettings?: ModelRuntimeSettings
  retrievedSources?: ChatSource[]
}

export interface EngineChatResponse {
  engineId: EngineInfo['id']
  modelName: string
  text: string
  sources: ChatSource[]
  followUpSuggestions?: string[]
  followUpError?: string
}

export interface EngineQuestionSuggestionRequest {
  messages: ChatMessage[]
  materials: CourseMaterial[]
  model: LocalModel
  settings: TokenSmithSettings
  applicationSettings?: ApplicationSettings
  modelSettings?: ModelRuntimeSettings
  retrievedSources?: ChatSource[]
}

export interface EngineQuestionSuggestionResponse {
  suggestions: string[]
}

export interface PdfSourceDocument {
  title: string
  dataUrl: string
  path: string
  page?: number
}

export interface PdfSourceThumbnail {
  title: string
  dataUrl: string
  path: string
  page?: number
}

export interface CleaningPreviewPage {
  page?: number
  text: string
}

export interface CleaningPreviewChunk {
  text: string
  wordCount: number
  pageStart?: number
  pageEnd?: number
  chunkSize?: number
  sectionHeader?: string
}

export interface CleaningPreviewRule {
  id: CleaningRuleId
  name: string
  description: string
  enabled: boolean
  locked?: boolean
}

export interface CleaningPreviewResult {
  profile: {
    id: CleaningProfileId
    name: string
    description: string
    version: number
  }
  document: {
    title: string
    path: string
    kind: CourseMaterial['kind']
    pageCount?: number
  }
  rawPages: CleaningPreviewPage[]
  cleanedPages: CleaningPreviewPage[]
  chunks: CleaningPreviewChunk[]
  rules: CleaningPreviewRule[]
  cleaningRuleIds: CleaningRuleId[]
}

export interface PickMaterialsResult {
  canceled: boolean
  materials: CourseMaterial[]
}

export interface PickMaterialFolderResult {
  canceled: boolean
  path?: string
  title?: string
}
