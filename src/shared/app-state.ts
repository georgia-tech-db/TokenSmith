import type { PreparationSettings } from './preparation'
import type { EmbeddingOptions } from './embedding-settings'
import type { CleaningProfileId, CleaningRuleId } from './cleaning'

export type ScreenId = 'chat' | 'library' | 'models' | 'settings'

export type MaterialStatus = 'ready' | 'indexing' | 'paused' | 'needsReview'
export type LocalModelRole = 'generator' | 'embedder' | 'both'

export interface MaterialIndexProgress {
  materialId: string
  phase: 'parsing' | 'chunking' | 'embedding' | 'saving' | 'complete' | 'error'
  percent: number
  processedFiles?: number
  totalFiles?: number
  processedEmbeddings?: number
  totalEmbeddings?: number
  message?: string
}

export interface ModelDownloadProgress {
  modelId: string
  filename: string
  status: 'downloading' | 'complete' | 'incomplete' | 'error' | 'removed'
  percent: number
  bytesReceived: number
  bytesTotal?: number
  speedBytesPerSecond?: number
  path?: string
  message?: string
  error?: string
}

export interface CourseMaterial {
  id: string
  title: string
  detail: string
  status: MaterialStatus
  kind: 'pdf' | 'folder' | 'document'
  path?: string
  addedAt: string
  fileCount?: number
  sizeBytes?: number
  wordCount?: number
  pageCount?: number
  chunkCount?: number
  chunkSize?: number
  indexedAt?: string
  isActive?: boolean
  embeddingModel?: string
  embeddingModelId?: string
  embeddingModelName?: string
  cleaningProfileId?: CleaningProfileId
  cleaningProfileName?: string
  cleaningProfileVersion?: number
  cleaningRuleIds?: CleaningRuleId[]
  preparation?: PreparationSettings
  preparationModelName?: string
  preparationIssueCount?: number
  error?: string
  indexing?: MaterialIndexProgress
}

export interface LocalModel {
  id: string
  name: string
  engine: 'ollama' | 'remote'
  role?: LocalModelRole
  status: 'ready' | 'needsRuntime' | 'missing' | 'downloading' | 'incomplete' | 'downloadError'
  source?: 'bundled' | 'local' | 'downloaded' | 'ollama' | 'remote'
  filename?: string
  path?: string
  ollamaModelName?: string
  ollamaBaseUrl?: string
  providerId?: 'groq' | 'openai' | 'gemini' | 'mistral' | 'custom'
  providerName?: string
  baseUrl?: string
  apiKey?: string
  connectionId?: string
  cloudCredentialStatus?: 'connected' | 'reconnect'
  remoteModelName?: string
  url?: string
  sizeBytes?: number
  ramRequiredGb?: number
  contextLength?: number
  parameters?: string
  quant?: string
  type?: string
  description?: string[]
  download?: ModelDownloadProgress
  addedAt: string
}

export type AppTheme = 'light' | 'sarah-and-duck'
export type AppFontSize = 'extra-small' | 'small' | 'normal' | 'large' | 'extra-large'
export type SuggestionMode = 'on' | 'off'
export type SearchMode = 'vector' | 'keyword' | 'hybrid'
export interface LibrarySearchOptions extends EmbeddingOptions {
  typoCorrectionEnabled?: boolean
}

export type ExplanationDepth = 'simple' | 'standard' | 'detailed'

export interface ApplicationSettings {
  theme: AppTheme
  fontSize: AppFontSize
  defaultModelId: string
  suggestionMode: SuggestionMode
  searchMode: SearchMode
  typoCorrectionEnabled: boolean
  explanationDepthEnabled: boolean
  explanationDepth: ExplanationDepth
  followUpSuggestionCount: number
  showSources: boolean
  embeddingGpuEnabled: boolean
}

export interface ModelRuntimeSettings {
  systemMessage: string
  suggestedFollowUpPrompt: string
  starterQuestionPrompt?: string
  contextLength: number
  maxLength: number
  thinking?: boolean
  temperature: number
  topP: number
  topK: number
  minP: number
  repeatPenalty: number
}

export type MessageRole = 'user' | 'assistant'

export interface ChatSource {
  title: string
  locator: string
  excerpt: string
  context?: string
  materialId?: string
  sourceId?: string
  chunkId?: string
  chunkRowid?: number | string
  chunkKind?: string
  sourceUnitId?: string
  sourceUnitComplete?: boolean
  sourceChunkIds?: string[]
  tokensmithChunkId?: string
  tokensmithChapter?: string
  tokensmithChunkKind?: string
  queryTerms?: string[]
  keywordTerms?: string[]
  documentId?: number | string
  documentTitle?: string
  collectionName?: string
  sectionHeader?: string
  path?: string
  lineFrom?: number
  lineTo?: number
  pageStart?: number
  pageEnd?: number
  thumbnailPath?: string
  chunkSize?: number
  score?: number
  retrievalMode?: 'vector' | 'keyword' | 'hybrid' | 'starter'
  embeddingModel?: string
  chunkEmbeddingModel?: string
}

export interface ChatSelectedPassage {
  messageId: string
  role: MessageRole
  text: string
  question?: string
}

export interface ChatMessage {
  id: string
  role: MessageRole
  text: string
  selectedPassage?: ChatSelectedPassage
  sources?: ChatSource[]
  conversationContextMode?: 'standalone' | 'contextual' | 'clarify'
  // The depth this answer was written at, so the reader knows what produced it.
  explanationDepth?: ExplanationDepth
  answerContext?: {
    prompt: string
    selectedPassage?: ChatSelectedPassage
    answerPrompt?: string
    retrievalQuery?: string
    conversationContextMode?: 'standalone' | 'contextual'
    referenceExchange?: { question: string; answer: string }
  }
  simplerExplanation?: { text: string; sources: ChatSource[]; responseDurationMs: number }
  explanationView?: 'original' | 'simple'
  responseDurationMs?: number
  followUpSuggestions?: string[]
  followUpError?: string
  kind?: 'chat' | 'quizQuestion' | 'quizAnswer' | 'quizFeedback'
  quiz?: {
    questionNumber?: number
    totalQuestions?: number
    complete?: boolean
  }
}

export interface QuizState {
  active: boolean
  questionNumber: number
  totalQuestions: number
  currentQuestion: string
  currentSources: ChatSource[]
  usedSourceKeys?: string[]
}

export interface Conversation {
  id: string
  title: string
  period: 'Today' | 'This week'
  messages: ChatMessage[]
  quizState?: QuizState
}

export interface TokenSmithSettings {
  maxSources: number
  application: ApplicationSettings
  modelDefaults: ModelRuntimeSettings
  modelSettingsById: Record<string, Partial<ModelRuntimeSettings>>
}

export interface AppStateSnapshot {
  version: 1
  appVersion: string
  activeScreen: ScreenId
  activeConversationId: string
  conversations: Conversation[]
  materials: CourseMaterial[]
  models: LocalModel[]
  selectedModelId: string
  selectedEmbeddingModelId: string
  settings: TokenSmithSettings
  updatedAt: string
}
