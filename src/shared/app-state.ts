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
  error?: string
  indexing?: MaterialIndexProgress
}

export interface LocalModel {
  id: string
  name: string
  engine: 'python' | 'ollama' | 'remote'
  role?: LocalModelRole
  status: 'ready' | 'needsRuntime' | 'missing' | 'downloading' | 'incomplete' | 'downloadError'
  source?: 'bundled' | 'local' | 'downloaded' | 'ollama' | 'remote'
  filename?: string
  path?: string
  embeddingPath?: string
  ollamaModelName?: string
  ollamaBaseUrl?: string
  providerId?: 'groq' | 'openai' | 'gemini' | 'mistral' | 'custom'
  providerName?: string
  baseUrl?: string
  apiKey?: string
  remoteModelName?: string
  url?: string
  sizeBytes?: number
  ramRequiredGb?: number
  parameters?: string
  quant?: string
  type?: string
  description?: string[]
  download?: ModelDownloadProgress
  addedAt: string
}

export type AppTheme = 'light' | 'system'
export type AppFontSize = 'small' | 'medium' | 'large'
export type SuggestionMode = 'on' | 'off'
export type ComputeDevice = 'applicationDefault' | 'cpu' | 'gpu'

export interface ApplicationSettings {
  theme: AppTheme
  fontSize: AppFontSize
  defaultModelId: string
  suggestionMode: SuggestionMode
  followUpSuggestionCount: number
  showSources: boolean
  cpuThreads: number
}

export interface ModelRuntimeSettings {
  systemMessage: string
  chatTemplate: string
  suggestedFollowUpPrompt: string
  contextLength: number
  maxLength: number
  promptBatchSize: number
  temperature: number
  topP: number
  topK: number
  minP: number
  repeatPenaltyTokens: number
  repeatPenalty: number
  gpuLayers: number
  device: ComputeDevice
}

export type MessageRole = 'user' | 'assistant'

export interface ChatSource {
  title: string
  locator: string
  excerpt: string
  materialId?: string
  chunkId?: string
  chunkRowid?: number | string
  documentId?: number | string
  documentTitle?: string
  collectionName?: string
  sectionHeader?: string
  path?: string
  pageStart?: number
  pageEnd?: number
  thumbnailPath?: string
  chunkSize?: number
  score?: number
  retrievalMode?: 'vector'
  embeddingModel?: string
  chunkEmbeddingModel?: string
}

export interface ChatMessage {
  id: string
  role: MessageRole
  text: string
  sources?: ChatSource[]
  followUpSuggestions?: string[]
  followUpError?: string
}

export interface Conversation {
  id: string
  title: string
  period: 'Today' | 'This week'
  messages: ChatMessage[]
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
