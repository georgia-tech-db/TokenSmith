export const defaultOllamaBaseUrl = 'http://127.0.0.1:11434'
export const recommendedOllamaChatModel = 'llama3'
export const recommendedOllamaEmbeddingModel = 'nomic-embed-text'

export interface OllamaModelInfo {
  name: string
  model?: string
  size?: number
  digest?: string
  modifiedAt?: string
  details?: {
    family?: string
    parameterSize?: string
    quantizationLevel?: string
  }
}

export interface OllamaStatus {
  baseUrl: string
  running: boolean
  installedApp: boolean
  models: OllamaModelInfo[]
  recommendedChatModel: string
  recommendedEmbeddingModel: string
  hasRecommendedChatModel: boolean
  hasRecommendedEmbeddingModel: boolean
  error?: string
}

export interface OllamaOpenResult {
  opened: boolean
  message?: string
}

export interface OllamaPullResult {
  model: string
  status: string
}

export interface OllamaDeleteResult {
  model: string
  status: string
}

export interface OllamaPullProgress {
  model: string
  status: 'starting' | 'downloading' | 'complete' | 'incomplete' | 'removed' | 'error'
  percent: number
  completed?: number
  total?: number
  message?: string
  digest?: string
  error?: string
}
