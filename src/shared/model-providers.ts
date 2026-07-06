export type RemoteProviderId = 'groq' | 'openai' | 'gemini' | 'mistral' | 'custom'

export interface RemoteProviderCatalogItem {
  id: RemoteProviderId
  name: string
  baseUrl?: string
  description: string
  apiKeyUrl?: string
  isCustom?: boolean
}

export type HuggingFaceSort = 'default' | 'likes' | 'downloads' | 'recent'
export type HuggingFaceSortDirection = 'asc' | 'desc'

export interface HuggingFaceSearchOptions {
  sort: HuggingFaceSort
  direction: HuggingFaceSortDirection
  limit: number
}

export const remoteProviderCatalog: RemoteProviderCatalogItem[] = [
  {
    id: 'groq',
    name: 'Groq',
    baseUrl: 'https://api.groq.com/openai/v1/',
    description:
      'Low-latency OpenAI-compatible inference for hosted open models. Add an API key, choose chat or embedder, then load compatible models.',
    apiKeyUrl: 'https://console.groq.com/keys'
  },
  {
    id: 'openai',
    name: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1/',
    description:
      'OpenAI-compatible hosted chat and embedding models. Add an API key, choose the model type, then load compatible models.',
    apiKeyUrl: 'https://platform.openai.com/api-keys'
  },
  {
    id: 'gemini',
    name: 'Gemini',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    description:
      'Google Gemini models through Google AI Studio. Add an API key, choose chat or embedder, then load compatible models.',
    apiKeyUrl: 'https://aistudio.google.com/apikey'
  },
  {
    id: 'mistral',
    name: 'Mistral',
    baseUrl: 'https://api.mistral.ai/v1/',
    description:
      'Hosted Mistral models through an OpenAI-compatible API. Add an API key, choose chat or embedder, then load compatible models.',
    apiKeyUrl: 'https://console.mistral.ai/api-keys'
  },
  {
    id: 'custom',
    name: 'Custom',
    description:
      'Connect any OpenAI-compatible chat or embeddings endpoint by entering its base URL, API key, and model name.',
    isCustom: true
  }
]
