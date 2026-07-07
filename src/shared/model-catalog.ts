import type { LocalModel, LocalModelRole } from './app-state'

export interface ModelCatalogItem {
  id: string
  name: string
  filename: string
  sourceFilename?: string
  sizeBytes: number
  ramRequiredGb: number
  parameters: string
  quant: string
  type: string
  description: string[]
  url: string
  tags?: string[]
  role?: LocalModelRole
}

export const tokenSmithTunedModels: ModelCatalogItem[] = [
  {
    id: 'nomic-embed-text-v1-5',
    name: 'Nomic Embed Text v1.5',
    filename: 'nomic-embed-text-v1.5.f16.gguf',
    sizeBytes: 274290560,
    ramRequiredGb: 1,
    parameters: '137 million',
    quant: 'f16',
    type: 'Bert',
    role: 'embedder',
    description: [
      'Local text embedding model for document collections',
      'Used to build vectors for retrieval',
      'Recommended local embedder for source-backed retrieval',
      'Default context length: 2048 tokens',
      '#embedder'
    ],
    url: 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf',
    tags: ['embedder']
  },
  {
    id: 'bge-small-en-v1-5-q8',
    name: 'BGE Small EN v1.5',
    filename: 'bge-small-en-v1.5-q8_0.gguf',
    sizeBytes: 36685152,
    ramRequiredGb: 1,
    parameters: '33 million',
    quant: 'q8_0',
    type: 'Bert',
    role: 'embedder',
    description: [
      'Small English embedding model for document collections',
      'Useful for testing alternate retrieval embeddings',
      'Lower memory footprint than Nomic Embed Text v1.5',
      'License: MIT',
      '#embedder'
    ],
    url: 'https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf',
    tags: ['embedder']
  },
  {
    id: 'reasoner-v1',
    name: 'Reasoner v1',
    filename: 'qwen2.5-coder-7b-instruct-q4_0.gguf',
    sizeBytes: 4431390720,
    ramRequiredGb: 8,
    parameters: '8 billion',
    quant: 'q4_0',
    type: 'qwen2',
    description: [
      'Based on Qwen2.5-Coder 7B',
      'Uses built-in javascript code interpreter',
      'Use for complex reasoning tasks that can be aided by computation analysis',
      'License: Apache License Version 2.0',
      '#reasoning'
    ],
    url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_0.gguf',
    tags: ['reasoning']
  },
  {
    id: 'deepseek-r1-distill-qwen-7b',
    name: 'DeepSeek-R1-Distill-Qwen-7B',
    filename: 'DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf',
    sizeBytes: 4444121056,
    ramRequiredGb: 8,
    parameters: '7 billion',
    quant: 'q4_0',
    type: 'deepseek',
    description: [
      'The official Qwen2.5-Math-7B distillation of DeepSeek-R1',
      'License: MIT',
      'No restrictions on commercial use',
      '#reasoning'
    ],
    url: 'https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf',
    tags: ['reasoning']
  },
  {
    id: 'llama-3-8b-instruct',
    name: 'Llama 3 8B Instruct',
    filename: 'Meta-Llama-3-8B-Instruct.Q4_0.gguf',
    sizeBytes: 4661724384,
    ramRequiredGb: 8,
    parameters: '8 billion',
    quant: 'q4_0',
    type: 'LLaMA3',
    description: [
      'Fast responses',
      'Chat based model',
      'Accepts system prompts in Llama 3 format',
      'Trained by Meta',
      'License: Meta Llama 3 Community License'
    ],
    url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf'
  },
  {
    id: 'llama-3-2-3b-instruct',
    name: 'Llama 3.2 3B Instruct',
    filename: 'Llama-3.2-3B-Instruct-Q4_0.gguf',
    sizeBytes: 1921909280,
    ramRequiredGb: 4,
    parameters: '3 billion',
    quant: 'q4_0',
    type: 'LLaMA3',
    description: [
      'Fast responses',
      'Instruct model',
      'Multilingual dialogue use',
      'Trained by Meta',
      'License: Meta Llama 3.2 Community License'
    ],
    url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf'
  }
]

export const firstRunRecommendedModelIds = ['llama-3-8b-instruct', 'nomic-embed-text-v1-5'] as const

export function normalizeModelFilename(filename?: string): string | undefined {
  return filename?.toLowerCase()
}

export function catalogItemForFilename(filename?: string): ModelCatalogItem | undefined {
  const normalizedFilename = normalizeModelFilename(filename)

  if (!normalizedFilename) {
    return undefined
  }

  return tokenSmithTunedModels.find((item) => normalizeModelFilename(item.filename) === normalizedFilename)
}

export function catalogItemToLocalModel(
  item: ModelCatalogItem,
  modelId: string,
  path: string | undefined,
  addedAt: string,
  status: LocalModel['status'] = path ? 'ready' : 'downloading'
): LocalModel {
  const role = item.role ?? 'generator'

  return {
    id: modelId,
    name: item.name,
    engine: 'python',
    role,
    status,
    source: 'downloaded',
    catalogId: item.id,
    filename: item.filename,
    path,
    embeddingPath: undefined,
    url: item.url,
    sizeBytes: item.sizeBytes,
    ramRequiredGb: item.ramRequiredGb,
    parameters: item.parameters,
    quant: item.quant,
    type: item.type,
    description: item.description,
    addedAt
  }
}
