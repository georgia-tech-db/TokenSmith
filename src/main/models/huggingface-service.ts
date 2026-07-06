import type { ModelCatalogItem } from '../../shared/model-catalog'
import type { HuggingFaceSearchOptions } from '../../shared/model-providers'

interface HuggingFaceSibling {
  rfilename?: string
  size?: number
}

interface HuggingFaceModelResponse {
  id?: string
  modelId?: string
  author?: string
  likes?: number
  downloads?: number
  lastModified?: string
  siblings?: HuggingFaceSibling[]
  config?: {
    model_type?: string
  }
}

function encodeHuggingFacePath(filename: string): string {
  return filename.split('/').map((part) => encodeURIComponent(part)).join('/')
}

function safeLocalFilename(repoId: string, remoteFilename: string): string {
  const leaf = remoteFilename.split('/').filter(Boolean).pop() ?? remoteFilename
  const repoSlug = repoId.replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '')
  const leafSlug = leaf.replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '')
  const filename = `${repoSlug}-${leafSlug}`.replace(/-+/g, '-')

  return filename.toLowerCase().endsWith('.gguf') ? filename : `${filename}.gguf`
}

function quantFromFilename(filename: string): string {
  const match = filename.match(/(?:^|[-_. ])(q\d(?:[-_][a-z0-9]+)*|f16|f32|fp16|bf16)(?:\.gguf)?$/i)
  return match?.[1] ?? 'GGUF'
}

function quantRank(filename: string): number {
  const quant = quantFromFilename(filename).toLowerCase()
  const ranks = ['q4_0', 'q4_1', 'q4_k_m', 'q4_k_s', 'q5_k_m', 'q5_k_s', 'q8_0', 'f16', 'f32']
  const index = ranks.indexOf(quant)
  return index >= 0 ? index : ranks.length
}

function parametersFromText(text: string): string {
  const match = text.match(/(?:^|[-_./ ])(\d+(?:\.\d+)?)\s*b(?:[-_./ ]|$)/i)
  return match ? `${match[1]} billion` : 'Unknown'
}

function ramEstimateGb(sizeBytes: number): number {
  if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
    return 0
  }

  return Math.max(1, Math.ceil(sizeBytes / 1024 / 1024 / 1024) + 2)
}

function modelType(response: HuggingFaceModelResponse, filename: string): string {
  const configuredType = response.config?.model_type
  if (configuredType) {
    return configuredType
  }

  const lowerText = `${response.id ?? ''} ${filename}`.toLowerCase()
  if (lowerText.includes('llama')) return 'llama'
  if (lowerText.includes('qwen')) return 'qwen'
  if (lowerText.includes('mistral')) return 'mistral'
  if (lowerText.includes('deepseek')) return 'deepseek'
  if (lowerText.includes('gemma')) return 'gemma'
  if (lowerText.includes('phi')) return 'phi'
  return 'gguf'
}

function bestGgufSibling(siblings: HuggingFaceSibling[] = []): HuggingFaceSibling | undefined {
  return siblings
    .filter((sibling) => {
      const filename = sibling.rfilename ?? ''
      return filename.toLowerCase().endsWith('.gguf') && !filename.toLowerCase().includes('mmproj')
    })
    .sort((left, right) => quantRank(left.rfilename ?? '') - quantRank(right.rfilename ?? ''))[0]
}

function sortQuery(sort: HuggingFaceSearchOptions['sort']): string | undefined {
  if (sort === 'likes') return 'likes'
  if (sort === 'downloads') return 'downloads'
  if (sort === 'recent') return 'lastModified'
  return undefined
}

async function linkedSize(url: string): Promise<number | undefined> {
  try {
    const response = await fetch(url, {
      method: 'HEAD',
      headers: {
        'Accept-Encoding': 'identity'
      }
    })
    const size = Number(response.headers.get('x-linked-size') ?? response.headers.get('content-length'))
    return Number.isFinite(size) && size > 0 ? size : undefined
  } catch {
    return undefined
  }
}

function formatPublishedDate(value?: string): string {
  if (!value) {
    return 'Unknown publish date'
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

export async function searchHuggingFaceModels(
  query: string,
  options: HuggingFaceSearchOptions
): Promise<ModelCatalogItem[]> {
  const search = query.trim()
  if (!search) {
    return []
  }

  const params = new URLSearchParams({
    filter: 'gguf',
    search,
    full: 'true',
    config: 'true'
  })

  if (options.limit > 0) {
    params.set('limit', String(options.limit))
  }

  const sort = sortQuery(options.sort)
  if (sort) {
    params.set('sort', sort)
    params.set('direction', options.direction === 'asc' ? '1' : '-1')
  }

  const response = await fetch(`https://huggingface.co/api/models?${params.toString()}`, {
    headers: {
      Accept: 'application/json'
    }
  })

  if (!response.ok) {
    throw new Error(`HuggingFace search failed with HTTP ${response.status}.`)
  }

  const payload = (await response.json()) as HuggingFaceModelResponse[]
  const items: ModelCatalogItem[] = []

  for (const model of payload) {
    const repoId = model.id ?? model.modelId
    const sibling = bestGgufSibling(model.siblings)
    const remoteFilename = sibling?.rfilename

    if (!repoId || !remoteFilename) {
      continue
    }

    const filename = safeLocalFilename(repoId, remoteFilename)
    const url = `https://huggingface.co/${repoId}/resolve/main/${encodeHuggingFacePath(remoteFilename)}`
    const sizeBytes = sibling.size ?? (await linkedSize(url)) ?? 0
    const likes = model.likes ?? 0
    const downloads = model.downloads ?? 0

    items.push({
      id: `hf:${repoId}:${remoteFilename}`,
      name: repoId,
      filename,
      sourceFilename: remoteFilename,
      sizeBytes,
      ramRequiredGb: ramEstimateGb(sizeBytes),
      parameters: parametersFromText(`${repoId} ${remoteFilename}`),
      quant: quantFromFilename(remoteFilename),
      type: modelType(model, remoteFilename),
      description: [
        `Created by ${model.author ?? repoId.split('/')[0] ?? 'Unknown'}`,
        `File: ${remoteFilename}`,
        `Published on ${formatPublishedDate(model.lastModified)}`,
        `This model has ${likes.toLocaleString()} likes`,
        `This model has ${downloads.toLocaleString()} downloads`,
        `More info: https://huggingface.co/${repoId}`
      ],
      url,
      tags: ['huggingface']
    })
  }

  return items
}
