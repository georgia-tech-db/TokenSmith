import { createHash } from 'node:crypto'
import type { LocalModel } from '../../shared/app-state'
import { mergeModelContextMetadata, normalizeTokenLimit, type ModelContextMetadata } from '../../shared/model-context'
import { defaultOllamaBaseUrl } from '../../shared/ollama'
import { remoteGeneratorFetch } from './remote-generator-network'

type Row = Record<string, unknown>
const record = (value: unknown): Row => value && typeof value === 'object' && !Array.isArray(value) ? value as Row : {}
const firstLimit = (...values: unknown[]) => values.map(normalizeTokenLimit).find(value => value !== undefined)

export function parseOllamaContextMetadata(payload: unknown): ModelContextMetadata {
  const row = record(payload), info = record(row.model_info), details = record(row.details)
  const architecture = typeof info['general.architecture'] === 'string' ? info['general.architecture'] : ''
  const textContext = firstLimit(info[`${architecture}.context_length`], details.context_length,
    ...Object.entries(info).filter(([key]) => /^[^.]+\.context_length$/.test(key)).map(([,value]) => value))
  // num_ctx in a Modelfile is a runtime default, not the model's supported maximum.
  return textContext ? { contextLength: textContext } : {}
}

export function parseRemoteContextMetadata(payload: unknown): ModelContextMetadata {
  const row = record(payload)
  const contextLength = firstLimit(row.context_window, row.max_context_length, row.context_length)
  const inputTokenLimit = firstLimit(row.inputTokenLimit, row.max_input_tokens)
  const maxOutputTokens = firstLimit(row.outputTokenLimit, row.max_completion_tokens, row.max_output_tokens)
  return { ...(contextLength ? { contextLength } : {}), ...(inputTokenLimit ? { inputTokenLimit } : {}),
    ...(maxOutputTokens ? { maxOutputTokens } : {}) }
}

export class ModelContextMetadataService {
  private cache = new Map<string, { expires: number; value: Promise<ModelContextMetadata> }>()
  constructor(private request: typeof fetch = (url, init) => remoteGeneratorFetch(url, init)) {}

  async get(model: LocalModel): Promise<ModelContextMetadata> {
    const key = JSON.stringify([model.engine, model.ollamaBaseUrl, model.ollamaModelName, model.baseUrl,
      model.remoteModelName, createHash('sha256').update(model.apiKey ?? '').digest('hex')])
    let cached = this.cache.get(key)
    if (!cached || cached.expires <= Date.now()) {
      const value = this.load(model).catch((): ModelContextMetadata => ({}))
      cached = { expires: Date.now() + 60_000, value }
      this.cache.set(key, cached)
      void value.then(metadata => {
        if ((metadata.contextLength || metadata.inputTokenLimit) && this.cache.get(key) === cached) {
          cached!.expires = Date.now() + 10 * 60_000
        }
      })
      if (this.cache.size > 200) this.cache.delete(this.cache.keys().next().value!)
    }
    return cached.value
  }

  private async load(model: LocalModel): Promise<ModelContextMetadata> {
    const read = async (url: string, init: RequestInit): Promise<unknown> => {
      const response = await this.request(url, { ...init, redirect: 'error', signal: AbortSignal.timeout(5000) })
      return response.ok ? response.json() : undefined
    }
    if (model.engine === 'ollama' && model.ollamaModelName) {
      const base = (model.ollamaBaseUrl || defaultOllamaBaseUrl).replace(/\/+$/, '').replace(/\/api$/, '')
      return parseOllamaContextMetadata(await read(`${base}/api/show`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ model: model.ollamaModelName })
      }))
    }
    if (model.engine !== 'remote' || !model.baseUrl || !model.remoteModelName || !model.apiKey?.trim()) return {}
    const base = model.baseUrl.trim().replace(/\/+$/, '')
    const endpoint = new URL(base)
    const name = model.remoteModelName.replace(/^models\//, '')
    if (endpoint.hostname === 'generativelanguage.googleapis.com' && endpoint.pathname.endsWith('/openai')) {
      return parseRemoteContextMetadata(await read(`${base.slice(0, -'/openai'.length)}/models/${encodeURIComponent(name)}`, {
        headers: { 'x-goog-api-key': model.apiKey.trim(), Accept: 'application/json' }
      }))
    }
    const init = { headers: { Authorization: `Bearer ${model.apiKey.trim()}`, Accept: 'application/json' } }
    const detail = await read(`${base}/models/${encodeURIComponent(model.remoteModelName)}`, init)
    const metadata = parseRemoteContextMetadata(detail)
    if (metadata.contextLength || metadata.inputTokenLimit) return metadata
    // OpenAI's Models API does not publish token limits. Do not invent a limit
    // from the model name or scrape documentation during a student's request.
    if (endpoint.hostname === 'api.openai.com') return metadata
    const list = record(await read(`${base}/models`, init)).data
    const match = Array.isArray(list) ? list.find(row => record(row).id === model.remoteModelName) : undefined
    return { ...metadata, ...parseRemoteContextMetadata(match) }
  }
}

const metadataService = new ModelContextMetadataService()
export const getModelContextMetadata = (model: LocalModel): Promise<ModelContextMetadata> => metadataService.get(model)

export async function withModelContextMetadata(model: LocalModel): Promise<LocalModel> {
  return mergeModelContextMetadata(model, await getModelContextMetadata(model))
}
