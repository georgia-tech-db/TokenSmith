import type { LocalModel } from './app-state'
import type { EmbeddingOptions } from './embedding-settings'

export interface PreparationSettings {
  mode: 'ai' | 'basic'
  extractTopics?: boolean
  ocr?: 'none' | 'rapid' | 'gemma'
  modelId?: string
  modelName?: string
  instructions: string
  documentInstructions: Record<string, string>
}

export interface IndexMaterialOptions extends EmbeddingOptions {
  resume?: boolean
  title?: string
  cleaningProfileId?: import('./cleaning').CleaningProfileId
  cleaningRuleIds?: import('./cleaning').CleaningRuleId[]
  preparation?: PreparationSettings
  preparationModel?: LocalModel
}

export interface PreparedChunk {
  text: string
  sectionHeader?: string
  tokensmithChunkKind?: string
  pageStart?: number
  pageEnd?: number
  lineFrom?: number
  lineTo?: number
  reason?: string
  parentId?: string
  part?: number
  parts?: number
}

export interface PreparationReport {
  documents: Array<{
    title: string
    path: string
    status: string
    error?: string
    warning?: string
    pageCount?: number
    chunkCount: number
    chunks: PreparedChunk[]
  }>
  updatedAt?: string
}

export const defaultPreparation = (): PreparationSettings => ({
  mode: 'ai', ocr: 'none', extractTopics: true, instructions: '', documentInstructions: {}
})

/** Keep document preparation and chapter topics on the same generator. */
export function preparationGenerator(models: LocalModel[], selectedId?: string, fallbackId?: string, ocr?: PreparationSettings['ocr']): LocalModel | undefined {
  const generators = models.filter(model => model.status === 'ready' && model.role !== 'embedder' &&
    (model.engine === 'ollama' || model.engine === 'remote') &&
    (ocr !== 'gemma' || isGemmaModel(model)))
  if (selectedId) return generators.find(model => model.id === selectedId)
  return generators.find(model => model.engine === 'ollama' && model.ollamaModelName === 'gemma4:e4b')
    ?? generators.find(model => model.id === fallbackId)
    ?? (ocr === 'gemma' ? generators[0] : undefined)
}

export function isGemmaModel(model: LocalModel): boolean {
  return /gemma/i.test([model.name, model.ollamaModelName, model.remoteModelName].join(' '))
}

export function preparationNeedsModel(settings: PreparationSettings): boolean {
  return settings.mode === 'ai' || settings.ocr === 'gemma' || settings.extractTopics !== false
}
