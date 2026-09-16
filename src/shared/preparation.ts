import type { LocalModel } from './app-state'

export interface PreparationSettings {
  mode: 'ai' | 'basic'
  modelId?: string
  modelName?: string
  instructions: string
  documentInstructions: Record<string, string>
}

export interface IndexMaterialOptions {
  resume?: boolean
  title?: string
  cleaningProfileId?: import('./cleaning').CleaningProfileId
  cleaningRuleIds?: import('./cleaning').CleaningRuleId[]
  preparation?: PreparationSettings
  preparationModel?: LocalModel
  typoCorrectionEnabled?: boolean
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
  mode: 'basic', instructions: '', documentInstructions: {}
})
