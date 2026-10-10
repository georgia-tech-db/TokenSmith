import type { LocalModel, ModelRuntimeSettings } from './app-state'

export interface ModelContextMetadata {
  contextLength?: number
  inputTokenLimit?: number
  maxOutputTokens?: number
}

export const fallbackContextLength = 8192

export function mergeModelContextMetadata(model: LocalModel, metadata: ModelContextMetadata): LocalModel {
  // A fresh input-only limit must replace an old, possibly clipped total limit.
  return metadata.contextLength || metadata.inputTokenLimit
    ? { ...model, contextLength: undefined, inputTokenLimit: undefined, maxOutputTokens: undefined, ...metadata }
    : { ...model, ...metadata }
}

export function normalizeTokenLimit(value: unknown): number | undefined {
  if (typeof value !== 'number' && typeof value !== 'string') return undefined
  const tokens = Number(value)
  return Number.isSafeInteger(tokens) && tokens > 0 ? tokens : undefined
}

export function usesAutomaticContext(settings?: Partial<ModelRuntimeSettings>): boolean {
  return settings?.contextLengthMode === 'auto' ||
    (settings?.contextLengthMode !== 'manual' && !normalizeTokenLimit(settings?.contextLength))
}

// Old versions persisted their defaults without recording whether the user changed them.
// Migrate shipped defaults to Auto; preserve other sizes and explicit Custom choices.
export function migratedContextMode(settings?: Partial<ModelRuntimeSettings>): 'auto' | 'manual' {
  if (settings?.contextLengthMode === 'auto' || settings?.contextLengthMode === 'manual') return settings.contextLengthMode
  const value = normalizeTokenLimit(settings?.contextLength)
  return !value || value === 2048 || value === 8192 ? 'auto' : 'manual'
}

export function answerTokenAllowance(model?: ModelContextMetadata, settings?: Partial<ModelRuntimeSettings>): number {
  return Math.min(normalizeTokenLimit(model?.maxOutputTokens) ?? 8192,
    Math.max(64, Math.min(8192, normalizeTokenLimit(settings?.maxLength) ?? 768)))
}

export function reportedContextLimit(model?: ModelContextMetadata, settings?: Partial<ModelRuntimeSettings>): number | undefined {
  const total = normalizeTokenLimit(model?.contextLength)
  const input = normalizeTokenLimit(model?.inputTokenLimit)
  // Gemini publishes separate input/output limits rather than a combined window.
  const inputAndAnswer = input ? input + answerTokenAllowance(model, settings) : undefined
  return total && inputAndAnswer ? Math.min(total, inputAndAnswer) : total ?? inputAndAnswer
}

export function effectiveContextLength(model?: ModelContextMetadata, settings?: Partial<ModelRuntimeSettings>): number {
  const reported = reportedContextLimit(model, settings)
  if (usesAutomaticContext(settings)) return reported ?? fallbackContextLength
  const configured = Math.max(512, normalizeTokenLimit(settings?.contextLength) ?? fallbackContextLength)
  return Math.min(configured, reported ?? configured)
}

export function localContextForTokens(requiredTokens: number, ceiling: number): number {
  // Small prompts should not allocate a model's entire 128K/1M KV cache.
  const rounded = 2 ** Math.ceil(Math.log2(Math.max(fallbackContextLength, requiredTokens)))
  return Math.min(ceiling, rounded)
}

export function contextMetadataIdentity(model: LocalModel): string {
  return JSON.stringify([model.id, model.engine, model.ollamaBaseUrl, model.ollamaModelName,
    model.baseUrl, model.remoteModelName, model.connectionId, model.cloudCredentialStatus, model.status])
}
