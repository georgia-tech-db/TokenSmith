import type { LocalModel, ModelRuntimeSettings } from './app-state'

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  const numericValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numericValue) ? Math.max(min, Math.min(max, Math.round(numericValue))) : fallback
}

// Candidate count budgeting is independent of question routing and source ranking.
export function modelAwareRetrievalLimit(
  configuredLimit: number,
  model?: LocalModel,
  settings?: Partial<ModelRuntimeSettings>
): number {
  const configured = clampNumber(settings?.contextLength, 2048, 512, 32768)
  const discovered = clampNumber(model?.contextLength, 0, 0, 32768)
  const contextTokens = discovered > 0 ? Math.min(discovered, Math.max(configured, 8192)) : configured
  const answerReserve = clampNumber(settings?.maxLength, 768, 256, 1024)
  const baseLimit = clampNumber(configuredLimit, 4, 1, 8)
  const budgetLimit = Math.floor(Math.max(0, contextTokens - answerReserve - 512) / 800)
  return Math.max(baseLimit, Math.min(8, budgetLimit))
}
