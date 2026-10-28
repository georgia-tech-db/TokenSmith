import type { LocalModel, ModelRuntimeSettings } from './app-state'
import { answerTokenAllowance, effectiveContextLength, reportedContextLimit, usesAutomaticContext } from './model-context'

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
  const contextTokens = effectiveContextLength(model, settings)
  const answerReserve = answerTokenAllowance(model, settings)
  const baseLimit = clampNumber(configuredLimit, 4, 1, 8)
  // Metadata can still be loading when retrieval starts. Fetch candidates now;
  // the engine packs them against the discovered limit before generation.
  if (usesAutomaticContext(settings) && !reportedContextLimit(model, settings)) return 8
  const budgetLimit = Math.floor(Math.max(0, contextTokens - answerReserve - 512) / 800)
  return Math.max(baseLimit, Math.min(8, budgetLimit))
}
