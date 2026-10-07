import type { LatencyDetail, LatencySpan } from '../../shared/latency-trace'
import type { SourceContextBudget } from './study-chat-format'

export interface OllamaTimingPayload {
  done_reason?: string
  prompt_eval_count?: number
  eval_count?: number
  load_duration?: number
  prompt_eval_duration?: number
  eval_duration?: number
}

function finiteNonNegative(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

function nanosecondsToMilliseconds(value: unknown): number | undefined {
  const nanoseconds = finiteNonNegative(value)
  return nanoseconds === undefined ? undefined : nanoseconds / 1_000_000
}

export function promptPreparationSpan(
  startedAt: number,
  durationMs: number,
  inputSourceCount: number,
  budget: SourceContextBudget
): LatencySpan {
  return {
    stage: 'Prompt preparation',
    startedAt,
    durationMs: Math.max(0, durationMs),
    inCount: Math.max(0, inputSourceCount),
    outCount: budget.includedSourceCount,
    details: {
      answer_reserve_tokens: budget.answerReserveTokens,
      context_tokens: budget.modelContextTokens,
      estimated_prompt_tokens: budget.estimatedPromptTokens,
      safety_margin_tokens: budget.safetyMarginTokens,
      source_budget_tokens: budget.sourceBudgetTokens,
      source_tokens: budget.usedSourceTokens,
      truncated_sources: budget.truncatedSourceCount
    }
  }
}

export function ollamaCompletionDetails(payload?: OllamaTimingPayload): Record<string, LatencyDetail> | undefined {
  if (!payload) return undefined
  const details: Record<string, LatencyDetail> = {}
  if (payload.done_reason) details.done_reason = payload.done_reason
  if (finiteNonNegative(payload.prompt_eval_count) !== undefined) details.prompt_tokens = payload.prompt_eval_count ?? 0
  if (finiteNonNegative(payload.eval_count) !== undefined) details.generated_tokens = payload.eval_count ?? 0
  return Object.keys(details).length ? details : undefined
}

export function ollamaTimingChildren(payload: OllamaTimingPayload | undefined, startedAt: number): LatencySpan[] {
  if (!payload) return []
  const timings = [
    ['Model load', nanosecondsToMilliseconds(payload.load_duration)],
    ['Prompt evaluation', nanosecondsToMilliseconds(payload.prompt_eval_duration)],
    ['Token generation', nanosecondsToMilliseconds(payload.eval_duration)]
  ] as const
  return timings.flatMap(([stage, durationMs]) => durationMs === undefined
    ? []
    : [{ stage, startedAt, durationMs }])
}
