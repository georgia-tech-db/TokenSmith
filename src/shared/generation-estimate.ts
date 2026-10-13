import type { LocalModel, ModelRuntimeSettings } from './app-state'
import type { DeviceCapabilities } from './device-capabilities'

export type GenerationKind = 'answer' | 'starter' | 'follow-ups' | 'simplify' | 'quiz'
export interface TimingContext {
  kind: GenerationKind
  model: LocalModel
  settings?: ModelRuntimeSettings
  inputChars?: number
  contextual?: boolean
  count?: number
  depth?: string
}
export interface TimingSample { key: string; family: string; ms: number; at: number }
export interface WaitEstimate { expectedMs: number; lowerMs: number; upperMs: number; samples: number }
export const timingStorageKey = 'tokensmith-generation-timings-v1'

// Store opaque configuration keys and durations, never prompts, sources or credentials.
function fingerprint(value: string): string {
  let hash = 2166136261
  for (let i = 0; i < value.length; i++) hash = Math.imul(hash ^ value.charCodeAt(i), 16777619)
  return (hash >>> 0).toString(36)
}
export function timingKeys(context: TimingContext) {
  const { model, settings, kind } = context
  let endpoint = model.ollamaBaseUrl || model.baseUrl || 'local'
  try { const url = new URL(endpoint); endpoint = url.origin + url.pathname } catch { endpoint = 'local' }
  const family = fingerprint(JSON.stringify([model.engine, endpoint, model.ollamaModelName || model.remoteModelName || model.id, model.quant, kind, settings?.contextLength, settings?.contextLengthMode,
    model.contextLength, model.inputTokenLimit, settings?.maxLength, settings?.thinking, settings?.reasoningMode]))
  const key = fingerprint(JSON.stringify([family, settings?.contextLength, settings?.maxLength, settings?.thinking, settings?.reasoningMode,
    (context.inputChars ?? 0) < 4000 ? 'short' : (context.inputChars ?? 0) < 16000 ? 'medium' : 'long',
    Boolean(context.contextual), context.count, context.depth]))
  return { key, family }
}

export function readTimingSamples(value: unknown, now = Date.now()): TimingSample[] {
  if (!Array.isArray(value)) return []
  return value.filter((s): s is TimingSample => Boolean(s) && typeof s.key === 'string' && typeof s.family === 'string'
    && Number.isFinite(s.ms) && s.ms > 0 && s.ms <= 86_400_000
    && Number.isFinite(s.at) && s.at <= now && now - s.at < 90 * 86_400_000).slice(-200)
}

export function recordTiming(samples: TimingSample[], context: TimingContext, ms: number, now = Date.now()): TimingSample[] {
  if (!Number.isFinite(ms) || ms <= 0 || ms > 86_400_000) return samples
  return readTimingSamples([...samples, { ...timingKeys(context), ms, at: now }], now)
}

export function estimateWait(context: TimingContext, samples: TimingSample[], device?: DeviceCapabilities): WaitEstimate {
  const { key, family } = timingKeys(context)
  const exact = samples.filter(s => s.key === key).slice(-12)
  const nearby = samples.filter(s => s.family === family).slice(-12)
  const history = exact.length ? exact : nearby
  const modelName = `${context.model.ollamaModelName || context.model.remoteModelName || context.model.name}`.toLowerCase()
  // Broad priors, not hardware performance promises. Completed local runs replace these.
  let prior = context.model.engine === 'remote' ? 20_000
    : /26b/.test(modelName) ? 45_000 : /12b/.test(modelName) ? 125_000 : /e4b|4b/.test(modelName) ? 40_000 : 60_000
  if (context.model.engine !== 'remote' && device && !device.accelerators.some(a => a.runtimeSupport === 'supported')) prior *= 2.5
  if (context.kind === 'starter' || context.kind === 'follow-ups') prior *= 0.45 * Math.max(0.6, Math.min(2, (context.count ?? 3) / 3))
  else prior *= Math.max(0.5, Math.min(2, Math.sqrt((context.settings?.maxLength ?? 1536) / 1536)))
  if (context.settings?.thinking) prior *= 1.7
  if (context.contextual && context.kind === 'answer') prior *= 1.3
  let expectedMs = prior
  if (history.length) {
    let total = 0, weights = 0
    history.forEach((sample, index) => { const weight = Math.pow(0.8, history.length - 1 - index); total += sample.ms * weight; weights += weight })
    // After one observation, mostly trust the machine, while retaining a broad range.
    expectedMs = total / weights * (history.length === 1 ? 0.8 : 1) + (history.length === 1 ? prior * 0.2 : 0)
  }
  const deviation = history.length ? history.reduce((sum, s) => sum + Math.abs(s.ms - expectedMs), 0) / history.length : 0
  const uncertainty = Math.max(5000, expectedMs * (history.length < 3 || !exact.length ? 0.6 : 0.3), deviation * 1.5)
  return { expectedMs, lowerMs: Math.max(1000, expectedMs - uncertainty), upperMs: expectedMs + uncertainty, samples: history.length }
}

export function waitDisplay(estimate: WaitEstimate, elapsedMs: number) {
  const elapsed = Math.max(0, elapsedMs)
  if (elapsed >= estimate.upperMs) return { overdue: true, fraction: undefined, text: `Taking longer than usual · ${Math.floor(elapsed / 1000)}s elapsed` }
  const rounded = (ms: number) => Math.max(5, Math.ceil(ms / 5000) * 5)
  const low = rounded(estimate.lowerMs - elapsed)
  const high = rounded(estimate.upperMs - elapsed)
  return { overdue: false, fraction: Math.min(0.95, elapsed / estimate.upperMs),
    text: `About ${low === high ? high : `${low}–${high}`}s left` }
}
