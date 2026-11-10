import type { DeviceCapabilities } from './device-capabilities'
import type { ModelLoadingEvent } from './model-loading'
import { estimateWait, recordTiming, type TimingContext, type TimingSample, type WaitEstimate } from './generation-estimate'

export interface ProgressStart extends TimingContext {
  id: string
  conversationId: string
  label: string
  onStop: () => void
  materialIds?: string[]
}
export interface GenerationProgress extends ProgressStart { startedAt: number; estimate: WaitEstimate; complete?: boolean }
interface Loading { startedAt: number; context: TimingContext; estimate: WaitEstimate }
interface Job extends GenerationProgress { loads: Map<string, Loading>; pausedAt?: number }

export function createGenerationProgressController(options: {
  publish: (value: GenerationProgress | null) => void
  samples: TimingSample[]
  save: (samples: TimingSample[]) => void
  device?: () => DeviceCapabilities | undefined
  now?: () => number
}) {
  const now = options.now ?? (() => performance.now())
  const jobs = new Map<string, Job>()
  let samples = options.samples
  const estimate = (context: TimingContext) => estimateWait(context, samples, options.device?.())
  const learn = (context: TimingContext, ms: number) => {
    samples = recordTiming(samples, context, ms)
    options.save(samples)
  }
  const publish = () => {
    const job = [...jobs.values()].find(job => job.kind !== 'starter') ?? [...jobs.values()][0]
    if (!job) { options.publish(null); return }
    const loading = job.loads.values().next().value as Loading | undefined
    options.publish(loading ? { ...job, label: 'Loading model', startedAt: loading.startedAt, estimate: loading.estimate } : { ...job })
  }
  return {
    stopForCollection(collectionId: string) {
      for (const job of [...jobs.values()]) if (job.materialIds?.some(id => id.startsWith(`${collectionId}/`))) { job.onStop(); jobs.delete(job.id) }
      publish()
    },
    begin(start: ProgressStart) {
      jobs.set(start.id, { ...start, startedAt: now(), estimate: estimate(start), loads: new Map() })
      publish()
    },
    stage(id: string, label: string, context?: Partial<TimingContext>) {
      const job = jobs.get(id)
      if (!job) return
      Object.assign(job, context, { label })
      if (context) job.estimate = estimate(job)
      publish()
    },
    modelLoading(event: ModelLoadingEvent) {
      const job = jobs.get(event.requestId)
      if (!job) return // Late events cannot revive canceled or completed work.
      if (event.state === 'loading') {
        if (job.loads.has(event.loadId)) return
        const context: TimingContext = { kind: 'model-load', model: event.model, loadContextLength: event.contextLength }
        if (!job.loads.size) job.pausedAt = now()
        job.loads.set(event.loadId, { startedAt: now(), context, estimate: estimate(context) })
      } else {
        const loading = job.loads.get(event.loadId)
        if (!loading) return
        if (event.state === 'ready') learn(loading.context, event.durationMs ?? now() - loading.startedAt)
        job.loads.delete(event.loadId)
        if (!job.loads.size && job.pausedAt !== undefined) {
          // Exclude the union of loading intervals, including overlapping searches.
          job.startedAt += now() - job.pausedAt
          job.pausedAt = undefined
          job.estimate = estimate(job)
        }
      }
      publish()
    },
    next(id: string, label: string, context: Partial<TimingContext>) {
      const job = jobs.get(id)
      if (!job) return
      if (!job.loads.size) learn(job, now() - job.startedAt)
      Object.assign(job, context, { label, startedAt: now(), pausedAt: job.loads.size ? now() : undefined })
      job.estimate = estimate(job)
      publish()
    },
    finish(id: string, label: string, shouldLearn = true) {
      const job = jobs.get(id)
      if (!job) return
      if (shouldLearn && !job.loads.size) learn(job, now() - job.startedAt)
      jobs.delete(id)
      if (jobs.size) publish()
      else options.publish({ ...job, label, complete: true })
    },
    cancel(id: string) { if (jobs.delete(id)) publish() }
  }
}
export type GenerationProgressApi = ReturnType<typeof createGenerationProgressController>
