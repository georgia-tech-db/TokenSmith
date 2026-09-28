import { useEffect, useMemo, useRef, useState } from 'react'
import type { DeviceCapabilities } from '@shared/device-capabilities'
import { estimateWait, readTimingSamples, recordTiming, timingStorageKey, type TimingContext, type TimingSample, type WaitEstimate } from '@shared/generation-estimate'

export interface ProgressStart extends TimingContext {
  id: string
  conversationId: string
  label: string
  onStop: () => void
}
export interface GenerationProgress extends ProgressStart {
  startedAt: number
  estimate: WaitEstimate
  complete?: boolean
}
export interface GenerationProgressApi {
  begin: (start: ProgressStart) => void
  stage: (id: string, label: string, context?: Partial<TimingContext>) => void
  next: (id: string, label: string, context: Partial<TimingContext>) => void
  finish: (id: string, label: string, learn?: boolean) => void
  cancel: (id: string) => void
}

export function useGenerationProgress() {
  const [current, setCurrent] = useState<GenerationProgress | null>(null)
  const device = useRef<DeviceCapabilities | undefined>(undefined)
  useEffect(() => {
    let disposed = false
    void window.tokensmith?.getDeviceCapabilities?.().then(value => { if (!disposed) device.current = value }).catch(() => {})
    return () => { disposed = true }
  }, [])
  const api = useMemo<GenerationProgressApi>(() => {
    const jobs = new Map<string, GenerationProgress>()
    let samples: TimingSample[] = []
    try { samples = readTimingSamples(JSON.parse(localStorage.getItem(timingStorageKey) || '[]')) } catch { /* Estimates still work without storage. */ }
    const publish = () => setCurrent([...jobs.values()].find(job => job.kind !== 'starter') ?? [...jobs.values()][0] ?? null)
    const learn = (job: GenerationProgress) => {
      samples = recordTiming(samples, job, performance.now() - job.startedAt)
      try { localStorage.setItem(timingStorageKey, JSON.stringify(samples)) } catch { /* In-memory learning remains available. */ }
    }
    return {
      begin(start) {
        const job = { ...start, startedAt: performance.now(), estimate: estimateWait(start, samples, device.current) }
        jobs.set(start.id, job)
        publish()
      },
      stage(id, label, context) {
        const job = jobs.get(id)
        if (!job) return
        const updated = { ...job, ...context, label }
        if (context) updated.estimate = estimateWait(updated, samples, device.current)
        jobs.set(id, updated)
        publish()
      },
      next(id, label, context) {
        const job = jobs.get(id)
        if (!job) return
        learn(job)
        const next = { ...job, ...context, label, startedAt: performance.now() }
        next.estimate = estimateWait(next, samples, device.current)
        jobs.set(id, next)
        publish()
      },
      finish(id, label, shouldLearn = true) {
        const job = jobs.get(id)
        if (!job) return
        if (shouldLearn) learn(job)
        jobs.delete(id)
        if (jobs.size) publish()
        else setCurrent({ ...job, label, complete: true })
      },
      cancel(id) {
        if (jobs.delete(id)) publish()
      }
    }
  }, [])
  useEffect(() => {
    if (!current?.complete) return
    const timer = window.setTimeout(() => setCurrent(value => value?.id === current.id && value.complete ? null : value), 6000)
    return () => window.clearTimeout(timer)
  }, [current])
  return { current, api }
}
