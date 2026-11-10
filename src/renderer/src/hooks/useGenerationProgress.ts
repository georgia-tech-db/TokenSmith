import { useEffect, useMemo, useRef, useState } from 'react'
import type { DeviceCapabilities } from '@shared/device-capabilities'
import { readTimingSamples, timingStorageKey, type TimingSample } from '@shared/generation-estimate'
import { createGenerationProgressController, type GenerationProgress } from '@shared/generation-progress'
export type { ProgressStart, GenerationProgress, GenerationProgressApi } from '@shared/generation-progress'

export function useGenerationProgress() {
  const [current, setCurrent] = useState<GenerationProgress | null>(null)
  const device = useRef<DeviceCapabilities | undefined>(undefined)
  useEffect(() => {
    let disposed = false
    void window.tokensmith?.getDeviceCapabilities?.().then(value => { if (!disposed) device.current = value }).catch(() => {})
    return () => { disposed = true }
  }, [])
  const api = useMemo(() => {
    let samples: TimingSample[] = []
    try { samples = readTimingSamples(JSON.parse(localStorage.getItem(timingStorageKey) || '[]')) } catch { /* Estimates still work without storage. */ }
    return createGenerationProgressController({ samples, device: () => device.current, publish: setCurrent,
      save: value => { try { localStorage.setItem(timingStorageKey, JSON.stringify(value)) } catch { /* Keep learning in memory. */ } } })
  }, [])
  useEffect(() => window.tokensmith?.onModelLoading?.(event => api.modelLoading(event)), [api])
  useEffect(() => {
    if (!current?.complete) return
    const timer = window.setTimeout(() => setCurrent(value => value?.id === current.id && value.complete ? null : value), 6000)
    return () => window.clearTimeout(timer)
  }, [current])
  return { current, api }
}
