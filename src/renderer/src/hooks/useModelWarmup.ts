import { useEffect, useRef } from 'react'
import type { LocalModel, ModelRuntimeSettings } from '@shared/app-state'

// Well inside Ollama's 10-minute keep-alive, so an active student never meets a cold model,
// while an idle one releases the model's memory after the keep-alive expires.
export const modelWarmupRefreshMs = 4 * 60_000

export function useModelWarmup(enabled: boolean, model?: LocalModel, settings?: ModelRuntimeSettings) {
  const target = useRef({ model, settings })
  target.current = { model, settings }
  const ready = enabled && model?.engine === 'ollama' && model.status === 'ready' && Boolean(settings)
  useEffect(() => {
    if (!ready) return
    let warmedAt = 0
    const warm = () => {
      const { model, settings } = target.current
      if (!model || !settings || Date.now() - warmedAt < modelWarmupRefreshMs) return
      warmedAt = Date.now()
      void Promise.resolve().then(() => window.tokensmith?.warmModel(model, settings)).catch(() => { warmedAt = 0 })
    }
    const onVisible = () => { if (document.visibilityState === 'visible') warm() }
    warm()
    window.addEventListener('pointerdown', warm)
    window.addEventListener('keydown', warm)
    window.addEventListener('focus', warm)
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.removeEventListener('pointerdown', warm)
      window.removeEventListener('keydown', warm)
      window.removeEventListener('focus', warm)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [ready, model?.id, settings?.contextLength, settings?.contextLengthMode])
}
