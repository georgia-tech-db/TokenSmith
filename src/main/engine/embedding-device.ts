import type { LocalModel } from '../../shared/app-state'
import { normalizeEmbeddingGpuEnabled } from '../../shared/embedding-settings'
import { defaultOllamaBaseUrl } from '../../shared/ollama'

// Ollama can reuse a CPU runner for num_gpu=-1. Unload it when returning to auto.
export function createEmbeddingDeviceManager(fetcher: typeof fetch = fetch) {
  const modes = new Map<string, boolean>()
  let pending: Promise<unknown> = Promise.resolve()
  return (models: LocalModel[], enabled?: boolean): Promise<void> => {
    const gpuEnabled = normalizeEmbeddingGpuEnabled(enabled)
    const next = pending.then(async () => {
      for (const model of models) {
        if (model.engine !== 'ollama' || !model.ollamaModelName) continue
        const base = (model.ollamaBaseUrl || defaultOllamaBaseUrl).replace(/\/+$/, '').replace(/\/api$/, '')
        const name = model.ollamaModelName.includes(':') ? model.ollamaModelName : `${model.ollamaModelName}:latest`
        const key = `${base}/${name}`
        const previous = modes.get(key)
        if (previous === gpuEnabled) continue
        let reload = gpuEnabled && previous === false
        if (gpuEnabled && previous === undefined) {
          const response = await fetcher(`${base}/api/ps`, { signal: AbortSignal.timeout(15_000) })
          if (!response.ok) throw new Error(`Could not check the embedding device (HTTP ${response.status}).`)
          const data = await response.json() as { models?: Array<{ name?: string; model?: string; size_vram?: number }> }
          reload = Boolean(data.models?.some(item => (item.name === name || item.model === name) && item.size_vram === 0))
        }
        if (reload) {
          const response = await fetcher(`${base}/api/embed`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: model.ollamaModelName, input: [], keep_alive: 0 }),
            signal: AbortSignal.timeout(30_000)
          })
          if (!response.ok) throw new Error(`Could not switch the embedding device (HTTP ${response.status}).`)
          await response.arrayBuffer()
        }
        modes.set(key, gpuEnabled)
      }
    })
    pending = next.catch(() => undefined)
    return next
  }
}
