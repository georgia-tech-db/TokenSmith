import { AsyncLocalStorage } from 'node:async_hooks'
import { randomUUID } from 'node:crypto'
import { setTimeout as delay } from 'node:timers/promises'
import type { ModelLoadingEvent } from '../../shared/model-loading'

interface ProgressScope { requestId: string; emit: (event: ModelLoadingEvent) => void; signal?: AbortSignal }
const progressScope = new AsyncLocalStorage<ProgressScope>()

export function withModelLoadingProgress<T>(requestId: string | undefined, emit: ProgressScope['emit'],
  signal: AbortSignal | undefined, run: () => Promise<T>): Promise<T> {
  return requestId ? progressScope.run({ requestId, emit, signal }, run) : run()
}

interface ModelRuntime { baseUrl: string; modelName: string; contextLength?: number; role?: 'embedder' | 'generator' }
const tagged = (name: string) => name.split('/').at(-1)?.includes(':') ? name : `${name}:latest`

/** Observe the existing request; do not preload, unload, or run extra inference. */
export async function observeModelLoading<T>(runtime: ModelRuntime, run: () => Promise<T>,
  fetcher: typeof fetch = fetch, pollMs = 500): Promise<T> {
  const scope = progressScope.getStore()
  if (!scope) return run()
  scope.signal?.throwIfAborted()
  const base = runtime.baseUrl.replace(/\/+$/, '').replace(/\/api$/, '')
  const stop = new AbortController()
  const signal = scope.signal ? AbortSignal.any([scope.signal, stop.signal]) : stop.signal
  async function resident(): Promise<boolean | undefined> {
    try {
      const response = await fetcher(`${base}/api/ps`, { signal: AbortSignal.any([signal, AbortSignal.timeout(1000)]) })
      if (!response.ok) return undefined
      const data = await response.json() as { models?: Array<{ name?: string; model?: string; context_length?: number; expires_at?: string }> }
      if (!Array.isArray(data.models)) return undefined
      return data.models.some(model => [model.name, model.model].some(name => name && tagged(name) === tagged(runtime.modelName)) &&
        (!model.expires_at || Date.parse(model.expires_at) > Date.now()) &&
        (!runtime.contextLength || !model.context_length || model.context_length === runtime.contextLength))
    } catch { return undefined } // Optional telemetry must not make inference fail.
  }
  let started = 0, loading = false, ended = false
  const identity = { requestId: scope.requestId, loadId: randomUUID(), contextLength: runtime.contextLength,
    model: { id: `ollama:${runtime.modelName}`, name: runtime.modelName, engine: 'ollama' as const,
      ollamaModelName: runtime.modelName, ollamaBaseUrl: base, role: runtime.role } }
  const emit = (event: ModelLoadingEvent) => { try { scope.emit(event) } catch { /* Window may have closed. */ } }
  const finish = (success: boolean) => {
    if (!loading || ended) return
    ended = true
    emit({ ...identity, state: success ? 'ready' : 'failed', durationMs: performance.now() - started })
  }
  // Observe in parallel: optional status requests must not delay inference or
  // returning its result. Late responses cannot emit after this request ends.
  void (async () => {
    if (await resident() !== false || signal.aborted) return
    started = performance.now()
    loading = true
    emit({ ...identity, state: 'loading' })
    while (!signal.aborted && !ended) {
      await delay(pollMs, undefined, { signal })
      const loaded = await resident()
      if (!signal.aborted && loaded) finish(true)
    }
  })().catch(() => undefined)
  try {
    const result = await run()
    scope.signal?.throwIfAborted()
    return result
  } finally {
    // Only learn startup time when the observer confirmed residency. A completed
    // operation alone does not prove the model loaded (for example, FTS results).
    finish(false)
    stop.abort()
  }
}
