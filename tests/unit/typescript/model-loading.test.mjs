import assert from 'node:assert/strict'
import test from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { withModelLoadingProgress, observeModelLoading } = requireTranspiledTs('src/main/engine/model-loading.ts')
const runtime = { baseUrl: 'http://localhost:11434/api/', modelName: 'gemma4:e4b', contextLength: 8192 }
const resident = { name: runtime.modelName, context_length: 8192 }
const response = models => ({ ok: true, json: async () => ({ models }) })
function deferred() {
  let resolve, reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

test('a resident model skips loading and runs inference exactly once', async () => {
  const events = [], urls = []
  let calls = 0
  const result = await withModelLoadingProgress('warm', event => events.push(event), undefined,
    () => observeModelLoading(runtime, async () => ++calls, async url => { urls.push(url); return response([resident]) }, 1))
  assert.equal(result, 1)
  assert.equal(calls, 1)
  assert.deepEqual(events, [])
  assert.deepEqual(urls, ['http://localhost:11434/api/ps'])
})

test('cold startup ends when the model becomes resident, before inference completes', async () => {
  const inference = deferred(), ready = deferred(), events = []
  let loaded = false, calls = 0
  const result = withModelLoadingProgress('cold', event => { events.push(event); if (event.state === 'ready') ready.resolve() }, undefined,
    () => observeModelLoading(runtime, () => { calls++; loaded = true; return inference.promise }, async () => response(loaded ? [resident] : []), 1))
  await ready.promise
  assert.deepEqual(events.map(event => event.state), ['loading', 'ready'])
  assert.equal(events[0].requestId, 'cold')
  assert.equal(events[0].loadId, events[1].loadId)
  assert.ok(events[1].durationMs > 0)
  assert.equal(calls, 1)
  inference.resolve('answer')
  assert.equal(await result, 'answer')
  assert.equal(events.length, 2)
})

test('a context change requires loading, while latest-tag aliases match', async () => {
  const events = [], ready = deferred()
  let loaded = false
  await withModelLoadingProgress('resize', event => { events.push(event); if (event.state === 'ready') ready.resolve() }, undefined,
    () => observeModelLoading(runtime, async () => { loaded = true; await ready.promise }, async () => response([{ ...resident, context_length: loaded ? 8192 : 4096 }]), 1))
  assert.deepEqual(events.map(event => event.state), ['loading', 'ready'])
  events.length = 0
  await withModelLoadingProgress('embedder', event => events.push(event), undefined,
    () => observeModelLoading({ ...runtime, modelName: 'nomic-embed-text', contextLength: undefined }, async () => 'ok',
      async () => response([{ name: 'nomic-embed-text:latest' }]), 1))
  assert.deepEqual(events, [])
})

test('unavailable residency telemetry cannot prevent inference', async () => {
  for (const fetcher of [async () => { throw Error('offline') }, async () => ({ ok: false }), async () => ({ ok: true, json: async () => ({}) })]) {
    const events = []
    assert.equal(await withModelLoadingProgress('unknown', event => events.push(event), undefined,
      () => observeModelLoading(runtime, async () => 'answer', fetcher, 1)), 'answer')
    assert.deepEqual(events, [])
  }
})

test('fallback success without a loaded model does not record a successful load', async () => {
  const events = [], loading = deferred()
  assert.equal(await withModelLoadingProgress('fallback', event => { events.push(event); if (event.state === 'loading') loading.resolve() }, undefined,
    () => observeModelLoading(runtime, async () => { await loading.promise; return 'keyword results' }, async () => response([]), 1)), 'keyword results')
  assert.deepEqual(events.map(event => event.state), ['loading', 'failed'])
})

test('cancellation stops polling and never reports a successful load', async () => {
  const events = [], started = deferred(), controller = new AbortController()
  let polls = 0
  const result = withModelLoadingProgress('cancel', event => { events.push(event); if (event.state === 'loading') started.resolve() }, controller.signal,
    () => observeModelLoading(runtime, () => delay(1000, 'answer', { signal: controller.signal }),
      async () => { polls++; return response([]) }, 1))
  await started.promise
  controller.abort()
  await assert.rejects(result, { name: 'AbortError' })
  const after = polls
  await delay(10)
  assert.equal(polls, after)
  assert.deepEqual(events.map(event => event.state), ['loading', 'failed'])
})

test('concurrent requests keep loading events associated with their own request', async () => {
  const events = [[], []]
  await Promise.all(events.map((items, index) => {
    const loading = deferred()
    return withModelLoadingProgress(`request-${index}`, event => { items.push(event); if (event.state === 'loading') loading.resolve() }, undefined,
      () => observeModelLoading(runtime, () => loading.promise, async () => response([]), 1))
  }))
  for (let index = 0; index < events.length; index++) {
    assert.ok(events[index].every(event => event.requestId === `request-${index}`))
    assert.deepEqual(events[index].map(event => event.state), ['loading', 'failed'])
  }
  assert.notEqual(events[0][0].loadId, events[1][0].loadId)
})

test('slow initial telemetry cannot delay inference or publish after it completes', { timeout: 1000 }, async () => {
  const status = deferred(), events = []
  let statusSignal, calls = 0
  const result = await withModelLoadingProgress('slow-status', event => events.push(event), undefined,
    () => observeModelLoading(runtime, async () => { calls++; return 'answer' }, (_url, { signal }) => {
      statusSignal = signal
      return status.promise
    }, 1))
  assert.equal(result, 'answer')
  assert.equal(calls, 1)
  assert.equal(statusSignal.aborted, true)
  status.resolve(response([]))
  await delay(0)
  assert.deepEqual(events, [])
})

test('a stalled loading poll cannot delay the answer or report late readiness', { timeout: 1000 }, async () => {
  const status = deferred(), polling = deferred(), inference = deferred(), events = []
  let polls = 0, statusSignal
  const result = withModelLoadingProgress('slow-poll', event => events.push(event), undefined,
    () => observeModelLoading(runtime, () => inference.promise, async (_url, { signal }) => {
      if (++polls === 1) return response([])
      statusSignal = signal
      polling.resolve()
      return status.promise
    }, 1))
  await polling.promise
  inference.resolve('answer')
  assert.equal(await result, 'answer')
  assert.equal(statusSignal.aborted, true)
  assert.deepEqual(events.map(event => event.state), ['loading', 'failed'])
  status.resolve(response([resident]))
  await delay(0)
  assert.deepEqual(events.map(event => event.state), ['loading', 'failed'])
  assert.equal(polls, 2)
})
