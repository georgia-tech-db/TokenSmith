import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { createGenerationProgressController } = requireTranspiledTs('src/shared/generation-progress.ts')
const { estimateWait, timingKeys } = requireTranspiledTs('src/shared/generation-estimate.ts')
const model = { id: 'gemma', name: 'Gemma', engine: 'ollama', ollamaModelName: 'gemma4:e4b', role: 'generator' }
const start = { id: 'question', conversationId: 'practice', kind: 'quiz', label: 'Preparing question', model, onStop() {} }
const loading = (loadId, state, overrides = {}) => ({ requestId: start.id, loadId, state, model, contextLength: 8192, ...overrides })
function fixture() {
  let clock = 0, current, samples = []
  const api = createGenerationProgressController({ now: () => clock, samples, publish: value => { current = value }, save: value => { samples = value } })
  return { api, get current() { return current }, get samples() { return samples }, at(value) { clock = value } }
}

test('cold embedding and generation startup are learned separately from question preparation', () => {
  const f = fixture(), embedder = { ...model, id: 'nomic', ollamaModelName: 'nomic-embed-text', role: 'embedder' }
  f.api.begin(start)
  f.api.modelLoading(loading('embedding', 'loading', { model: embedder, contextLength: undefined }))
  assert.equal(f.current.label, 'Loading model')
  f.at(20_000)
  f.api.modelLoading(loading('embedding', 'ready', { model: embedder, durationMs: 20_000 }))
  assert.equal(f.current.label, 'Preparing question')
  f.api.modelLoading(loading('generator', 'loading'))
  f.at(55_000)
  f.api.modelLoading(loading('generator', 'ready', { durationMs: 35_000 }))
  f.at(62_000)
  f.api.stage(start.id, 'Checking answer')
  assert.equal(f.current.label, 'Checking answer')
  assert.equal(f.current.startedAt, 55_000)
  f.at(65_000)
  f.api.finish(start.id, 'Question ready')
  assert.deepEqual(f.samples.map(sample => sample.ms), [20_000, 35_000, 10_000])
  assert.equal(f.current.complete, true)
  assert.equal(estimateWait(start, f.samples).samples, 1)
  f.api.begin({ ...start, id: 'warm' })
  assert.equal(f.current.label, 'Preparing question')
  assert.equal(f.current.estimate.samples, 1)
  assert.ok(f.current.estimate.expectedMs < 20_000)
  f.at(75_000)
  f.api.finish('warm', 'Question ready')
  assert.equal(estimateWait(start, f.samples).expectedMs, 10_000)
})

test('a loading overlay restores the latest stage and excludes overlapping loads once', () => {
  const f = fixture()
  f.api.begin(start)
  f.at(1000)
  f.api.modelLoading(loading('a', 'loading'))
  f.at(2000)
  f.api.modelLoading(loading('b', 'loading'))
  f.api.stage(start.id, 'Checking answer')
  assert.equal(f.current.label, 'Loading model')
  f.at(4000)
  f.api.modelLoading(loading('a', 'ready'))
  assert.equal(f.current.label, 'Loading model')
  f.at(5000)
  f.api.modelLoading(loading('b', 'ready'))
  assert.equal(f.current.label, 'Checking answer')
  f.at(10_000)
  f.api.finish(start.id, 'Done')
  assert.equal(f.samples.at(-1).ms, 6000)
})

test('canceled or failed startup cannot teach a successful load or revive the job', () => {
  const f = fixture()
  f.api.begin(start)
  f.api.modelLoading(loading('failed', 'loading'))
  f.at(3000)
  f.api.modelLoading(loading('failed', 'failed'))
  assert.deepEqual(f.samples, [])
  f.api.modelLoading(loading('canceled', 'loading'))
  f.api.cancel(start.id)
  f.api.modelLoading(loading('canceled', 'ready', { durationMs: 5000 }))
  assert.equal(f.current, null)
  assert.deepEqual(f.samples, [])
})

test('background loading never replaces foreground status, and duplicate events are ignored', () => {
  const f = fixture()
  f.api.begin({ ...start, id: 'starter', kind: 'starter', label: 'Suggesting questions' })
  f.api.modelLoading(loading('background', 'loading', { requestId: 'starter' }))
  f.api.begin(start)
  assert.equal(f.current.label, 'Preparing question')
  f.api.modelLoading(loading('foreground', 'loading'))
  f.at(5000)
  f.api.modelLoading(loading('foreground', 'loading'))
  assert.equal(f.current.startedAt, 0)
  f.api.modelLoading(loading('foreground', 'ready'))
  f.api.modelLoading(loading('foreground', 'ready'))
  assert.equal(f.samples.length, 1)
  f.api.cancel(start.id)
  assert.equal(f.current.label, 'Loading model')
  assert.equal(f.current.id, 'starter')
})

test('phase changes during loading only subtract startup within the new phase', () => {
  const f = fixture()
  f.api.begin(start)
  f.api.modelLoading(loading('model', 'loading'))
  f.at(5000)
  f.api.next(start.id, 'Suggesting questions', { kind: 'follow-ups' })
  f.at(8000)
  f.api.modelLoading(loading('model', 'ready'))
  f.at(12_000)
  f.api.finish(start.id, 'Done')
  assert.equal(f.samples.at(-1).ms, 4000)
})

test('loading estimates cannot use warm timings and are scoped to model, endpoint and context', () => {
  const context = { kind: 'model-load', model, loadContextLength: 8192 }
  assert.notEqual(timingKeys(context).family, timingKeys(start).family)
  assert.notEqual(timingKeys(context).key, timingKeys({ ...context, loadContextLength: 32768 }).key)
  for (const change of [{ ollamaModelName: 'gemma4:26b' }, { ollamaBaseUrl: 'http://another-machine:11434' }]) {
    assert.notEqual(timingKeys(context).family, timingKeys({ ...context, model: { ...model, ...change } }).family)
  }
})
