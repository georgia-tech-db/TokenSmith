import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { estimateWait, recordTiming, readTimingSamples, timingKeys, waitDisplay } = requireTranspiledTs('src/shared/generation-estimate.ts')
const context = { kind: 'answer', model: { id: 'gemma', name: 'Gemma', engine: 'ollama', ollamaModelName: 'gemma4:26b' }, settings: { contextLength: 8192, maxLength: 1536 }, inputChars: 5000 }

test('the first actual run updates a broad initial estimate; repeated runs replace it', () => {
  const initial = estimateWait(context, [])
  let history = recordTiming([], context, 120_000)
  const first = estimateWait(context, history)
  assert.ok(first.expectedMs > initial.expectedMs)
  assert.ok(first.lowerMs < 120_000 && first.upperMs > 120_000)
  for (let i = 0; i < 5; i++) history = recordTiming(history, context, 120_000)
  assert.equal(estimateWait(context, history).expectedMs, 120_000)
  const restored = readTimingSamples(JSON.parse(JSON.stringify(history)))
  assert.deepEqual(estimateWait(context, restored), estimateWait(context, history))
})
test('timing separates models, endpoints, task kinds and important runtime settings', () => {
  const history = recordTiming([], context, 120_000)
  for (const changed of [
    { ...context, kind: 'starter' },
    { ...context, model: { ...context.model, ollamaModelName: 'gemma4:12b' } },
    { ...context, model: { ...context.model, ollamaBaseUrl: 'http://other-machine:11434' } },
    { ...context, settings: { ...context.settings, thinking: true } },
    { ...context, settings: { ...context.settings, reasoningMode: 'auto' } },
    { ...context, settings: { ...context.settings, contextLength: 16384 } },
    { ...context, settings: { ...context.settings, maxLength: 4096 } }
  ]) assert.equal(estimateWait(changed, history).samples, 0)
  const modes = ['auto', 'off', 'on'].map(reasoningMode => timingKeys({ ...context, settings: { ...context.settings, reasoningMode } }))
  assert.equal(new Set(modes.map(({ family }) => family)).size, 3)
  assert.equal(new Set(modes.map(({ key }) => key)).size, 3)
  const key = JSON.stringify(timingKeys({ ...context, model: { ...context.model, ollamaBaseUrl: 'https://secret:password@example.com/?key=private' } }))
  assert.doesNotMatch(key, /secret|password|private|example/)
})
test('recent speed changes influence estimates without discarding legitimate slow runs', () => {
  let history = []
  for (let i = 0; i < 8; i++) history = recordTiming(history, context, 30_000)
  const before = estimateWait(context, history)
  history = recordTiming(history, context, 240_000)
  assert.ok(estimateWait(context, history).expectedMs > before.expectedMs)
  assert.ok(estimateWait(context, history).upperMs > before.upperMs)
})
test('overdue runs become indeterminate and never claim completion or negative remaining time', () => {
  const estimate = { expectedMs: 40_000, lowerMs: 20_000, upperMs: 60_000, samples: 3 }
  for (const elapsed of [0, 19_000, 40_000, 59_999]) {
    const display = waitDisplay(estimate, elapsed)
    assert.ok(display.fraction < 1)
    assert.doesNotMatch(display.text, /-\d|\b0s left/)
  }
  assert.equal(waitDisplay(estimate, 60_000).fraction, undefined)
  assert.match(waitDisplay(estimate, 70_000).text, /70s elapsed/)
})
test('invalid, future and stale history is ignored and storage is bounded', () => {
  assert.deepEqual(readTimingSamples({}), [])
  assert.deepEqual(readTimingSamples([null, {}, { key:'k', family:'f', ms:-1, at:Date.now() }]), [])
  assert.deepEqual(readTimingSamples([{ key:'k', family:'f', ms:100, at:Date.now()+86400000 }]), [])
  let history = []
  for (let i=0; i<250; i++) history = recordTiming(history, context, 100)
  assert.equal(history.length, 200)
  assert.deepEqual(recordTiming(history, context, NaN), history)
})
