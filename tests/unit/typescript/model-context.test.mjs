import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { effectiveContextLength, migratedContextMode, localContextForTokens, normalizeTokenLimit, mergeModelContextMetadata } = requireTranspiledTs('src/shared/model-context.ts')
const { modelAwareRuntimeSettings, sourceContextBudgetForRequest } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { modelAwareRetrievalLimit } = requireTranspiledTs('src/shared/retrieval-budget.ts')

test('Auto uses reported local and cloud limits without 8K or 32K caps', () => {
  for (const engine of ['ollama', 'remote']) {
    for (const contextLength of [4096, 32768, 131072, 1048576]) {
      assert.equal(effectiveContextLength({ engine, contextLength }, { contextLength: 2048, contextLengthMode: 'auto' }), contextLength)
    }
  }
  assert.equal(effectiveContextLength({}, { contextLengthMode: 'auto' }), 8192)
})

test('Custom sizes are honored, including below 8K, but cannot exceed the model limit', () => {
  assert.equal(effectiveContextLength({ contextLength: 131072 }, { contextLengthMode: 'manual', contextLength: 2048 }), 2048)
  assert.equal(effectiveContextLength({ contextLength: 4096 }, { contextLengthMode: 'manual', contextLength: 16384 }), 4096)
  assert.equal(effectiveContextLength({}, { contextLengthMode: 'manual', contextLength: 65536 }), 65536)
})

test('legacy defaults migrate to Auto while custom preferences survive', () => {
  for (const contextLength of [undefined, 2048, 8192]) assert.equal(migratedContextMode({ contextLength }), 'auto')
  for (const contextLength of [4096, 16384, 32768, 131072]) assert.equal(migratedContextMode({ contextLength }), 'manual')
  assert.equal(migratedContextMode({ contextLengthMode: 'manual', contextLength: 8192 }), 'manual')
})

test('invalid metadata does not become a context size', () => {
  for (const value of [null, true, '', 0, -1, NaN, Infinity, 123.4, Number.MAX_SAFE_INTEGER + 1, {}, 'many']) {
    assert.equal(normalizeTokenLimit(value), undefined)
  }
  assert.equal(normalizeTokenLimit('1048576'), 1048576)
})

test('local allocation grows to fit a request without allocating the entire model window', () => {
  assert.equal(localContextForTokens(11000, 262144), 16384)
  assert.equal(localContextForTokens(20000, 262144), 32768)
  assert.equal(localContextForTokens(20000, 12000), 12000)
  assert.equal(localContextForTokens(3000, 262144), 8192)
})

test('separate provider input and output limits both constrain source packing', () => {
  const model = { engine: 'remote', inputTokenLimit: 65536, maxOutputTokens: 2048 }
  const modelSettings = modelAwareRuntimeSettings({ model, modelSettings: { contextLengthMode: 'auto', contextLength: 8192, maxLength: 4096 } })
  assert.equal(modelSettings.maxLength, 2048)
  assert.equal(modelSettings.contextLength, 65536 + 2048)
  const budget = sourceContextBudgetForRequest({ model, modelSettings, prompt: 'Explain', messages: [], materials: [], retrievedSources: [] })
  assert.ok(budget.fixedPromptTokens + budget.sourceBudgetTokens + budget.safetyMarginTokens <= 65536)
  assert.equal(budget.answerReserveTokens, 2048)
})

test('fresh input-only metadata replaces stale clipped total limits; unavailable metadata preserves the last known limit', () => {
  const saved = {engine:'remote',contextLength:32768}
  const updated = mergeModelContextMetadata(saved,{inputTokenLimit:1048576,maxOutputTokens:65536})
  assert.equal(updated.contextLength,undefined)
  assert.equal(effectiveContextLength(updated,{contextLengthMode:'auto',maxLength:4096}),1048576+4096)
  assert.deepEqual(mergeModelContextMetadata(saved,{}),saved)
})

test('retrieval and generation agree on manual context and the full output allowance', () => {
  const model = { engine: 'remote', contextLength: 131072 }
  assert.equal(modelAwareRetrievalLimit(4, model, { contextLengthMode: 'manual', contextLength: 8192, maxLength: 4096 }), 4)
  assert.equal(modelAwareRetrievalLimit(4, model, { contextLengthMode: 'auto', contextLength: 8192, maxLength: 4096 }), 8)
  assert.equal(modelAwareRetrievalLimit(4, {}, { contextLengthMode: 'auto', maxLength: 4096 }), 8)
})
