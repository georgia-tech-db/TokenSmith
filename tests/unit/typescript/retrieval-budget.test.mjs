import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { modelAwareRetrievalLimit } = requireTranspiledTs('src/shared/retrieval-budget.ts')

test('source candidate budgeting preserves small contexts and expands for larger ones', () => {
  assert.equal(modelAwareRetrievalLimit(4, undefined, { contextLength: 2048 }), 4)
  assert.equal(modelAwareRetrievalLimit(4, { contextLength: 8192 }, { maxLength: 1024 }), 8)
  assert.equal(modelAwareRetrievalLimit(4, { contextLength: 131072 }, { contextLength: 32768 }), 8)
  assert.equal(modelAwareRetrievalLimit(100), 8)
})

test('cross-domain evidence uses the same source budget as the live answer profile', () => {
  const cases = JSON.parse(readFileSync('tests/benchmarks/cross_domain_cases.json', 'utf8'))
  const limit = modelAwareRetrievalLimit(4, { contextLength: 8192 }, { contextLength: 8192, maxLength: 1536 })
  for (const item of cases) assert.equal(item.topK, limit, item.id)
})
