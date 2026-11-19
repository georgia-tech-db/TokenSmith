import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { readCurrentCollection } = requireTranspiledTs('src/main/collections/collection-read.ts')

test('catalog read retries an indexing race and returns only the current documents', async () => {
  const collection = { indexRevision: 'before' }
  let calls = 0
  const result = await readCurrentCollection(collection, () => {}, async () => {
    if (++calls === 1) { collection.indexRevision = 'after'; return ['stale'] }
    return ['current']
  }, 3)
  assert.deepEqual(result, ['current'])
  assert.equal(calls, 2)
})
test('closing the library discards an in-flight result without retrying', async () => {
  const collection = { indexRevision: 'before' }
  let closed = false, calls = 0
  await assert.rejects(readCurrentCollection(collection, () => { if (closed) throw Error('closed') }, async () => {
    calls++; closed = true; return ['stale']
  }, 3), /closed/)
  assert.equal(calls, 1)
})
test('continuous changes have a bounded retry and other requests remain single attempt', async () => {
  for (const attempts of [1, 3]) {
    const collection = { indexRevision: '0' }; let calls = 0
    await assert.rejects(readCurrentCollection(collection, () => {}, async () => {
      collection.indexRevision = String(++calls); return []
    }, attempts), /library is updating/)
    assert.equal(calls, attempts)
  }
})
