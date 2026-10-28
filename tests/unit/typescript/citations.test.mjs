import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { withCitationMarkers } = requireTranspiledTs('src/renderer/src/citations.ts')
const { gradeClaims } = requireTranspiledTs('src/shared/confidence.ts')

function claim(text, sourceIndex) {
  return { text, verdict: sourceIndex === undefined ? 'unsupported' : 'supported', entailment: 0.9, contradiction: 0, sourceIndex }
}

test('withCitationMarkers cites each traced sentence after its punctuation', () => {
  const text = 'Pinning prevents eviction. BuzzDB uses GPUs. Unpinned pages can be evicted.'
  const marked = withCitationMarkers(text, [
    claim('Pinning prevents eviction.', 0),
    claim('BuzzDB uses GPUs.'),
    claim('Unpinned pages can be evicted.', 2)
  ])

  assert.equal(marked, 'Pinning prevents eviction. [1](#cite-0) BuzzDB uses GPUs. Unpinned pages can be evicted. [3](#cite-2)')
})

test('withCitationMarkers finds sentences through Markdown the engine stripped', () => {
  const text = '- **Pinning** keeps a page *in memory*.\n- It is released on unpin.'
  const marked = withCitationMarkers(text, [claim('Pinning keeps a page in memory.', 1)])

  assert.equal(marked, '- **Pinning** keeps a page *in memory*. [2](#cite-0)\n- It is released on unpin.')
})

test('withCitationMarkers leaves text alone when a claim is not found', () => {
  const text = 'Pinning prevents eviction.'

  assert.equal(withCitationMarkers(text, [claim('Something the answer never said.', 0)]), text)
})

test('gradeClaims names the source that entails a supported claim', () => {
  const [supported, unsupported] = gradeClaims([
    { text: 'a b c d', passages: [{ index: 0, entailment: 0.2, contradiction: 0 }, { index: 1, entailment: 0.93, contradiction: 0 }] },
    { text: 'e f g h', passages: [{ index: 0, entailment: 0.1, contradiction: 0 }, { index: 1, entailment: 0.2, contradiction: 0 }] }
  ])

  assert.equal(supported.sourceIndex, 1)
  assert.equal(unsupported.sourceIndex, undefined)
})
