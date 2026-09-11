import assert from 'node:assert/strict'
import test from 'node:test'
import { buzzdbSource } from '../../helpers/buzzdb-fixture.mjs'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  buildContextualRetrievalContext,
  chooseRetrievalContext,
  mergeChatSources,
  profileQuestion,
  shouldTryContextualRetrieval
} = requireTranspiledTs('src/shared/chat-context.ts')

const pinningSource = buzzdbSource('ch05.084')
const lruScanSource = buzzdbSource('ch06.012')
const lruTouchSource = buzzdbSource('ch06.041')
const lruEvictSource = buzzdbSource('ch06.042')
const lruTraceTableSource = buzzdbSource('ch06.043')
const lruTraceStepsSource = buzzdbSource('ch06.044')
const lruHotColdSource = buzzdbSource('ch06.045')
const twoQIntroSource = buzzdbSource('ch06.057')
const twoQMechanismSource = buzzdbSource('ch06.059')
const twoQScanSource = buzzdbSource('ch06.064')
const hashRangeComparisonSource = buzzdbSource('ch08.058')
const hashBplusComparisonSource = buzzdbSource('ch08.059')
const bplusBenchmarkSource = buzzdbSource('ch09.013')
const sequenceSetSource = buzzdbSource('ch09.030')
const sequenceSetScanSource = buzzdbSource('ch09.033')
const bplusTreeSource = buzzdbSource('ch09.019')
const accidentalBetterSource = {
  title: 'BuzzDBBook / buzzdb-book.tokensmith',
  documentTitle: 'buzzdb-book.tokensmith',
  collectionName: 'BuzzDBBook',
  path: '/tmp/buzzdb-book.tokensmith.md',
  sectionHeader: 'Measuring a Filter Power: Selectivity',
  excerpt: 'Here, the index scan is dramatically better.',
  context: 'Here, the index scan is dramatically better because it uses fewer page reads.'
}

function completedTurn(userText, assistantText, sources) {
  return [
    {
      id: 'u1',
      role: 'user',
      text: userText
    },
    {
      id: 'a1',
      role: 'assistant',
      text: assistantText,
      sources
    }
  ]
}

test('profileQuestion scores underspecified prompts by anchor terms, not by canned follow-up phrases', () => {
  const underspecified = profileQuestion('why is that needed?')
  assert.deepEqual(underspecified.anchorTerms, [])
  assert.equal(underspecified.specificity, 0)

  const namedShortQuestion = profileQuestion('What is CC BY-SA?')
  assert.ok(namedShortQuestion.anchorTerms.includes('by-sa'))
  assert.ok(namedShortQuestion.specificity >= 1)
})

test('profileQuestion tolerates small reference typos without turning grammar words into references', () => {
  const typoProfile = profileQuestion('Elaborate on taht')
  const comparisonProfile = profileQuestion('Why is a B+ tree better than a binary search tree?')

  assert.equal(typoProfile.hasExternalReference, true)
  assert.deepEqual(typoProfile.anchorTerms, [])
  assert.equal(comparisonProfile.hasExternalReference, false)
})

test('buildContextualRetrievalContext uses the previous answer only for contextual prompt resolution', () => {
  const context = buildContextualRetrievalContext(
    'why is that needed?',
    completedTurn(
      'What is pinning a page?',
      'Pinning fixes a page in a buffer frame while it is being used.',
      [pinningSource]
    ),
    { turnCount: 1, carriedSourceLimit: 1 }
  )

  assert.ok(context)
  assert.equal(context.mode, 'contextual')
  assert.match(context.query, /pinning/)
  assert.match(context.query, /page/)
  assert.doesNotMatch(context.query, /buzzdbbook/i)
  assert.doesNotMatch(context.query, /tokensmith/i)
  assert.doesNotMatch(context.query, /fixes a page/)
  assert.match(context.answerPrompt, /Previous question: What is pinning a page\?/)
  assert.match(context.answerPrompt, /Previous answer: Pinning fixes a page/)
  assert.match(context.answerPrompt, /Current question: why is that needed\?/)
  assert.deepEqual(context.carriedSources, [pinningSource])
})

test('shouldTryContextualRetrieval skips contextual search when standalone sources ground a fresh question', () => {
  const messages = completedTurn(
    'Why does LRU fail for large sequential scans?',
    'LRU can pollute the buffer pool with single-use scan pages.',
    [lruScanSource]
  )

  assert.equal(
    shouldTryContextualRetrieval('What exactly is a B+ tree and how is it different from a binary search tree?', messages, [bplusTreeSource]),
    false
  )
})

test('chooseRetrievalContext keeps a fresh topic switch standalone even when old context exists', () => {
  const messages = completedTurn(
    'Why does LRU fail for large sequential scans?',
    'LRU can pollute the buffer pool with single-use scan pages.',
    [lruScanSource]
  )
  const contextualContext = buildContextualRetrievalContext(
    'What exactly is a B+ tree and how is it different from a binary search tree?',
    messages,
    { turnCount: 1, carriedSourceLimit: 1 }
  )

  const choice = chooseRetrievalContext(
    'What exactly is a B+ tree and how is it different from a binary search tree?',
    [bplusTreeSource],
    contextualContext,
    [lruScanSource],
    2
  )

  assert.equal(choice.mode, 'standalone')
  assert.equal(choice.query, 'What exactly is a B+ tree and how is it different from a binary search tree?')
  assert.deepEqual(choice.sources, [bplusTreeSource])
})

test('chooseRetrievalContext selects contextual sources for an underspecified follow-up', () => {
  const messages = completedTurn(
    'What is pinning a page?',
    'Pinning fixes a page in a buffer frame while it is being used.',
    [pinningSource]
  )
  const contextualContext = buildContextualRetrievalContext(
    'why is that needed?',
    messages,
    { turnCount: 1, carriedSourceLimit: 1 }
  )

  const choice = chooseRetrievalContext(
    'why is that needed?',
    [],
    contextualContext,
    [pinningSource],
    2
  )

  assert.equal(choice.mode, 'contextual')
  assert.match(choice.query, /pinning/)
  assert.match(choice.answerPrompt, /Previous question: What is pinning a page\?/)
  assert.deepEqual(choice.sources, [pinningSource])
})

test('chooseRetrievalContext selects contextual sources for a referring follow-up with a weak standalone match', () => {
  const messages = completedTurn(
    'What exactly is a B+ tree and how is it different from a binary search tree?',
    'A B+ tree is a wide, shallow search tree designed for disk pages.',
    [bplusTreeSource]
  )
  const contextualContext = buildContextualRetrievalContext(
    'why is it better',
    messages,
    { turnCount: 1, carriedSourceLimit: 1 }
  )

  assert.ok(contextualContext)
  assert.doesNotMatch(contextualContext.query, /buzzdbbook/i)
  assert.doesNotMatch(contextualContext.query, /tokensmith/i)

  const choice = chooseRetrievalContext(
    'why is it better',
    [accidentalBetterSource],
    contextualContext,
    [bplusTreeSource],
    2
  )

  assert.equal(choice.standaloneQuality, 1)
  assert.equal(choice.mode, 'contextual')
  assert.match(choice.query, /btree|b\+/)
  assert.match(choice.answerPrompt, /Previous question: What exactly is a B\+ tree/)
  assert.deepEqual(choice.sources, [bplusTreeSource])
})

test('contextual follow-ups preserve the previous focus instead of drifting to lexical neighbors', () => {
  const previousQuestion = 'How does the LRU policy ensure that "hot" pages stay in the buffer, while "cold" pages are evicted?'
  const messages = completedTurn(
    previousQuestion,
    'LRU keeps recently touched pages near the front of its list and evicts from the back.',
    [twoQMechanismSource, lruHotColdSource]
  )
  const contextualContext = buildContextualRetrievalContext(
    'Elaborate on that',
    messages,
    { turnCount: 1, carriedSourceLimit: 2 }
  )

  assert.ok(contextualContext)
  assert.equal(contextualContext.isContextualFollowUp, true)
  assert.deepEqual(contextualContext.carriedSources, [lruHotColdSource])

  const choice = chooseRetrievalContext(
    'Elaborate on that',
    [],
    contextualContext,
    [
      twoQIntroSource,
      twoQMechanismSource,
      twoQScanSource,
      lruTouchSource,
      lruEvictSource,
      lruTraceTableSource,
      lruTraceStepsSource
    ],
    4
  )
  const selectedChunkIds = choice.sources.map((source) => source.chunkId)
  const selectedTwoQCount = selectedChunkIds.filter((chunkId) =>
    ['ch06.057', 'ch06.059', 'ch06.064'].includes(chunkId)
  ).length

  assert.equal(choice.mode, 'contextual')
  assert.equal(selectedChunkIds[0], 'ch06.045')
  assert.ok(selectedChunkIds.includes('ch06.044'))
  assert.ok(selectedTwoQCount <= 1)
})

test('contextual follow-ups include current-question evidence when asking for a contrast', () => {
  const messages = completedTurn(
    'What exact part of the B+ tree makes the range scan fast?',
    'The sequence set connects the leaf pages, so a range scan can continue through adjacent leaves.',
    [sequenceSetSource, sequenceSetScanSource]
  )
  const contextualContext = buildContextualRetrievalContext(
    'Does that mean we should always prefer it over hashing?',
    messages,
    { turnCount: 1, carriedSourceLimit: 2 }
  )

  assert.ok(contextualContext)

  const choice = chooseRetrievalContext(
    'Does that mean we should always prefer it over hashing?',
    [],
    contextualContext,
    [
      bplusBenchmarkSource,
      hashRangeComparisonSource,
      hashBplusComparisonSource
    ],
    4
  )
  const selectedChunkIds = choice.sources.map((source) => source.chunkId)
  const comparisonIndex = selectedChunkIds.findIndex((chunkId) => ['ch08.058', 'ch08.059'].includes(chunkId))
  const extraFocusIndex = selectedChunkIds.indexOf('ch09.033')

  assert.equal(choice.mode, 'contextual')
  assert.ok(selectedChunkIds.includes('ch09.030'))
  assert.ok(comparisonIndex >= 0)
  assert.ok(extraFocusIndex < 0 || comparisonIndex < extraFocusIndex)
})

test('contextual scoring uses corrected search terms from retrieval', () => {
  const messages = completedTurn(
    'What exact part of the B+ tree makes the range scan fast?',
    'The sequence set connects the leaf pages, so a range scan can continue through adjacent leaves.',
    [sequenceSetSource, sequenceSetScanSource]
  )
  const contextualContext = buildContextualRetrievalContext(
    'Does that mean we should always prefer it over hasing?',
    messages,
    { turnCount: 1, carriedSourceLimit: 2 }
  )
  const correctedHashSource = {
    ...hashBplusComparisonSource,
    queryTerms: ['hashing', 'b+', 'tree', 'range'],
    keywordTerms: ['hashing', 'range']
  }

  assert.ok(contextualContext)

  const choice = chooseRetrievalContext(
    'Does that mean we should always prefer it over hasing?',
    [],
    contextualContext,
    [
      sequenceSetScanSource,
      correctedHashSource
    ],
    3
  )
  const selectedChunkIds = choice.sources.map((source) => source.chunkId)

  assert.equal(choice.mode, 'contextual')
  assert.equal(selectedChunkIds[0], correctedHashSource.chunkId)
  assert.ok(selectedChunkIds.includes(sequenceSetSource.chunkId))
})

test('shouldTryContextualRetrieval keeps a short specific standalone question when its source is grounded', () => {
  const messages = completedTurn(
    'What exactly is a B+ tree and how is it different from a binary search tree?',
    'A B+ tree is a wide, shallow search tree designed for disk pages.',
    [bplusTreeSource]
  )

  assert.equal(
    shouldTryContextualRetrieval('What is LRU?', messages, [lruScanSource]),
    false
  )
})

test('mergeChatSources keeps carried context while deduping retrieved results', () => {
  assert.deepEqual(
    mergeChatSources([pinningSource], [pinningSource, bplusTreeSource], 4),
    [pinningSource, bplusTreeSource]
  )
})

test('mergeChatSources keeps matching chunk labels from different collections distinct', () => {
  const firstCollectionSource = {
    ...pinningSource,
    materialId: '1',
    sourceId: '1|/first/book.tokensmith.md|ch05.084',
    path: '/first/book.tokensmith.md'
  }
  const secondCollectionSource = {
    ...pinningSource,
    materialId: '2',
    sourceId: '2|/second/book.tokensmith.md|ch05.084',
    path: '/second/book.tokensmith.md'
  }

  assert.deepEqual(
    mergeChatSources([firstCollectionSource], [secondCollectionSource], 4),
    [firstCollectionSource, secondCollectionSource]
  )
})
