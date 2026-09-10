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

test('buildContextualRetrievalContext uses the previous question and carried sources without copying the old answer', () => {
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
  assert.match(context.answerPrompt, /Previous question: What is pinning a page\?/)
  assert.match(context.answerPrompt, /Current question: why is that needed\?/)
  assert.doesNotMatch(context.answerPrompt, /fixes a page/)
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
