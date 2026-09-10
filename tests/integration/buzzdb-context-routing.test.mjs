import assert from 'node:assert/strict'
import { buzzdbSource, searchBuzzdbFixture } from '../helpers/buzzdb-fixture.mjs'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const {
  routeRetrievalContext
} = requireTranspiledTs('src/shared/chat-context.ts')
const {
  studyChatMessages
} = requireTranspiledTs('src/main/engine/study-chat-format.ts')

const pinningSource = buzzdbSource('ch05.084')
const lruScanSource = buzzdbSource('ch06.012')
const phaseThreeSource = buzzdbSource('ch02.046')
const bplusTreeSource = buzzdbSource('ch09.019')
const accidentalBetterSource = {
  title: 'BuzzDBBook / buzzdb-book.tokensmith',
  documentTitle: 'buzzdb-book.tokensmith',
  collectionName: 'BuzzDBBook',
  path: '/tmp/buzzdb-book.tokensmith.md',
  sectionHeader: 'Measuring a Filter Power: Selectivity',
  excerpt: 'Here, the index scan is dramatically better.',
  context: 'Here, the index scan is dramatically better because it uses fewer page reads.',
  chunkId: 'wrong-better-source'
}
const buzzdbSources = [pinningSource, lruScanSource, phaseThreeSource, bplusTreeSource, accidentalBetterSource]
const model = {
  id: 'ollama:llama3',
  name: 'Ollama llama3',
  engine: 'ollama',
  role: 'generator',
  status: 'ready',
  source: 'ollama',
  ollamaModelName: 'llama3',
  addedAt: '2026-09-10T00:00:00.000Z'
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

function formatMessages(prompt, messages, choice) {
  return studyChatMessages({
    prompt,
    answerPrompt: choice.answerPrompt,
    retrievalQuery: choice.query,
    conversationContextMode: choice.mode,
    messages,
    materials: [],
    model,
    settings: {},
    applicationSettings: {
      suggestionMode: 'off',
      followUpSuggestionCount: 2
    },
    modelSettings: {},
    retrievedSources: choice.sources
  })
}

{
  const priorMessages = completedTurn(
    'What is Phase 3 of the tuple construction process?',
    'Phase 3 composes the tuple from the individual Field objects.',
    [phaseThreeSource]
  )
  const calls = []
  const prompt = 'What exactly is a B+ tree and how is it different from a binary search tree?'
  const choice = await routeRetrievalContext(prompt, priorMessages, {
    limit: 2,
    turnCount: 1,
    carriedSourceLimit: 1,
    search: (query) => searchBuzzdbFixture(query, buzzdbSources, calls, 2)
  })
  const modelMessages = formatMessages(prompt, priorMessages, choice)

  assert.equal(choice.mode, 'standalone')
  assert.deepEqual(calls, [prompt])
  assert.equal(choice.sources[0], bplusTreeSource)
  assert.equal(modelMessages.length, 1)
  assert.match(modelMessages[0].content, /It's a search tree, but it's not a binary tree/)
  assert.match(modelMessages[0].content, /wide, shallow tree/)
  assert.doesNotMatch(modelMessages[0].content, /Phase 3/)
  assert.doesNotMatch(modelMessages[0].content, /Compose the Tuple/)
}

{
  const priorMessages = completedTurn(
    'What is pinning a page?',
    'This stale assistant answer must not be replayed to the model.',
    [pinningSource]
  )
  const calls = []
  const prompt = 'why is that needed?'
  const choice = await routeRetrievalContext(prompt, priorMessages, {
    limit: 2,
    turnCount: 1,
    carriedSourceLimit: 1,
    search: (query) => searchBuzzdbFixture(query, buzzdbSources, calls, 2)
  })
  const modelMessages = formatMessages(prompt, priorMessages, choice)

  assert.equal(choice.mode, 'contextual')
  assert.equal(calls.length, 2)
  assert.equal(calls[0], prompt)
  assert.match(calls[1], /pinning/)
  assert.equal(choice.sources[0], pinningSource)
  assert.equal(modelMessages.length, 1)
  assert.match(modelMessages[0].content, /Previous question: What is pinning a page\?/)
  assert.match(modelMessages[0].content, /Current question: why is that needed\?/)
  assert.match(modelMessages[0].content, /silent data corruption or a catastrophic crash/)
  assert.doesNotMatch(modelMessages[0].content, /stale assistant answer/)
}

{
  const priorMessages = completedTurn(
    'What exactly is a B+ tree and how is it different from a binary search tree?',
    'This old answer text should not be replayed.',
    [bplusTreeSource]
  )
  const calls = []
  const prompt = 'why is it better'
  const choice = await routeRetrievalContext(prompt, priorMessages, {
    limit: 2,
    turnCount: 1,
    carriedSourceLimit: 1,
    search: (query) => searchBuzzdbFixture(query, buzzdbSources, calls, 2)
  })
  const modelMessages = formatMessages(prompt, priorMessages, choice)

  assert.equal(choice.mode, 'contextual')
  assert.equal(calls.length, 2)
  assert.equal(calls[0], prompt)
  assert.equal(calls[1].includes('buzzdbbook'), false)
  assert.equal(calls[1].includes('tokensmith'), false)
  assert.match(calls[1], /btree|b\+/)
  assert.equal(choice.sources[0], bplusTreeSource)
  assert.equal(modelMessages.length, 1)
  assert.match(modelMessages[0].content, /Previous question: What exactly is a B\+ tree/)
  assert.match(modelMessages[0].content, /Current question: why is it better/)
  assert.match(modelMessages[0].content, /wide, shallow tree/)
  assert.doesNotMatch(modelMessages[0].content, /index scan is dramatically better/)
}

console.log('BuzzDB context routing integration test passed.')
