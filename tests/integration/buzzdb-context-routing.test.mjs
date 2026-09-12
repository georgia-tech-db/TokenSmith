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
const lruElaborationSources = [
  lruTouchSource,
  lruEvictSource,
  lruTraceTableSource,
  lruTraceStepsSource,
  lruHotColdSource,
  twoQIntroSource,
  twoQMechanismSource,
  twoQScanSource
]
const hashPreferenceSources = [
  bplusBenchmarkSource,
  hashRangeComparisonSource,
  hashBplusComparisonSource
]
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

function packedUserContent(modelMessages) {
  assert.equal(modelMessages.at(-1).role, 'user')
  return modelMessages.at(-1).content
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
  const packedPrompt = packedUserContent(modelMessages)
  assert.match(packedPrompt, /It's a search tree, but it's not a binary tree/)
  assert.match(packedPrompt, /wide, shallow tree/)
  assert.doesNotMatch(packedPrompt, /Phase 3/)
  assert.doesNotMatch(packedPrompt, /Compose the Tuple/)
}

{
  const priorMessages = completedTurn(
    'What is pinning a page?',
    'Pinning keeps a page from being evicted while a component is using it.',
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
  const packedPrompt = packedUserContent(modelMessages)
  assert.match(packedPrompt, /Previous question: What is pinning a page\?/)
  assert.doesNotMatch(packedPrompt, /Previous answer/)
  assert.doesNotMatch(packedPrompt, /Pinning keeps a page/)
  assert.match(packedPrompt, /Current question: why is that needed\?/)
  assert.match(packedPrompt, /silent data corruption or a catastrophic crash/)
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
  const packedPrompt = packedUserContent(modelMessages)
  assert.match(packedPrompt, /Previous question: What exactly is a B\+ tree/)
  assert.match(packedPrompt, /Current question: why is it better/)
  assert.match(packedPrompt, /wide, shallow tree/)
  assert.doesNotMatch(packedPrompt, /index scan is dramatically better/)
}

{
  const previousQuestion = 'How does the LRU policy ensure that "hot" pages stay in the buffer, while "cold" pages are evicted?'
  const priorMessages = completedTurn(
    previousQuestion,
    'LRU keeps recently touched pages near the front of its list and evicts from the back.',
    [twoQMechanismSource, lruHotColdSource]
  )
  const calls = []
  const prompt = 'Elaborate on that'
  const choice = await routeRetrievalContext(prompt, priorMessages, {
    limit: 4,
    turnCount: 1,
    carriedSourceLimit: 2,
    search: (query, limit = 4) => searchBuzzdbFixture(query, lruElaborationSources, calls, limit)
  })
  const modelMessages = formatMessages(prompt, priorMessages, choice)
  const selectedChunkIds = choice.sources.map((source) => source.chunkId)
  const selectedTwoQCount = selectedChunkIds.filter((chunkId) =>
    ['ch06.057', 'ch06.059', 'ch06.064'].includes(chunkId)
  ).length

  assert.equal(choice.mode, 'contextual')
  assert.equal(calls.length, 2)
  assert.equal(calls[0], prompt)
  assert.match(calls[1], /lru/)
  assert.equal(calls[1].includes('2q'), false)
  assert.equal(calls[1].includes('fifo'), false)
  assert.equal(selectedChunkIds[0], 'ch06.045')
  assert.ok(selectedChunkIds.includes('ch06.044'))
  assert.ok(selectedTwoQCount <= 1)
  const packedPrompt = packedUserContent(modelMessages)
  assert.match(packedPrompt, /Current question: Elaborate on that/)
  assert.match(packedPrompt, /This trace shows the dynamism of the LRU policy/)
  assert.doesNotMatch(packedPrompt, /The 2Q policy's first choice/)
}

{
  const priorMessages = completedTurn(
    'What exact part of the B+ tree makes the range scan fast?',
    'The sequence set connects the leaf pages, so a range scan can continue through adjacent leaves.',
    [sequenceSetSource, sequenceSetScanSource]
  )
  const calls = []
  const prompt = 'Does that mean we should always prefer it over hashing?'
  const choice = await routeRetrievalContext(prompt, priorMessages, {
    limit: 4,
    turnCount: 1,
    carriedSourceLimit: 2,
    search: (query, limit = 4) => searchBuzzdbFixture(query, hashPreferenceSources, calls, limit)
  })
  const modelMessages = formatMessages(prompt, priorMessages, choice)
  const selectedChunkIds = choice.sources.map((source) => source.chunkId)
  const comparisonIndex = selectedChunkIds.findIndex((chunkId) => ['ch08.058', 'ch08.059'].includes(chunkId))
  const extraFocusIndex = selectedChunkIds.indexOf('ch09.033')

  assert.equal(choice.mode, 'contextual')
  assert.equal(calls.length, 2)
  assert.match(calls[1], /hashing/)
  assert.ok(selectedChunkIds.includes('ch09.030'))
  assert.ok(comparisonIndex >= 0)
  assert.ok(extraFocusIndex < 0 || comparisonIndex < extraFocusIndex)
  assert.match(packedUserContent(modelMessages), /Current question: Does that mean we should always prefer it over hashing\?/)
}

console.log('BuzzDB context routing integration test passed.')
