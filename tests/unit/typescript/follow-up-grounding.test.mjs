import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { filterSuggestedQuestions } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { runRemoteStudyEngine } = requireTranspiledTs('src/main/engine/remote-chat-service.ts')
const { defaultSuggestedFollowUpPrompt, legacySuggestionPrompts } = requireTranspiledTs('src/shared/model-defaults.ts')

test('the default follow-up prompt anchors on the study material, and the old default is upgraded', () => {
  assert.match(defaultSuggestedFollowUpPrompt, /appear in the study material excerpts/)
  assert.doesNotMatch(defaultSuggestedFollowUpPrompt, /subjects the answer has already introduced/)
  assert.ok(legacySuggestionPrompts.some((prompt) => /subjects the answer has already introduced/.test(prompt)))
})

test('the filter drops near-duplicates of kept suggestions and code names the student has not met', () => {
  const kept = filterSuggestedQuestions(
    [
      'Why does it use LRU instead of FIFO in this case?',   // repeats the student's question
      'How does temporal locality justify LRU?',
      'How does temporal locality justify the LRU policy?',  // near-duplicate of the previous one
      'What does pageMap.erase() do during eviction?',       // identifier the student has not seen
      'When does LRU perform poorly?',
      'What happens on a buffer miss?',
      'How is the lruList ordered?'                          // named in the answer, so allowed
    ],
    ['Why does it use LRU instead of FIFO?'],
    4,
    'Why does it use LRU instead of FIFO?\nLRU keeps recently used pages because of temporal locality. The lruList holds pages in recency order.'
  )
  assert.deepEqual(kept, [
    'How does temporal locality justify LRU?',
    'When does LRU perform poorly?',
    'What happens on a buffer miss?',
    'How is the lruList ordered?'
  ])
})

test('follow-ups run on the smallest installed local chat model, skipping embedding models', async () => {
  const original = globalThis.fetch
  const chatModels = []
  try {
    globalThis.fetch = async (url, options) => {
      if (/\/api\/tags$/.test(url)) {
        return { ok: true, json: async () => ({ models: [
          { name: 'nomic-embed-text:latest', size: 274_000_000 },
          { name: 'llama3.2:3b', size: 2_000_000_000 },
          { name: 'llama3:latest', size: 4_700_000_000 }
        ] }) }
      }
      if (/\/api\/show$/.test(url)) return { ok: false, status: 404, text: async () => '' }
      const body = JSON.parse(options.body)
      chatModels.push(body.model)
      return { ok: true, json: async () => ({ done_reason: 'stop', message: {
        content: chatModels.length === 1 ? 'Eviction removes the least recently used page.' : '["Why is the least recently used page chosen?"]'
      } }) }
    }
    const result = await runOllamaStudyEngine({
      prompt: 'What does eviction do?', messages: [], materials: [], settings: {},
      model: { engine: 'ollama', ollamaModelName: 'llama3', contextLength: 8192 },
      modelSettings: { contextLength: 8192, maxLength: 64 },
      applicationSettings: { suggestionMode: 'on', followUpSuggestionCount: 4 },
      retrievedSources: [{ title: 'Buffer manager', excerpt: 'The evict method removes the least recently used page.' }]
    })
    assert.deepEqual(chatModels, ['llama3', 'llama3.2:3b'])
    assert.deepEqual(result.followUpSuggestions, ['Why is the least recently used page chosen?'])
  } finally { globalThis.fetch = original }
})

test('a cloud answer gets no follow-ups, and no error, when no local chat model is installed', async () => {
  const original = globalThis.fetch
  const calls = []
  try {
    globalThis.fetch = async (url) => {
      calls.push(String(url))
      if (/\/api\/tags$/.test(url)) return { ok: true, json: async () => ({ models: [{ name: 'nomic-embed-text:latest', size: 274_000_000 }] }) }
      return { ok: true, json: async () => ({ choices: [{ message: { content: 'Eviction removes the least recently used page.' } }] }) }
    }
    const result = await runRemoteStudyEngine({
      prompt: 'What does eviction do?', messages: [], materials: [], settings: {},
      model: { engine: 'remote', name: 'Gemini', baseUrl: 'https://example.test/v1', apiKey: 'key', remoteModelName: 'gemini-2.5-flash' },
      modelSettings: { maxLength: 512, temperature: 0.2 },
      applicationSettings: { suggestionMode: 'on', followUpSuggestionCount: 4 },
      retrievedSources: [{ title: 'Buffer manager', excerpt: 'The evict method removes the least recently used page.' }]
    })
    assert.equal(result.text, 'Eviction removes the least recently used page.')
    assert.deepEqual(result.followUpSuggestions, [])
    assert.equal(result.followUpError, undefined)
    assert.equal(calls.filter((url) => url.startsWith('https://example.test')).length, 1)
  } finally { globalThis.fetch = original }
})
