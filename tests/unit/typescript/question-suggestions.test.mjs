import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { generateOllamaStudyQuestionSuggestions, runOllamaStudyEngine } =
  requireTranspiledTs('src/main/engine/ollama-service.ts')
const { legacySuggestionPrompts } = requireTranspiledTs('src/shared/model-defaults.ts')
const request = {
  prompt: 'What is a range query?', messages: [], materials: [], settings: {},
  model: { engine: 'ollama', ollamaModelName: 'test-model', contextLength: 8192 },
  modelSettings: { contextLength: 8192, maxLength: 64, suggestedFollowUpPrompt: legacySuggestionPrompts.at(-1) },
  applicationSettings: { suggestionMode: 'on', followUpSuggestionCount: 4 },
  retrievedSources: [{ title: 'Indexing', excerpt: 'Ordered indexes support range queries.' }]
}

test('initial suggestions use their own prompt and a complete structured response', async () => {
  const original = globalThis.fetch
  try {
    let calls = 0
    globalThis.fetch = async (_url, options) => {
      calls += 1
      const body = JSON.parse(options.body)
      assert.equal(body.format.type, 'array')
      assert.equal(body.format.maxItems, 4)
      assert.equal(body.options.num_predict, 384)
      assert.equal(body.think, false)
      assert.match(body.messages.at(-1).content, /short opening questions/)
      assert.doesNotMatch(body.messages.at(-1).content, /after this answer|latest answer/)
      return { ok: true, json: async () => ({ done_reason: 'stop', message: { content: '["Why do indexes speed up queries?"]' } }) }
    }
    assert.deepEqual((await generateOllamaStudyQuestionSuggestions(request)).suggestions, ['Why do indexes speed up queries?'])
    assert.equal(calls, 1)
    globalThis.fetch = async () => ({ ok: true, json: async () => ({ done_reason: 'length', message: { content: '["Unfinished' } }) })
    await assert.rejects(generateOllamaStudyQuestionSuggestions(request), /output limit/)
  } finally { globalThis.fetch = original }
})

test('follow-ups use the actual new answer and do not drop complete questions solely for exceeding 18 words', async () => {
  const original = globalThis.fetch
  const question = 'When does the complexity difference between a range scan and a full scan become practically noticeable when dealing with millions of records?'
  try {
    let calls = 0
    globalThis.fetch = async (_url, options) => {
      const body = JSON.parse(options.body)
      if (++calls === 2) {
        assert.equal(body.format.type, 'array')
        assert.equal(body.options.num_predict, 384)
        assert.match(body.messages.at(-1).content, /Latest answer:\nRange queries return records within an interval/)
        assert.match(body.messages.at(-1).content, /after the latest answer/)
        assert.doesNotMatch(body.messages.at(-1).content, /short opening questions/)
      }
      return { ok: true, json: async () => ({ done_reason: 'stop', message: {
        content: calls === 1 ? 'Range queries return records within an interval.' : JSON.stringify([question])
      } }) }
    }
    const result = await runOllamaStudyEngine(request)
    assert.equal(calls, 2)
    assert.deepEqual(result.followUpSuggestions, [question])
  } finally { globalThis.fetch = original }
})
