import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { prepareStudyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { runOllamaStudyEngine, resolveOllamaChatQuestion } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const model = { id: 'gemma', name: 'Gemma', engine: 'ollama', ollamaModelName: 'gemma4:e4b', contextLength: 131072 }
const source = (title, context) => ({ title, locator: title, excerpt: context, context })
const request = (overrides = {}) => ({ model, modelSettings: { contextLength: 8192, maxLength: 1536 },
  prompt: 'Trace the updates.', messages: [], materials: [], settings: {}, applicationSettings: { suggestionMode: 'off' },
  retrievedSources: [source('first', 'Initial state is zero.'), source('second', 'Update with max(current, value).')], ...overrides })

async function mockFetch(handler, callback) {
  const saved = globalThis.fetch
  globalThis.fetch = async (url, options) => ({ ok: true, json: async () => handler(JSON.parse(options.body)) })
  try { return await callback() } finally { globalThis.fetch = saved }
}

test('Gemma reserves the full generation allowance and never slices a code unit', () => {
  const complete = '```cpp\nint aggregate = 0;\n```'
  const prepared = prepareStudyChatMessages(request({ modelSettings: { contextLength: 8192, maxLength: 4096 },
    retrievedSources: [source('too large', 'int x = 0;\n'.repeat(3000)), source('complete', complete)] }))
  assert.equal(prepared.budget.answerReserveTokens, 4096)
  assert.equal(prepared.budget.truncatedSourceCount, 0)
  assert.deepEqual(prepared.sources.map(s => s.title), ['complete'])
  assert.ok(prepared.messages.at(-1).content.includes(complete))
  assert.ok(prepared.budget.estimatedPromptTokens + 4096 + prepared.budget.safetyMarginTokens <= 8192)
})

test('an impossible fixed prompt fails before silently dropping the evidence', () => {
  assert.throws(() => prepareStudyChatMessages(request({ prompt: 'word '.repeat(6000) })), /too little room/)
  assert.throws(() => prepareStudyChatMessages(request({ retrievedSources: [source('large', 'x'.repeat(60000))] })), /No complete source/)
})

test('actual preflight overflow removes a whole source, preserves the question, and forwards thinking only to answer generation', async () => {
  const calls = []
  const answer = await mockFetch(body => {
    calls.push(body)
    if (body.options.num_predict === 1) return { prompt_eval_count: calls.length === 1 ? 7500 : 2000, message: { content: 'discard me' }, done_reason: 'length' }
    return { prompt_eval_count: 2000, eval_count: 100, done_reason: 'stop', message: { content: 'The actual result is zero.', thinking: 'internal' } }
  }, () => runOllamaStudyEngine(request({ modelSettings: { contextLength: 8192, maxLength: 1536, thinking: true } })))
  assert.equal(calls.length, 3)
  assert.equal(calls[2].think, true)
  assert.equal(calls[2].options.num_predict, 1536)
  assert.match(calls[2].messages.at(-1).content, /Trace the updates/)
  assert.doesNotMatch(calls[2].messages.at(-1).content, /Update with max|discard me|internal/)
  assert.equal(answer.text, 'The actual result is zero.')
  assert.deepEqual(answer.sources.map(s => s.title), ['first'])
})

test('length-limited plain answers and missing token measurements are not reported as complete', async () => {
  await assert.rejects(mockFetch(body => body.options.num_predict === 1
    ? { prompt_eval_count: 2000 }
    : { done_reason: 'length', message: { content: 'Partial trace' } }, () => runOllamaStudyEngine(request())), /length limit/)
  await assert.rejects(mockFetch(() => ({ message: { content: 'token' } }), () => runOllamaStudyEngine(request())), /could not be verified/)
})

test('question rewriting stays non-thinking even when answer thinking is enabled', async () => {
  const bodies = []
  const result = await mockFetch(body => {
    bodies.push(body)
    return { done_reason: 'stop', message: { content: JSON.stringify({ mode: 'contextual', query: 'Why is the aggregate state zero?', clarification: '' }) } }
  }, () => resolveOllamaChatQuestion(request({ prompt: 'Why?', modelSettings: { contextLength: 8192, maxLength: 4096, thinking: true },
    messages: [{ role: 'user', text: 'What is the aggregate state?' }, { role: 'assistant', text: 'Zero.' }] })))
  assert.equal(result.mode, 'contextual')
  assert.equal(bodies.length, 1)
  assert.equal(bodies[0].think, false)
  assert.equal(bodies[0].options.num_predict, 512)
})
