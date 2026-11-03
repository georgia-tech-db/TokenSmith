import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { runOllamaStudyEngine, unloadOllamaModels, warmOllamaModel } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const model = { id: 'ollama:test', engine: 'ollama', name: 'Test', ollamaModelName: 'test', status: 'ready', contextLength: 131072 }
const auto = { contextLengthMode: 'auto', contextLength: 8192, maxLength: 1024, reasoningMode: 'off', thinking: false, systemMessage: '' }

async function mocked(handler, run) {
  const saved = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url, options = {}) => {
    const call = { url: String(url), body: options.body ? JSON.parse(options.body) : undefined }
    calls.push(call)
    return { ok: true, json: async () => handler(call) }
  }
  try { await unloadOllamaModels(); calls.length = 0; await run(); return calls } finally { globalThis.fetch = saved }
}

test('warming a model that is already loaded keeps its context size, so Ollama does not reload it', async () => {
  const calls = await mocked(({ url }) => url.endsWith('/ps') ? { models: [{ name: 'test:latest', context_length: 16384 }] } : {},
    () => warmOllamaModel(model, auto))
  const warm = calls.find(call => call.url.endsWith('/chat'))
  assert.deepEqual(warm.body, { model: 'test', messages: [], keep_alive: '10m', options: { num_ctx: 16384 } })
})

test('warming an unloaded model uses the same automatic context size as practice requests', async () => {
  const calls = await mocked(({ url }) => url.endsWith('/ps') ? { models: [] } : {}, () => warmOllamaModel(model, auto))
  assert.deepEqual(calls.find(call => call.url.endsWith('/chat')).body.options, { num_ctx: 8192 })
  const manual = await mocked(({ url }) => url.endsWith('/ps') ? { models: [] } : {},
    () => warmOllamaModel(model, { ...auto, contextLengthMode: 'manual', contextLength: 12288 }))
  assert.deepEqual(manual.find(call => call.url.endsWith('/chat')).body.options, { num_ctx: 12288 })
})

test('model requests ask Ollama to keep the model loaded for ten minutes', async () => {
  const request = { practiceTask: 'true_false_question', prompt: '{"questionNumber":1}', messages: [], materials: [], settings: {},
    retrievedSources: [{ title: 'Notes', materialId: 'c1', documentId: 1, chunkId: 'ch.1', excerpt: 'Ancestor paths propagate splits.' }],
    applicationSettings: { suggestionMode: 'off' }, model, modelSettings: auto }
  const calls = await mocked(({ url, body }) => {
    if (url.endsWith('/show')) return { model_info: { 'test.context_length': 131072 } }
    return body.options.num_predict === 1 ? { prompt_eval_count: 900 } : { done_reason: 'stop', message: { content: '{}' } }
  }, () => runOllamaStudyEngine(request))
  const chats = calls.filter(call => call.url.endsWith('/chat'))
  assert.equal(chats.length, 2)
  assert.ok(chats.every(call => call.body.keep_alive === '10m'))
})

test('quitting unloads every model the app used, once', async () => {
  const calls = await mocked(({ url }) => url.endsWith('/ps') ? { models: [] } : {}, async () => {
    await warmOllamaModel(model, auto)
    await warmOllamaModel({ ...model, ollamaBaseUrl: 'http://127.0.0.1:11500', ollamaModelName: 'other' }, auto)
    await warmOllamaModel(model, auto)
    await unloadOllamaModels()
  })
  const unloads = calls.filter(call => call.url.endsWith('/generate'))
  assert.deepEqual(unloads.map(call => [call.url, call.body]).sort(), [
    ['http://127.0.0.1:11434/api/generate', { model: 'test', keep_alive: 0 }],
    ['http://127.0.0.1:11500/api/generate', { model: 'other', keep_alive: 0 }]
  ])
  const again = await mocked(() => ({}), () => unloadOllamaModels())
  assert.equal(again.filter(call => call.url.endsWith('/generate')).length, 0)
})

test('an unreachable Ollama does not delay quitting', async () => {
  const saved = globalThis.fetch
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ models: [] }) })
  try { await warmOllamaModel(model, auto) } finally { globalThis.fetch = saved }
  globalThis.fetch = async () => { throw new Error('connection refused') }
  try { await assert.doesNotReject(unloadOllamaModels()) } finally { globalThis.fetch = saved }
})
