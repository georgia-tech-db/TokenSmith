import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { prepareStudyChatMessages, shouldGenerateFollowUps } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { runRemoteStudyEngine } = requireTranspiledTs('src/main/engine/remote-chat-service.ts')
const source = { title: 'Notes', materialId: 'c1', documentId: 1, chunkId: 'ch.1', excerpt: 'The ancestor path lets insertion propagate a split to its parent.' }
const base = {
  practiceTask: 'feedback', prompt: '{"answer":"My answer"}', messages: [], materials: [], settings: {},
  retrievedSources: [source], applicationSettings: { suggestionMode: 'on', explanationDepthEnabled: true, explanationDepth: 'detailed' },
  model: { engine: 'ollama', name: 'Test', ollamaModelName: 'test', contextLength: 8192 },
  modelSettings: { contextLength: 8192, maxLength: 1024, reasoningMode: 'off', thinking: false, systemMessage: 'Custom chat persona.' }
}
const raw = '{"checks":[],"nextStep":"Source 1 is quoted verbatim.","improvement":"","questionIssue":""}'

async function mocked(handler, run) {
  const saved = globalThis.fetch
  globalThis.fetch = async (url, options) => ({ ok: true, json: async () => handler(String(url), JSON.parse(options.body)) })
  try { return await run() } finally { globalThis.fetch = saved }
}

test('practice has a dedicated grading prompt, evidence IDs, and no chat persona or history', () => {
  const prepared = prepareStudyChatMessages({ ...base, messages: [{ role: 'assistant', text: 'Unrelated chat history.' }] })
  assert.match(prepared.messages[0].content, /formative feedback/)
  assert.doesNotMatch(prepared.messages[0].content, /Custom chat persona/)
  assert.match(prepared.messages[1].content, /Evidence ID: c1:1:ch.1/)
  assert.doesNotMatch(JSON.stringify(prepared.messages), /Unrelated chat history/)
  assert.deepEqual(prepared.sources, [source])
  assert.equal(shouldGenerateFollowUps(base), false)
})

test('grading refuses to silently drop oversized saved evidence', () => {
  assert.throws(() => prepareStudyChatMessages({ ...base, retrievedSources: [source,
    { ...source, chunkId: 'large', excerpt: 'A long source sentence. '.repeat(10000) }] }), /full grading evidence/)
})

test('Ollama practice uses structured output and preserves JSON, with no follow-up call', async () => {
  const calls = []
  const result = await mocked((url, body) => {
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    calls.push(body)
    if (body.options.num_predict === 1) return { prompt_eval_count: 2000 }
    assert.equal(body.think, false)
    assert.equal(body.options.temperature, 0)
    assert.ok(body.format.required.includes('checks'))
    return { done_reason: 'stop', message: { content: raw } }
  }, () => runOllamaStudyEngine(base))
  assert.equal(calls.length, 2) // Token preflight, then the feedback completion.
  assert.equal(result.text, raw)
  assert.deepEqual(result.sources, [source])
})

test('Ollama grading refuses actual-token overflow instead of pruning rubric evidence', async () => {
  let completions = 0
  await assert.rejects(mocked((url, body) => {
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    if (body.options.num_predict !== 1) completions++
    return { prompt_eval_count: 8000 }
  }, () => runOllamaStudyEngine({ ...base, retrievedSources: [source, { ...source, chunkId: 'ch.2' }] })), /context|fit|room/i)
  assert.equal(completions, 0)
})

test('remote practice keeps raw structured output and reports truncated completions', async () => {
  const request = { ...base, model: { ...base.model, engine: 'remote', remoteModelName: 'test',
    baseUrl: 'https://provider.example/v1', apiKey: 'test' } }
  for (const finish_reason of ['stop', 'length']) {
    let calls = 0
    const run = () => mocked((_url, body) => {
      calls++
      assert.match(body.messages[0].content, /formative feedback/)
      assert.equal(body.temperature, 0)
      return { choices: [{ finish_reason, message: { content: raw } }] }
    }, () => runRemoteStudyEngine(request))
    if (finish_reason === 'length') await assert.rejects(run(), /limit|complete|truncat/i)
    else assert.equal((await run()).text, raw)
    assert.equal(calls, 1)
  }
})
