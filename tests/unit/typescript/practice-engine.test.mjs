import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { prepareStudyChatMessages, shouldGenerateFollowUps } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { runRemoteStudyEngine } = requireTranspiledTs('src/main/engine/remote-chat-service.ts')
const source = { title: 'Notes', materialId: 'c1', documentId: 1, chunkId: 'ch.1', excerpt: 'The ancestor path lets insertion propagate a split to its parent.' }
const base = {
  practiceTask: 'true_false_question', prompt: '{"questionNumber":1,"targetAnswer":true}', messages: [], materials: [], settings: {},
  retrievedSources: [source], applicationSettings: { suggestionMode: 'on', explanationDepthEnabled: true, explanationDepth: 'detailed' },
  model: { engine: 'ollama', name: 'Test', ollamaModelName: 'test', contextLength: 8192 },
  modelSettings: { contextLength: 8192, maxLength: 1024, reasoningMode: 'off', thinking: false, systemMessage: 'Custom chat persona.' }
}
const raw = '{"question":"Ancestor paths let insertion propagate splits to parents.","correctAnswer":true,"explanation":"Source 1 is quoted verbatim."}'

async function mocked(handler, run) {
  const saved = globalThis.fetch
  globalThis.fetch = async (url, options) => ({ ok: true, json: async () => handler(String(url), JSON.parse(options.body)) })
  try { return await run() } finally { globalThis.fetch = saved }
}

test('practice has a dedicated question prompt, evidence IDs, and no chat persona or history', () => {
  const prepared = prepareStudyChatMessages({ ...base, messages: [{ role: 'assistant', text: 'Unrelated chat history.' }] })
  assert.match(prepared.messages[0].content, /True\/False practice statement/)
  assert.doesNotMatch(prepared.messages[0].content, /Custom chat persona/)
  assert.match(prepared.messages[1].content, /Evidence ID: c1:1:ch.1/)
  assert.doesNotMatch(JSON.stringify(prepared.messages), /Unrelated chat history/)
  assert.deepEqual(prepared.sources, [source])
  assert.equal(shouldGenerateFollowUps(base), false)
})

test('question generation can use complete passages that fit without requiring all supplied evidence', () => {
  const prepared = prepareStudyChatMessages({ ...base, retrievedSources: [source,
    { ...source, chunkId: 'large', excerpt: 'A long source sentence. '.repeat(10000) }] })
  assert.deepEqual(prepared.sources, [source])
})

test('Ollama practice uses structured output and preserves JSON, with no follow-up call', async () => {
  const calls = []
  const result = await mocked((url, body) => {
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    calls.push(body)
    if (body.options.num_predict === 1) return { prompt_eval_count: 2000 }
    assert.equal(body.think, false)
    assert.equal(body.options.temperature, 0.3)
    assert.ok(body.format.required.includes('correctAnswer'))
    return { done_reason: 'stop', message: { content: raw } }
  }, () => runOllamaStudyEngine(base))
  assert.equal(calls.length, 1) // No token probe or verification completion.
  assert.equal(result.text, raw)
  assert.deepEqual(result.sources, [source])
})

test('Auto practice budgets locally and sends one completion without a token probe', async () => {
  const contexts = []
  const result = await mocked((url, body) => {
    if (url.endsWith('/show')) return { model_info: { 'test.context_length': 262144 } }
    contexts.push(body.options.num_ctx)
    assert.equal(body.think, false)
    assert.notEqual(body.options.num_predict, 1)
    assert.equal(body.options.num_predict, 1024)
    assert.ok(body.format.required.includes('correctAnswer'))
    return { done_reason: 'stop', message: { content: raw } }
  }, () => runOllamaStudyEngine({ ...base, model: { ...base.model, ollamaModelName: 'practice-auto-grow' },
    modelSettings: { ...base.modelSettings, contextLengthMode: 'auto', contextLength: 2048 } }))
  assert.deepEqual(contexts, [8192])
  assert.equal(result.text, raw)
  assert.deepEqual(result.sources, [source])
})

test('True/False preparation uses its dedicated schema on Ollama and prompt on remote providers', async () => {
  const request = { ...base, practiceTask: 'true_false_question', prompt: '{"questionNumber":3}' }
  const prepared = prepareStudyChatMessages(request)
  assert.match(prepared.messages[0].content, /True\/False practice statement/)
  assert.match(prepared.messages[0].content, /targetAnswer/)
  assert.equal(shouldGenerateFollowUps(request), false)
  const binary = '{"correctAnswer":false}'
  const local = await mocked((url, body) => {
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    if (body.options.num_predict === 1) return { prompt_eval_count: 2000 }
    assert.equal(body.format.properties.correctAnswer.type, 'boolean')
    assert.equal(body.think, false)
    assert.equal(body.options.temperature, 0.3)
    return { done_reason: 'stop', message: { content: binary } }
  }, () => runOllamaStudyEngine(request))
  assert.equal(local.text, binary)
  const remote = await mocked((_url, body) => {
    assert.match(body.messages[0].content, /True\/False practice statement/)
    assert.match(body.messages[0].content, /"correctAnswer"/)
    assert.equal(body.temperature, 0.3)
    return { choices: [{ finish_reason: 'stop', message: { content: binary } }] }
  }, () => runRemoteStudyEngine({ ...request, model: { ...request.model, engine: 'remote',
    baseUrl: 'https://provider.example/v1', remoteModelName: 'test', apiKey: 'test' } }))
  assert.equal(remote.text, binary)
})

test('Ollama chapter generation refuses actual-token overflow instead of pruning chapter evidence', async () => {
  let completions = 0, preflights = 0
  await assert.rejects(mocked((url, body) => {
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    if (body.options.num_predict !== 1) completions++
    else preflights++
    return { prompt_eval_count: 8000 }
  }, () => runOllamaStudyEngine({ ...base, practiceTask: 'chapter_topics', retrievedSources: [source, { ...source, chunkId: 'ch.2' }] })), /context|fit|room/i)
  assert.equal(completions, 0)
  assert.equal(preflights, 1)
})

test('remote practice keeps raw structured output and reports truncated completions', async () => {
  const request = { ...base, model: { ...base.model, engine: 'remote', remoteModelName: 'test',
    baseUrl: 'https://provider.example/v1', apiKey: 'test' } }
  for (const finish_reason of ['stop', 'length']) {
    let calls = 0
    const run = () => mocked((_url, body) => {
      calls++
      assert.match(body.messages[0].content, /True\/False practice statement/)
      assert.equal(body.temperature, 0.3)
      return { choices: [{ finish_reason, message: { content: raw } }] }
    }, () => runRemoteStudyEngine(request))
    if (finish_reason === 'length') await assert.rejects(run(), /limit|complete|truncat/i)
    else assert.equal((await run()).text, raw)
    assert.equal(calls, 1)
  }
})


test('chapter review generation refuses incomplete context instead of generating from selected fragments', () => {
  assert.throws(() => prepareStudyChatMessages({ ...base, practiceTask: 'chapter_topics', retrievedSources: [source,
    { ...source, chunkId: 'large', excerpt: 'A long source sentence. '.repeat(10000) }] }), /full chapter text/)
})

test('chapter generation retains provider temperatures, structured output and no follow-up calls', async () => {
  const request = { ...base, practiceTask: 'chapter_topics', practiceAllowedIds: ['c1:1:ch.1'], prompt: '{"chapter":"Trees"}' }
  const reply = '{"topics":[{"title":"Splits","aliases":[],"objectives":["Follow the path","Update parents"],"sourceKeys":["c1:1:ch.1"]}]}'
  assert.equal(shouldGenerateFollowUps(request), false)
  let localCalls = 0
  const local = await mocked((url, body) => {
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    localCalls++
    if (body.options.num_predict === 1) return { prompt_eval_count: 2000 }
    assert.equal(body.think, false)
    assert.equal(body.options.temperature, 0)
    assert.deepEqual(body.format.required, ['topics'])
    assert.deepEqual(body.format.properties.topics.items.properties.sourceKeys.items.enum, request.practiceAllowedIds)
    return { done_reason: 'stop', message: { content: reply } }
  }, () => runOllamaStudyEngine(request))
  assert.equal(localCalls, 2)
  assert.equal(local.text, reply)
  assert.deepEqual(local.sources, [source])
  let remoteCalls = 0
  const remote = await mocked((_url, body) => {
    remoteCalls++
    assert.match(body.messages[0].content, /Review Terms for ONE textbook chapter/)
    assert.equal(body.temperature, 0.3)
    return { choices: [{ finish_reason: 'stop', message: { content: reply } }] }
  }, () => runRemoteStudyEngine({ ...request, model: { ...request.model, engine: 'remote',
    baseUrl: 'https://provider.example/v1', remoteModelName: 'test', apiKey: 'test' } }))
  assert.equal(remoteCalls, 1)
  assert.equal(remote.text, reply)
  assert.deepEqual(remote.sources, [source])
})
