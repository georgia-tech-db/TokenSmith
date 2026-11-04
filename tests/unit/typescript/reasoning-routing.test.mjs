import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { answerReasoningSettings, reasoningMode } = requireTranspiledTs('src/shared/reasoning.ts')
const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
const { parseQuestionRewrite, questionRewriteMessages } = requireTranspiledTs('src/main/engine/question-rewrite.ts')
const { resolveOllamaChatQuestion, runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const base = {
  prompt: 'Apply the procedure to these inputs.', messages: [], materials: [], settings: {},
  model: { engine: 'ollama', name: 'Custom', ollamaModelName: 'custom-model', contextLength: 8192 },
  modelSettings: { reasoningMode: 'auto', contextLength: 8192, maxLength: 1536 },
  applicationSettings: { suggestionMode: 'off' },
  retrievedSources: [{ title: 'Notes', excerpt: 'The procedure and its constraints.' }]
}

async function withFetch(handler, run) {
  const saved = globalThis.fetch
  globalThis.fetch = async (url, options) => ({ ok: true, json: async () => handler(String(url), JSON.parse(options.body)) })
  try { return await run() } finally { globalThis.fetch = saved }
}

test('reasoning is independent of standalone/contextual and has a strict boolean contract', () => {
  for (const mode of ['standalone', 'contextual']) {
    for (const reasoning of [true, false]) {
      assert.equal(parseQuestionRewrite(JSON.stringify({ mode, query: 'Question', clarification: '', reasoning }), 'Question').reasoning, reasoning)
    }
  }
  for (const reasoning of [undefined, 'true', 1, null]) {
    assert.throws(() => parseQuestionRewrite(JSON.stringify({ mode: 'standalone', query: 'Q', clarification: '', reasoning }), 'Q'))
  }
  assert.throws(() => parseQuestionRewrite(JSON.stringify({ mode: 'clarify', query: '', clarification: 'Which?', reasoning: true }), 'Q'))
  assert.equal(parseQuestionRewrite(JSON.stringify({ mode: 'clarify', query: '', clarification: 'Which?', reasoning: false }), 'Q').reasoning, false)
})

test('automatic first turns plan once, preserve the original task, and do not rerank sources', async () => {
  let plans = 0
  let searches = 0
  const result = await prepareRewrittenStudyChat(base, {
    resolve: async () => { plans++; return { mode: 'standalone', query: 'Unwanted paraphrase', clarification: '', reasoning: true } },
    search: async query => { searches++; assert.equal(query, base.prompt); return { sources: base.retrievedSources } }
  })
  assert.equal(plans, 1)
  assert.equal(searches, 1)
  assert.equal(result.request.reasoning, true)
  assert.equal(result.request.answerPrompt, base.prompt)
  assert.equal(result.request.retrievedSources, base.retrievedSources)
  assert.equal(result.request.referenceExchange, undefined)
})

test('reasoning policy overrides a plan without mutating saved settings or previous effective flags', () => {
  const settings = { ...base.modelSettings, thinking: true }
  assert.equal(answerReasoningSettings(settings, false).thinking, false)
  assert.equal(answerReasoningSettings(settings, false).maxLength, 1536)
  assert.equal(answerReasoningSettings(settings, true).maxLength, 4096)
  assert.equal(answerReasoningSettings({ ...settings, maxLength: 5000 }, true).maxLength, 5000)
  assert.equal(answerReasoningSettings({ ...settings, reasoningMode: 'off' }, true).thinking, false)
  assert.equal(answerReasoningSettings({ ...settings, reasoningMode: 'on' }, false).thinking, true)
  assert.equal(reasoningMode({ thinking: true }), 'on')
  assert.equal(settings.maxLength, 1536)
})

test('planner gets the actual preceding exchange; the instruction contains no subject-specific routing', () => {
  const messages = questionRewriteMessages({ ...base, prompt: 'What changes now?', messages: [
    { role: 'user', text: 'Original task.' }, { role: 'assistant', text: 'Current example.' }
  ] })
  assert.equal(JSON.parse(messages[1].content).previous_exchange.answer, 'Current example.')
  assert.match(messages[0].content, /not question length/)
  assert.doesNotMatch(messages[0].content, /B\+|C3|SIMD|BuzzDB|LRU/)
})

test('first-turn automatic planning uses non-thinking structured output on any supported model', async () => {
  const calls = []
  const result = await withFetch((url, body) => {
    calls.push({ url, body })
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    assert.equal(body.think, false)
    assert.equal(body.options.temperature, 0)
    assert.ok(body.format.required.includes('reasoning'))
    return { done_reason: 'stop', message: { content: JSON.stringify({ mode: 'standalone', query: 'Wrong rewrite', clarification: '', reasoning: true }) } }
  }, () => resolveOllamaChatQuestion(base))
  assert.equal(result.query, base.prompt)
  assert.equal(result.reasoning, true)
  assert.equal(calls.filter(call => call.url.endsWith('/chat')).length, 1)
})

test('Auto skips unsupported first-turn planning; On reports unsupported reasoning rather than silently disabling it', async () => {
  for (const metadata of [{ capabilities: ['completion'] }, { capabilities: ['thinking'], thinking: { values: ['low', 'high'] } }]) {
    await withFetch(url => {
      assert.ok(url.endsWith('/show'))
      return metadata
    }, async () => {
      assert.equal((await resolveOllamaChatQuestion(base)).reasoning, false)
      await assert.rejects(runOllamaStudyEngine({ ...base, modelSettings: { ...base.modelSettings, reasoningMode: 'on' } }), /does not advertise/)
    })
  }
})

test('Auto routes reasoning before source packing; probes and final request share settings without extra planning', async () => {
  const calls = []
  await withFetch((url, body) => {
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    calls.push(body)
    assert.equal(body.format, undefined)
    assert.equal(body.think, true)
    return body.options.num_predict === 1 ? { prompt_eval_count: 2000 } : { done_reason: 'stop', message: { content: 'Answer.' } }
  }, () => runOllamaStudyEngine({ ...base, reasoning: true }))
  assert.equal(calls.length, 2)
  assert.equal(calls[1].options.num_predict, 4096)
  assert.deepEqual(calls[0].messages, calls[1].messages)
})

test('direct engine callers in Auto also plan; missing or malformed reasoning is never silently treated as off', async () => {
  let plans = 0
  await withFetch((url, body) => {
    if (url.endsWith('/show')) return { capabilities: ['thinking'] }
    if (body.format) { plans++; return { done_reason: 'stop', message: { content: JSON.stringify({ mode: 'standalone', query: base.prompt, clarification: '', reasoning: false }) } } }
    assert.equal(body.think, false)
    return body.options.num_predict === 1 ? { prompt_eval_count: 2000 } : { done_reason: 'stop', message: { content: 'Answer.' } }
  }, () => runOllamaStudyEngine(base))
  assert.equal(plans, 1)
  await assert.rejects(withFetch(url => url.endsWith('/show') ? { capabilities: ['thinking'] }
    : { done_reason: 'stop', message: { content: '{"mode":"standalone","query":"Q","clarification":""}' } },
  () => runOllamaStudyEngine(base)), /invalid response/)
})

test('reasoning output allowance never silently overflows a small context', async () => {
  await assert.rejects(withFetch(url => {
    assert.ok(url.endsWith('/show'))
    return { capabilities: ['thinking'] }
  }, () => runOllamaStudyEngine({ ...base, reasoning: true,
    model: { ...base.model, contextLength: 2048 }, modelSettings: { ...base.modelSettings, contextLength: 2048 }
  })), /too little room/)
})
