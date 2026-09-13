import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { questionRewriteMessages, parseQuestionRewrite } = requireTranspiledTs('src/main/engine/question-rewrite.ts')
const { prepareRewrittenStudyChat, lastChatExchange } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
const { studyChatMessages, sourceContextBudgetForRequest, answerWithOrderedSources } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { resolveOllamaChatQuestion } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { resolveRemoteChatQuestion } = requireTranspiledTs('src/main/engine/remote-chat-service.ts')

const history = [
  { role: 'user', text: 'How do checkpoints work?' },
  { role: 'assistant', text: 'A checkpoint records progress. Consider records A and B.' }
]
const base = {
  prompt: 'Can you show what happens to A?', messages: history,
  model: { engine: 'ollama', ollamaModelName: 'test-model', contextLength: 8192 },
  modelSettings: { contextLength: 8192, maxLength: 1024, temperature: 0.7 },
  materials: [], settings: {}, retrievedSources: []
}
const source = { title: 'Course', excerpt: 'Records are processed in order.' }

test('rewrite input keeps the current task and prior exchange in a single labeled data message', () => {
  const messages = questionRewriteMessages(base)
  assert.deepEqual(messages.map((message) => message.role), ['system', 'user'])
  assert.deepEqual(JSON.parse(messages[1].content), {
    current_question: base.prompt,
    previous_exchange: { question: history[0].text, answer: history[1].text }
  })
  assert.doesNotMatch(messages[0].content, /LRU|SIMD|pinning|BuzzDB/)
})

test('an incomplete exchange is not replaced with older history', () => {
  assert.equal(lastChatExchange([...history, { role: 'user', text: 'New subject' }]), undefined)
})

test('standalone resolution cannot replace the original question with model paraphrasing', () => {
  assert.equal(parseQuestionRewrite(JSON.stringify({ mode: 'standalone', query: 'different task', clarification: '' }), 'Original?').query, 'Original?')
})

test('invalid, contradictory, and truncated rewrite responses fail without a heuristic fallback', () => {
  for (const text of ['not json', '[]', '{"mode":',
    JSON.stringify({ mode: 'other', query: 'question', clarification: '' }),
    JSON.stringify({ mode: 'contextual', query: '', clarification: '' }),
    JSON.stringify({ mode: 'standalone', query: 'question', clarification: 'Which?' }),
    JSON.stringify({ mode: 'clarify', query: 'guessed subject', clarification: 'Which?' })]) {
    assert.throws(() => parseQuestionRewrite(text, base.prompt))
  }
})

test('first turns skip the rewriter and still retrieve normally', async () => {
  let queries = []
  const result = await prepareRewrittenStudyChat({ ...base, messages: [] }, {
    resolve: () => assert.fail('First turn should not call the model twice'),
    search: async (query) => { queries.push(query); return [source] }
  })
  assert.deepEqual(queries, [base.prompt])
  assert.equal(result.rewriteMs, 0)
  assert.equal(result.request.referenceExchange, undefined)
})

test('contextual rewrites affect retrieval but never replace the student task', async () => {
  const result = await prepareRewrittenStudyChat(base, {
    resolve: async () => ({ mode: 'contextual', query: 'checkpoint record A example', clarification: '' }),
    search: async (query) => { assert.equal(query, 'checkpoint record A example'); return [source] }
  })
  assert.equal(result.request.prompt, base.prompt)
  assert.equal(result.request.answerPrompt, base.prompt)
  assert.equal(result.request.referenceExchange.answer, history[1].text)
  const prompt = studyChatMessages(result.request).at(-1).content
  assert.ok(prompt.endsWith(`Question: ${base.prompt}`))
  assert.match(prompt, /not factual evidence/)
  assert.match(prompt, /records A and B/)
  assert.doesNotMatch(prompt, /checkpoint record A example/)
})

test('a topic change excludes the previous answer and ignores a rewritten standalone query', async () => {
  const result = await prepareRewrittenStudyChat({ ...base, prompt: 'What is isolation?' }, {
    resolve: async () => ({ mode: 'standalone', query: 'wrong checkpoint question', clarification: '' }),
    search: async (query) => { assert.equal(query, 'What is isolation?'); return [source] }
  })
  assert.equal(result.request.referenceExchange, undefined)
  assert.doesNotMatch(studyChatMessages(result.request).at(-1).content, /checkpoint|records A and B/)
})

test('clarification stops before retrieval or answer generation', async () => {
  const result = await prepareRewrittenStudyChat(base, {
    resolve: async () => ({ mode: 'clarify', query: '', clarification: 'Which record do you mean?' }),
    search: () => assert.fail('Must not retrieve a guessed question')
  })
  assert.equal(result.request, undefined)
  assert.equal(result.resolution.clarification, 'Which record do you mean?')
})

test('resolver failures propagate without silently searching the unresolved question', async () => {
  await assert.rejects(prepareRewrittenStudyChat(base, {
    resolve: async () => { throw new Error('Offline') },
    search: () => assert.fail('No fallback search')
  }), /Offline/)
})

test('history consumes context budget and is bounded before source packing', () => {
  const standalone = sourceContextBudgetForRequest(base)
  const contextual = { ...base, conversationContextMode: 'contextual',
    referenceExchange: { question: 'Long prior question '.repeat(1000), answer: 'Long prior answer '.repeat(10000) } }
  const budget = sourceContextBudgetForRequest(contextual)
  assert.ok(budget.fixedPromptTokens > standalone.fixedPromptTokens)
  assert.ok(budget.sourceBudgetTokens < standalone.sourceBudgetTokens)
  assert.ok(budget.fixedPromptTokens < 3000)
  assert.match(studyChatMessages(contextual).at(-1).content, /\[truncated\]/)
})

test('answer normalization preserves paragraphs, list indentation, code and tables', () => {
  const markdown = 'An explanation.\n\n1. Step\n   - Detail\n\n```cpp\nif (ready) {\n  use();\n}\n```\n\n| A | B |\n|---|---|\n| 1 | 2 |'
  assert.equal(answerWithOrderedSources(markdown, []).text, markdown)
})

test('Ollama rewrite uses the selected model, schema, thinking off and a true temperature-zero override', async () => {
  const originalFetch = globalThis.fetch
  try {
    globalThis.fetch = async (_url, options) => {
      const request = JSON.parse(options.body)
      assert.equal(request.model, 'test-model')
      assert.equal(request.think, false)
      assert.equal(request.options.temperature, 0)
      assert.equal(request.options.num_predict, 512)
      assert.equal(request.format.type, 'object')
      return { ok: true, json: async () => ({ done_reason: 'stop', message: { content: JSON.stringify({
        mode: 'contextual', query: 'checkpoint example with record A', clarification: ''
      }) } }) }
    }
    const result = await resolveOllamaChatQuestion(base)
    assert.equal(result.mode, 'contextual')
    globalThis.fetch = async () => ({ ok: true, json: async () => ({ done_reason: 'length', message: { content: '{}' } }) })
    await assert.rejects(resolveOllamaChatQuestion(base), /output limit/)
  } finally { globalThis.fetch = originalFetch }
})

test('remote chat uses the same rewrite contract without a heuristic routing path', async () => {
  const originalFetch = globalThis.fetch
  try {
    globalThis.fetch = async (url, options) => {
      const request = JSON.parse(options.body)
      assert.equal(url, 'https://provider.example/v1/chat/completions')
      assert.equal(request.model, 'remote-test')
      assert.equal(request.temperature, 0)
      assert.equal(request.messages[1].content, questionRewriteMessages(base)[1].content)
      return { ok: true, json: async () => ({ choices: [{ finish_reason: 'stop', message: { content: JSON.stringify({
        mode: 'standalone', query: 'unwanted paraphrase', clarification: ''
      }) } }] }) }
    }
    const result = await resolveRemoteChatQuestion({ ...base, model: {
      engine: 'remote', remoteModelName: 'remote-test', apiKey: 'test-key', baseUrl: 'https://provider.example/v1'
    } })
    assert.equal(result.query, base.prompt)
  } finally { globalThis.fetch = originalFetch }
})
