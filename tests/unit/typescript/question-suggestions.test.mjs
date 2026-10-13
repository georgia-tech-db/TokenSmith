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

test('Advanced Math uses raw code and a stable prompt with existing publication and cancellation', async () => {
  const original = globalThis.fetch
  const controller = new AbortController()
  const published = []
  let calls = 0
  let trace
  let systemPrompt
  try {
    globalThis.fetch = async (_url, options) => {
      calls++
      assert.ok(options.signal instanceof AbortSignal)
      const body = JSON.parse(options.body)
      assert.equal(body.model, request.model.ollamaModelName)
      assert.equal(body.options.temperature, 0)
      assert.equal(body.think, false)
      assert.equal(body.format, undefined)
      if (calls === 1) systemPrompt = body.messages[0].content
      else {
        assert.equal(body.messages[0].content, systemPrompt)
        assert.match(body.messages.at(-1).content, /javascript_interpret result:\n42/)
      }
      return { ok: true, json: async () => ({ done_reason: 'stop', message: { content: calls === 1
        ? 'I will calculate the product and subtract the specified amount.\n<javascript_interpret>\nlet total=0; for(let i=0;i<6;i++){total+=7;} console.log(total);\n</javascript_interpret>'
        : 'Multiply 6 by 7, then subtract 0: the result is **42**.' } }) }
    }
    const result = await runOllamaStudyEngine({ ...request, prompt: 'Calculate the product of 6 and 7, then subtract 0.', retrievedSources: [],
      advancedMathEnabled: true, advancedMathPlan: { route: 'math' } }, {
      signal: controller.signal, onAnswer: (answer, followUps) => published.push({ answer, followUps }),
      onJavascriptInterpretTrace: value => { trace = value }
    })
    assert.equal(calls, 2)
    assert.equal(trace.attempts[0].execution.result, '42')
    assert.equal(published.length, 1)
    assert.equal(published[0].followUps, false)
    assert.deepEqual(published[0].answer, result)
    assert.match(result.text, /42/)
  } finally { globalThis.fetch = original }
})

test('Advanced Math source calculations use the existing packed retrieval context and return sources', async () => {
  const original = globalThis.fetch
  let calls = 0
  try {
    globalThis.fetch = async (_url, options) => {
      const body = JSON.parse(options.body)
      calls++
      const context = body.messages.filter(message => message.role === 'user').map(message => message.content).join('\n')
      assert.match(context, /Capacity is 12 per batch; there are 5 batches/)
      assert.doesNotMatch(context, /Hidden value/)
      return { ok: true, json: async () => ({ done_reason: 'stop', message: { content: calls === 1
        ? 'I will multiply capacity by the number of batches.\n<javascript_interpret>\nconsole.log("Total capacity: "+12*5);\n</javascript_interpret>'
        : 'Multiply 12 units per batch by 5 batches: total capacity is 60 units.' } }) }
    }
    const result = await runOllamaStudyEngine({ ...request, prompt: 'Using the textbook, calculate total capacity.',
      advancedMathEnabled: true, advancedMathPlan: { route: 'math' },
      retrievedSources: [{ title: 'Capacity', excerpt: 'Hidden value: 999.', context: 'Capacity is 12 per batch; there are 5 batches.' }] })
    assert.equal(calls, 2)
    assert.match(result.text, /60 units/)
    assert.equal(result.sources.length, 1)
  } finally { globalThis.fetch = original }
})

test('Advanced Math sends line-specific errors back with the same prompt and no JSON schema', async () => {
  const original = globalThis.fetch
  let calls = 0
  let systemPrompt
  try {
    globalThis.fetch = async (_url, options) => {
      const body = JSON.parse(options.body)
      calls++
      assert.equal(body.format, undefined)
      if (calls === 1) systemPrompt = body.messages[0].content
      else assert.equal(body.messages[0].content, systemPrompt)
      if (calls === 2) {
        assert.match(body.messages.at(-1).content, /ReferenceError at line 2, column \d+/)
        assert.match(body.messages.at(-1).content, /2: console.log\(value\+missing\);/)
      }
      const content = calls === 1
        ? '<javascript_interpret>\nconst value=6*7;\nconsole.log(value+missing);\n</javascript_interpret>'
        : calls === 2 ? '<javascript_interpret>\nconsole.log(6*7-0);\n</javascript_interpret>' : 'Multiply 6 by 7 and subtract 0 to get 42.'
      return { ok: true, json: async () => ({ done_reason: 'stop', message: { content } }) }
    }
    const result = await runOllamaStudyEngine({ ...request, prompt: 'Calculate the product of 6 and 7, then subtract 0.',
      retrievedSources: [], advancedMathEnabled: true, advancedMathPlan: { route: 'math' } })
    assert.equal(calls, 3)
    assert.match(result.text, /42/)
  } finally { globalThis.fetch = original }
})
