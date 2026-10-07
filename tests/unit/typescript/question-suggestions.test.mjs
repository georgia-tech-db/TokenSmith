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
    let calls = 0, probes = 0
    globalThis.fetch = async (_url, options) => {
      const body = JSON.parse(options.body)
      if (body.options.num_predict === 1) {
        probes++
        return { ok: true, json: async () => ({ prompt_eval_count: 1800 }) }
      }
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
    assert.equal(probes, 1)
    assert.deepEqual(result.followUpSuggestions, [question])
  } finally { globalThis.fetch = original }
})

test('JavaScript reasoning publishes an answer and forwards the cancellation signal', async () => {
  const original = globalThis.fetch
  const controller = new AbortController()
  const published = []
  let trace
  let calls = 0
  try {
    globalThis.fetch = async (_url, options) => {
      calls += 1
      assert.ok(options.signal instanceof AbortSignal)
      const body = JSON.parse(options.body)
      assert.equal(body.think, false)
      if (calls === 1) {
        assert.equal(body.format.type, 'object')
      } else {
        assert.equal(body.format, undefined)
      }
      return { ok: true, json: async () => ({ done_reason: 'stop', message: {
        content: calls === 1
          ? JSON.stringify({ tool: 'javascript_interpret', code: 'console.log(6 * 7)' })
          : '42'
      } }) }
    }
    const result = await runOllamaStudyEngine({
      ...request, prompt: 'Calculate the product of 6 and 7, then subtract 0.', retrievedSources: [], javascriptInterpret: true
    }, {
      signal: controller.signal,
      onAnswer: (answer, hasFollowUps) => published.push({ answer, hasFollowUps }),
      onJavascriptInterpretTrace: value => { trace = value }
    })
    assert.equal(calls, 2)
    assert.equal(trace.attempts[0].execution.result, '42')
    assert.equal(published.length, 1)
    assert.equal(published[0].hasFollowUps, false)
    assert.deepEqual(published[0].answer, result)
    assert.match(result.text, /42/)
  } finally { globalThis.fetch = original }
})

test('JavaScript toggle allows one direct source answer without tool execution', async () => {
  const original = globalThis.fetch
  let calls = 0
  let trace
  const prompt = 'According to the textbook, which SQL standards were published after 2000, and how many are there?'
  const excerpt = 'The SQL-92 standard was followed by SQL: 1999, SQL: 2003, SQL: 2006, SQL: 2008, SQL: 2011, and SQL: 2016.'
  try {
    globalThis.fetch = async (_url, options) => {
      calls += 1
      const body = JSON.parse(options.body)
      assert.equal(body.format, undefined)
      assert.match(body.messages.at(-1).content, /SQL: 2003/)
      return { ok: true, json: async () => ({ done_reason: 'stop', message: {
        content: 'SQL:2003, SQL:2006, SQL:2008, SQL:2011, and SQL:2016 were published after 2000. There are five.'
      } }) }
    }
    const result = await runOllamaStudyEngine({
      ...request, prompt, javascriptInterpret: true,
      retrievedSources: [{ title: 'Textbook', excerpt }],
      applicationSettings: { suggestionMode: 'off', followUpSuggestionCount: 0 }
    }, { onJavascriptInterpretTrace: value => { trace = value } })
    assert.equal(calls, 1)
    assert.equal(trace, undefined)
    assert.match(result.text, /SQL:2003.*SQL:2016/s)
    assert.match(result.text, /five/)
    assert.doesNotMatch(result.text, /executed JavaScript/)
  } finally { globalThis.fetch = original }
})

test('a greeting uses normal chat even when JavaScript reasoning is enabled', async () => {
  const original = globalThis.fetch
  let calls = 0
  let trace
  try {
    globalThis.fetch = async (_url, options) => {
      calls += 1
      const body = JSON.parse(options.body)
      assert.equal(body.format, undefined)
      return { ok: true, json: async () => ({ done_reason: 'stop', message: { content: 'Hello!' } }) }
    }
    const result = await runOllamaStudyEngine({
      ...request, prompt: 'hi', javascriptInterpret: true, retrievedSources: [],
      applicationSettings: { suggestionMode: 'off', followUpSuggestionCount: 0 }
    }, { onJavascriptInterpretTrace: value => { trace = value } })
    assert.equal(calls, 1)
    assert.equal(trace, undefined)
    assert.equal(result.text, 'Hello!')
  } finally { globalThis.fetch = original }
})

test('disabled JavaScript setting keeps the normal chat request for a calculation', async () => {
  const original = globalThis.fetch
  let calls = 0
  let trace
  try {
    globalThis.fetch = async (_url, options) => {
      calls += 1
      const body = JSON.parse(options.body)
      assert.equal(body.format, undefined)
      return { ok: true, json: async () => ({ done_reason: 'stop', message: { content: '6 × 7 = 42.' } }) }
    }
    const result = await runOllamaStudyEngine({
      ...request, prompt: 'Calculate the product of 6 and 7, then subtract 0.', javascriptInterpret: false,
      applicationSettings: { suggestionMode: 'off', followUpSuggestionCount: 0 }, retrievedSources: []
    }, { onJavascriptInterpretTrace: value => { trace = value } })
    assert.equal(calls, 1)
    assert.equal(trace, undefined)
    assert.equal(result.text, '6 × 7 = 42.')
  } finally { globalThis.fetch = original }
})

test('JavaScript toggle still computes when a retrieved quantitative operation is verified', async () => {
  const original = globalThis.fetch
  let calls = 0
  let trace
  const prompt = 'From the section results, state the section sizes and scores, list each weighted contribution, and calculate the overall weighted average score.'
  const excerpt = 'Section A has 30 students with an average score of 80. Section B has 45 students with an average score of 70. Section C has 25 students with an average score of 90.'
  try {
    globalThis.fetch = async (_url, options) => {
      calls += 1
      const body = JSON.parse(options.body)
      if (calls === 1) assert.equal(body.format?.type, 'object')
      else assert.equal(body.format, undefined)
      return { ok: true, json: async () => ({ done_reason: 'stop', message: { content: calls === 1
        ? JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("2400, 3150, 2250, 78")' })
        : 'Section A: 30 × 80 = 2400; section B: 45 × 70 = 3150; section C: 25 × 90 = 2250. Add the contributions and divide by 100 students for an overall weighted average of 78.'
      } }) }
    }
    const result = await runOllamaStudyEngine({
      ...request, prompt, javascriptInterpret: true,
      retrievedSources: [{ title: 'Section results', excerpt }],
      applicationSettings: { suggestionMode: 'off', followUpSuggestionCount: 0 }
    }, { onJavascriptInterpretTrace: value => { trace = value } })
    assert.equal(calls, 2)
    assert.equal(trace?.inferenceCount, 2)
    assert.match(result.text, /weighted average of 78/)
  } finally { globalThis.fetch = original }
})

test('ambiguous requests leave the direct-or-tool choice to the model', async () => {
  const original = globalThis.fetch
  let trace
  try {
    globalThis.fetch = async (_url, options) => {
      const body = JSON.parse(options.body)
      assert.deepEqual(body.format.properties.route.enum, ['answer', 'javascript_interpret'])
      return { ok: true, json: async () => ({ done_reason: 'stop', message: {
        content: JSON.stringify({ route: 'answer', content: 'Enrollment changed over time.' })
      } }) }
    }
    const result = await runOllamaStudyEngine({
      ...request, prompt: 'How has enrollment changed?', javascriptInterpret: true,
      applicationSettings: { suggestionMode: 'off', followUpSuggestionCount: 0 }
    }, { onJavascriptInterpretTrace: value => { trace = value } })
    assert.equal(result.text, 'Enrollment changed over time.')
    assert.equal(trace?.directAnswer, true)
    assert.deepEqual(trace?.attempts, [])
  } finally { globalThis.fetch = original }
})
