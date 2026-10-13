import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { canRetryWithReasoning, reasoningRetryRequest, simplerExplanationRequest } = requireTranspiledTs('src/renderer/src/chat-interactions.ts')
const { answerForDisplay, lastChatExchange } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
const { studyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const context = {
  model: { id: 'test', name: 'Custom', engine: 'ollama', ollamaModelName: 'custom', status: 'ready', contextLength: 8192 },
  materials: [], settings: { application: { suggestionMode: 'on', followUpSuggestionCount: 3 } },
  modelSettings: { reasoningMode: 'auto', contextLength: 8192, maxLength: 768 }
}
const answer = {
  id: 'a2', role: 'assistant', text: 'OLD ANSWER TO REPLACE.', conversationContextMode: 'contextual',
  reasoning: { modelId: 'test', supported: true, used: false },
  sources: [{ title: 'Course notes', excerpt: 'Each internal node has one more child than key.' }],
  answerContext: {
    prompt: 'Apply it to these inputs.', answerPrompt: 'Apply it to these inputs.', retrievalQuery: 'Split the internal node.',
    conversationContextMode: 'contextual', referenceExchange: { question: 'Consider keys 10, 20 and children A, B, C.', answer: 'We can split this node.' },
    modelSettings: { reasoningMode: 'off', contextLength: 8192, maxLength: 1536, temperature: 0.2 },
    applicationSettings: { suggestionMode: 'off', explanationDepth: 'standard' }
  }
}
const messages = [
  { id: 'q1', role: 'user', text: answer.answerContext.referenceExchange.question },
  { id: 'a1', role: 'assistant', text: answer.answerContext.referenceExchange.answer },
  { id: 'q2', role: 'user', text: answer.answerContext.prompt }, answer,
  { id: 'q3', role: 'user', text: 'LATER UNRELATED TOPIC.' },
  { id: 'a3', role: 'assistant', text: 'LATER UNRELATED ANSWER.' }
]

test('reasoning retry requires recorded non-reasoning generation on the same ready, supported model', () => {
  assert.equal(canRetryWithReasoning(answer, context.model), true)
  for (const patch of [
    { reasoning: undefined }, { reasoning: { ...answer.reasoning, supported: undefined } },
    { reasoning: { ...answer.reasoning, supported: false } }, { reasoning: { ...answer.reasoning, used: true } },
    { answerContext: undefined }, { reasoningAnswer: { text: 'Already retried.' } },
    { role: 'user' }, { text: '' }, { conversationContextMode: 'clarify' }
  ]) assert.equal(canRetryWithReasoning({ ...answer, ...patch }, context.model), false, JSON.stringify(patch))
  for (const model of [undefined, { ...context.model, engine: 'remote' }, { ...context.model, id: 'different' }, { ...context.model, status: 'error' }]) {
    assert.equal(canRetryWithReasoning(answer, model), false)
  }
})

test('retry preserves the original task, settings and evidence, excluding its answer and later turns', () => {
  const before = structuredClone({ messages, context })
  const request = reasoningRetryRequest(messages, 'a2', context)
  assert.equal(request.prompt, answer.answerContext.prompt)
  assert.equal(request.answerPrompt, answer.answerContext.answerPrompt)
  assert.deepEqual(request.referenceExchange, answer.answerContext.referenceExchange)
  assert.equal(request.retrievalQuery, answer.answerContext.retrievalQuery)
  assert.deepEqual(request.messages, messages.slice(0, 2))
  assert.deepEqual(request.retrievedSources, answer.sources)
  assert.equal(request.reasoning, true)
  assert.equal(request.modelSettings.thinking, true)
  assert.equal(request.modelSettings.reasoningMode, 'on')
  assert.equal(request.modelSettings.maxLength, 4096)
  assert.equal(request.modelSettings.temperature, 0.2)
  assert.equal(request.answerToSimplify, undefined)
  assert.equal(request.applicationSettings.suggestionMode, 'on')
  assert.equal(request.applicationSettings.followUpSuggestionCount, 3)
  const prompt = studyChatMessages(request).map(message => message.content).join('\n')
  assert.match(prompt, /children A, B, C/)
  assert.match(prompt, /one more child than key/)
  assert.doesNotMatch(prompt, /OLD ANSWER|LATER UNRELATED/)
  assert.deepEqual({ messages, context }, before)
  assert.equal(reasoningRetryRequest(messages, 'missing', context), undefined)
  assert.equal(reasoningRetryRequest([answer], 'a2', context), undefined)
})

test('retry preserves selected passages and supports a first question without history or sources', () => {
  const selectedPassage = { messageId: 'earlier', role: 'assistant', text: 'Selected example.', question: 'Original question?' }
  const selected = { ...answer, answerContext: { ...answer.answerContext, selectedPassage } }
  assert.deepEqual(reasoningRetryRequest([...messages.slice(0, 3), selected], 'a2', context).selectedPassage, selectedPassage)
  const first = { ...answer, sources: [], answerContext: { prompt: 'An independent question.', conversationContextMode: 'standalone' } }
  const request = reasoningRetryRequest([{ id: 'q', role: 'user', text: first.answerContext.prompt }, first], 'a2', context)
  assert.deepEqual(request.messages, [])
  assert.deepEqual(request.retrievedSources, [])
  assert.equal(request.referenceExchange, undefined)
})

test('saved retry settings do not change the simpler explanation action on a newly selected model', () => {
  const request = simplerExplanationRequest(messages, 'a2', context)
  assert.deepEqual(request.modelSettings, context.modelSettings)
  assert.equal(request.applicationSettings.explanationDepth, 'simple')
  assert.equal(request.applicationSettings.suggestionMode, 'off')
})

test('both answers survive persistence; subsequent turns and selections see the chosen version', () => {
  const retried = JSON.parse(JSON.stringify({ ...answer, explanationView: 'reasoning', reasoningAnswer: {
    text: 'New answer.', sources: answer.sources, reasoning: { ...answer.reasoning, used: true },
    responseDurationMs: 1234, followUpSuggestions: ['Why retain both children?']
  } }))
  assert.equal(answerForDisplay(retried).text, 'New answer.')
  assert.equal(lastChatExchange([...messages.slice(0, 3), retried]).answer, 'New answer.')
  assert.equal(reasoningRetryRequest([...messages.slice(0, 3), retried], 'a2', context), undefined)
  retried.explanationView = 'original'
  assert.equal(answerForDisplay(retried).text, answer.text)
  assert.equal(lastChatExchange([...messages.slice(0, 3), retried]).answer, answer.text)
})

test('Ollama reports effective reasoning with both early and final answers; explicit retry skips planning', async () => {
  const originalFetch = globalThis.fetch
  try {
    for (const enabled of [false, true]) {
      const calls = []
      globalThis.fetch = async (url, options) => {
        const body = JSON.parse(options.body)
        if (url.endsWith('/show')) return { ok: true, json: async () => ({ capabilities: ['thinking'] }) }
        calls.push(body)
        assert.equal(body.think, enabled)
        assert.equal(body.format, undefined)
        return { ok: true, json: async () => body.options.num_predict === 1
          ? { prompt_eval_count: 1500 } : { done_reason: 'stop', message: { content: 'An answer.' } } }
      }
      const retry = reasoningRetryRequest(messages, 'a2', context)
      let early
      const response = await runOllamaStudyEngine({ ...retry, reasoning: true,
        modelSettings: { ...retry.modelSettings, reasoningMode: enabled ? 'on' : 'off' },
        applicationSettings: { suggestionMode: 'off' }
      }, { onAnswer: answer => { early = answer } })
      assert.deepEqual(response.reasoning, { modelId: 'test', supported: true, used: enabled })
      assert.deepEqual(early.reasoning, response.reasoning)
      assert.equal(calls.length, 2, 'Only token preflight and answer generation, no rewrite or retrieval')
    }
  } finally { globalThis.fetch = originalFetch }
})

test('unavailable capability metadata does not block Off answers or claim reasoning availability', async () => {
  const originalFetch = globalThis.fetch
  try {
    globalThis.fetch = async (url, options) => {
      if (url.endsWith('/show')) throw new Error('Metadata unavailable')
      const body = JSON.parse(options.body)
      return { ok: true, json: async () => body.options.num_predict === 1
        ? { prompt_eval_count: 1500 } : { done_reason: 'stop', message: { content: 'An answer.' } } }
    }
    const request = { ...reasoningRetryRequest(messages, 'a2', context), applicationSettings: { suggestionMode: 'off' } }
    const response = await runOllamaStudyEngine({ ...request, modelSettings: { ...request.modelSettings, reasoningMode: 'off' } })
    assert.equal(response.reasoning.supported, undefined)
    assert.equal(response.reasoning.used, false)
    await assert.rejects(runOllamaStudyEngine(request), /Metadata unavailable/)
  } finally { globalThis.fetch = originalFetch }
})
