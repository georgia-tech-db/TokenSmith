import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { selectChatPassage } = requireTranspiledTs('src/shared/chat-selection.ts')
const { canExplainSimpler, questionForAnswer, replaceQuestion, simplerExplanationSettings, simplerExplanationRequest } =
  requireTranspiledTs('src/renderer/src/chat-interactions.ts')
const { studyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')

const history = Object.freeze([
  Object.freeze({ id: 'q1', role: 'user', text: 'What is a B+ tree?' }),
  Object.freeze({ id: 'a1', role: 'assistant', text: 'A balanced search tree.' }),
  Object.freeze({ id: 'q2', role: 'user', text: 'Why track the path?' }),
  Object.freeze({ id: 'a2', role: 'assistant', text: 'To update parents.' }),
  Object.freeze({ id: 'q3', role: 'user', text: 'Give an example.' }),
  Object.freeze({ id: 'a3', role: 'assistant', text: 'Here is an example.' })
])

test('a selected passage snapshots its own message and original question, not the latest turn', () => {
  assert.deepEqual(selectChatPassage(history, 'a1', 'balanced search tree'), {
    messageId: 'a1', role: 'assistant', text: 'balanced search tree', question: 'What is a B+ tree?'
  })
  assert.deepEqual(selectChatPassage(history, 'q2', 'track the path'), {
    messageId: 'q2', role: 'user', text: 'track the path', question: 'Why track the path?'
  })
})

test('selections preserve code, whitespace, math and markup literally without editing the question', () => {
  const text = '  path.push_back(node);\r\n\r\nnode = child;\n$x > 1$ <tag>'
  const passage = selectChatPassage(history, 'a3', text)
  assert.equal(passage.text, text)
  assert.equal(history.at(-1).text, 'Here is an example.')
  assert.deepEqual(JSON.parse(JSON.stringify(passage)), passage)
})

test('empty or stale selections are not attached', () => {
  assert.equal(selectChatPassage(history, 'a1', ' \n '), undefined)
  assert.equal(selectChatPassage(history, 'missing', 'A passage'), undefined)
})

test('selecting from an answer to an attached-passage question retains its earlier reference', () => {
  const selectedPassage = selectChatPassage(history, 'a1', 'balanced search tree')
  const messages = [...history,
    { id: 'q4', role: 'user', text: 'Why?', selectedPassage },
    { id: 'a4', role: 'assistant', text: 'To bound search depth.' }
  ]
  const selection = selectChatPassage(messages, 'a4', 'search depth')
  assert.match(selection.question, /balanced search tree/)
  assert.match(selection.question, /Question: Why\?/)
})

test('in-place edits can retain or remove a selected passage without changing the original question', () => {
  const selectedPassage = selectChatPassage(history, 'a1', 'balanced search tree')
  const messages = [...history, { id: 'q4', role: 'user', text: 'Why?', selectedPassage }]
  const edited = replaceQuestion(messages, 'q4', 'How does that work?', selectedPassage)
  assert.equal(edited.at(-1).selectedPassage, selectedPassage)
  assert.equal(edited.at(-1).text, 'How does that work?')
  assert.equal(replaceQuestion(messages, 'q4', 'A different question').at(-1).selectedPassage, undefined)
  assert.equal(messages.at(-1).selectedPassage, selectedPassage)
})

test('an edit replaces a question in place, keeping its id and only its preceding history', () => {
  const edited = replaceQuestion(history, 'q2', '  How are parents updated?  ')
  assert.deepEqual(edited, [...history.slice(0, 2), { id: 'q2', role: 'user', text: 'How are parents updated?' }])
  assert.equal(history.length, 6)
  assert.equal(history[2].text, 'Why track the path?')
  assert.deepEqual(edited.slice(0, -1), history.slice(0, 2))
})

test('editing the first question leaves no old exchange for rewriting', () => {
  const edited = replaceQuestion(history, 'q1', 'How do hash indexes work?')
  assert.equal(edited.length, 1)
  assert.deepEqual(edited.slice(0, -1), [])
})

test('editing the final question replaces rather than duplicates it', () => {
  const edited = replaceQuestion(history, 'q3', 'Show the code.')
  assert.equal(edited.length, 5)
  assert.equal(edited.filter((message) => message.id === 'q3').length, 1)
  assert.deepEqual(edited.slice(0, -1), history.slice(0, 4))
})

test('invalid, stale or empty edits never truncate a conversation', () => {
  assert.equal(replaceQuestion(history, 'missing', 'New question'), undefined)
  assert.equal(replaceQuestion(history, 'a1', 'New question'), undefined)
  assert.equal(replaceQuestion(history, 'q2', '  \n '), undefined)
  assert.equal(history.length, 6)
})

test('re-explaining an answer reuses the question that produced it, not a later one', () => {
  assert.equal(questionForAnswer(history, 'a2').id, 'q2')
  assert.equal(questionForAnswer(history, 'a1').id, 'q1')
  assert.equal(questionForAnswer(history, 'missing'), undefined)
  // An answer with no question before it cannot be re-explained.
  assert.equal(questionForAnswer([{ id: 'a0', role: 'assistant', text: 'Orphan.' }], 'a0'), undefined)
  assert.equal(history[0].text, 'What is a B+ tree?')
})

const evidence = [{ title: 'BuzzDB', locator: '2Q replacement', excerpt: 'If FIFO is empty, skip the FIFO branch and choose a victim from the protected LRU list.' }]
const whyHistory = [
  { id: 'q1', role: 'user', text: 'Assume the cache has three pages and FIFO is empty. Which list supplies the victim?' },
  { id: 'a1', role: 'assistant', text: 'The protected LRU list.', sources: evidence, conversationContextMode: 'standalone' },
  { id: 'q2', role: 'user', text: 'Why?' },
  { id: 'a2', role: 'assistant', text: 'The FIFO branch is skipped because the queue is empty.', sources: evidence, conversationContextMode: 'contextual',
    answerContext: { prompt: 'Why?', retrievalQuery: 'Why does empty FIFO select a victim from protected LRU in a three-page cache?', conversationContextMode: 'contextual',
      referenceExchange: { question: 'Assume the cache has three pages and FIFO is empty. Which list supplies the victim?', answer: 'The protected LRU list.' } } },
  { id: 'q3', role: 'user', text: 'Now explain B+ trees.' },
  { id: 'a3', role: 'assistant', text: 'B+ trees are balanced search trees.', sources: [{ ...evidence[0], excerpt: 'B+ tree leaves are linked.' }] }
]
const simpleContext = {
  model: { id: 'test', name: 'test', engine: 'ollama', source: 'ollama', ollamaModelName: 'test', status: 'ready', contextLength: 8192 },
  settings: { application: { suggestionMode: 'on', followUpSuggestionCount: 4, showSources: false, explanationDepthEnabled: false, explanationDepth: 'standard' } },
  materials: [], modelSettings: { contextLength: 8192, maxLength: 768 }
}

test('manual simplification needs saved evidence, but does not depend on automatic suggestions', () => {
  const answer = whyHistory[1]
  assert.equal(canExplainSimpler(answer), true)
  assert.equal(canExplainSimpler({ ...answer, sources: [] }), false)
  assert.equal(canExplainSimpler({ ...answer, text: ' ' }), false)
  assert.equal(canExplainSimpler({ ...answer, explanationDepth: 'simple' }), false)
  assert.equal(canExplainSimpler({ ...answer, conversationContextMode: 'clarify' }), false)
  assert.equal(canExplainSimpler({ ...answer, role: 'user' }), false)
  const request = simplerExplanationRequest(whyHistory, 'a2', {
    ...simpleContext, settings: { application: { ...simpleContext.settings.application, suggestionMode: 'off' } }
  })
  assert.ok(request)
  assert.equal(request.applicationSettings.explanationDepth, 'simple')
  assert.equal(request.applicationSettings.suggestionMode, 'off')
  assert.equal(request.applicationSettings.followUpSuggestionCount, 0)
  assert.equal(simpleContext.settings.application.explanationDepth, 'standard')
})

test('simplifying an older Why answer preserves assumptions, evidence and target answer, excluding later topics', () => {
  const request = simplerExplanationRequest(whyHistory, 'a2', simpleContext)
  assert.deepEqual(request.referenceExchange, whyHistory[3].answerContext.referenceExchange)
  assert.equal(request.retrievalQuery, whyHistory[3].answerContext.retrievalQuery)
  assert.deepEqual(request.retrievedSources, evidence)
  assert.equal(request.answerToSimplify, whyHistory[3].text)
  assert.equal(request.prompt, 'Why?')
  const prompt = studyChatMessages(request).map(message => message.content).join('\n')
  assert.match(prompt, /three pages and FIFO is empty/)
  assert.match(prompt, /FIFO branch is skipped because the queue is empty/)
  assert.match(prompt, /not factual evidence/)
  assert.match(prompt, /Check claims against the study excerpts; correct mistakes/)
  assert.match(prompt, /Keep the conditions, assumptions, exceptions/)
  assert.doesNotMatch(prompt, /B\+ trees/)
})

test('legacy contextual answers recover their preceding exchange without consulting later turns', () => {
  const history = structuredClone(whyHistory)
  delete history[3].answerContext
  const request = simplerExplanationRequest(history, 'a2', simpleContext)
  assert.equal(request.referenceExchange.question, whyHistory[0].text)
  assert.equal(request.referenceExchange.answer, whyHistory[1].text)
  const standalone = simplerExplanationRequest(history, 'a3', {
    ...simpleContext, settings: { ...simpleContext.settings, application: simplerExplanationSettings(simpleContext.settings.application) }
  })
  assert.equal(standalone.prompt, 'Now explain B+ trees.')
})

test('cached explanations survive serialization and never need another generation request', () => {
  const history = structuredClone(whyHistory)
  history[3].simplerExplanation = { text: 'FIFO has no page to remove, so use LRU.', sources: evidence, responseDurationMs: 500 }
  history[3].explanationView = 'simple'
  const restored = JSON.parse(JSON.stringify(history))
  assert.equal(simplerExplanationRequest(restored, 'a2', simpleContext), undefined)
  const { answerForDisplay, lastChatExchange } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
  assert.equal(answerForDisplay(restored[3]).text, history[3].simplerExplanation.text)
  assert.equal(lastChatExchange(restored.slice(0, 4)).answer, history[3].simplerExplanation.text)
  restored[3].explanationView = 'original'
  assert.equal(answerForDisplay(restored[3]).text, whyHistory[3].text)
  assert.equal(lastChatExchange(restored.slice(0, 4)).answer, whyHistory[3].text)
  assert.equal(simplerExplanationRequest(restored, 'missing', simpleContext), undefined)
})

test('simplification preserves a selected passage together with its original question context', () => {
  const history = structuredClone(whyHistory)
  const selectedPassage = { messageId: 'earlier', role: 'assistant', text: 'A pinned page stays in the buffer.', question: 'When can a buffer frame be reused?' }
  history[2].selectedPassage = selectedPassage
  history[3].answerContext.selectedPassage = selectedPassage
  const request = simplerExplanationRequest(history, 'a2', simpleContext)
  assert.deepEqual(request.selectedPassage, selectedPassage)
  const prompt = studyChatMessages(request).at(-1).content
  assert.match(prompt, /Selected passage/)
  assert.match(prompt, /A pinned page stays in the buffer/)
  assert.match(prompt, /Answer to explain more simply/)
})

test('a simplification makes one provider call even with automatic suggestions enabled globally', async () => {
  const { runRemoteStudyEngine } = requireTranspiledTs('src/main/engine/remote-chat-service.ts')
  const request = simplerExplanationRequest(whyHistory, 'a2', simpleContext)
  request.model = { ...request.model, engine: 'remote', remoteModelName: 'test', baseUrl: 'https://example.test/v1', apiKey: 'test-key' }
  const originalFetch = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) })
    return { ok: true, json: async () => ({ choices: [{ message: { content: 'FIFO is empty, so choose from LRU.' } }] }) }
  }
  try {
    const reply = await runRemoteStudyEngine(request)
    assert.equal(calls.length, 1)
    assert.match(calls[0].body.messages.at(-1).content, /three pages and FIFO is empty/)
    assert.deepEqual(reply.followUpSuggestions, [])
    assert.ok(reply.sources.length)
  } finally { globalThis.fetch = originalFetch }
})

test('resending an edited follow-up passes only the earlier exchange through the real chat pipeline', async () => {
  const edited = replaceQuestion(history, 'q2', 'How is it kept balanced?')
  let rewriteMessages
  let searchQuery
  const result = await prepareRewrittenStudyChat({
    prompt: edited.at(-1).text,
    messages: edited.slice(0, -1)
  }, {
    resolve: async (request) => {
      rewriteMessages = request.messages
      return { mode: 'contextual', query: 'How is a B+ tree kept balanced?', clarification: '', reasoning: false }
    },
    search: async (query) => { searchQuery = query; return [] }
  })
  assert.deepEqual(rewriteMessages, history.slice(0, 2))
  assert.equal(searchQuery, 'How is a B+ tree kept balanced?')
  assert.deepEqual(result.request.referenceExchange, { question: history[0].text, answer: history[1].text })
  assert.equal(result.request.answerPrompt, 'How is it kept balanced?')
  assert.deepEqual(result.request.messages, history.slice(0, 2))
})

test('resending the first question cannot reuse answers from the discarded branch', async () => {
  const edited = replaceQuestion(history, 'q1', 'How do hash indexes work?')
  const result = await prepareRewrittenStudyChat({ prompt: edited[0].text, messages: [] }, {
    resolve: async () => assert.fail('The first question should not be rewritten using old history'),
    search: async () => []
  })
  assert.equal(result.resolution.mode, 'standalone')
  assert.equal(result.request.referenceExchange, undefined)
})
