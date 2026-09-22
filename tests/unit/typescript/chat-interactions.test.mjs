import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { replaceQuestion } = requireTranspiledTs('src/renderer/src/chat-interactions.ts')
const { selectChatPassage } = requireTranspiledTs('src/shared/chat-selection.ts')
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

test('resending a quiz turn as a study question removes stale quiz metadata', () => {
  assert.deepEqual(replaceQuestion([
    { id: 'quiz-answer', role: 'user', text: 'An answer', kind: 'quizAnswer', quiz: { questionNumber: 2 } }
  ], 'quiz-answer', 'Explain my answer.'), [
    { id: 'quiz-answer', role: 'user', text: 'Explain my answer.' }
  ])
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
      return { mode: 'contextual', query: 'How is a B+ tree kept balanced?', clarification: '' }
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
