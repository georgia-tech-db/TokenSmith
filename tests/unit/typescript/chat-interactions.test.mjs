import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { addQuoteToDraft, canExplainSimpler, questionForAnswer, replaceQuestion, simplerExplanationSettings } =
  requireTranspiledTs('src/renderer/src/chat-interactions.ts')
const { studyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')

test('adding a selection keeps the existing draft and quotes every line', () => {
  assert.equal(addQuoteToDraft('Why is this needed?', 'Track the path.\nUpdate the parent.'),
    '> Track the path.\n> Update the parent.\n\nWhy is this needed?')
})

test('quotes preserve blank lines and code without interpreting the text', () => {
  assert.equal(addQuoteToDraft('', '  path.push_back(node);\r\n\r\nnode = child;  '),
    '> path.push_back(node);\n> \n> node = child;\n\n')
  assert.equal(addQuoteToDraft('draft', ' \n '), 'draft')
})

test('adding another quote does not erase a previous quote or draft', () => {
  const first = addQuoteToDraft('Explain this.', 'First selection')
  assert.equal(addQuoteToDraft(first, 'Second selection'),
    '> Second selection\n\n> First selection\n\nExplain this.')
})

const history = Object.freeze([
  Object.freeze({ id: 'q1', role: 'user', text: 'What is a B+ tree?' }),
  Object.freeze({ id: 'a1', role: 'assistant', text: 'A balanced search tree.' }),
  Object.freeze({ id: 'q2', role: 'user', text: 'Why track the path?' }),
  Object.freeze({ id: 'a2', role: 'assistant', text: 'To update parents.' }),
  Object.freeze({ id: 'q3', role: 'user', text: 'Give an example.' }),
  Object.freeze({ id: 'a3', role: 'assistant', text: 'Here is an example.' })
])

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

test('re-explaining an answer reuses the question that produced it, not a later one', () => {
  assert.equal(questionForAnswer(history, 'a2').id, 'q2')
  assert.equal(questionForAnswer(history, 'a1').id, 'q1')
  assert.equal(questionForAnswer(history, 'missing'), undefined)
  // An answer with no question before it cannot be re-explained.
  assert.equal(questionForAnswer([{ id: 'a0', role: 'assistant', text: 'Orphan.' }], 'a0'), undefined)
  assert.equal(history[0].text, 'What is a B+ tree?')
})

test('explain simpler is offered only where a simpler answer makes sense', () => {
  const answer = { id: 'a1', role: 'assistant', text: 'An answer.' }
  assert.equal(canExplainSimpler(answer, 'on'), true)
  assert.equal(canExplainSimpler({ ...answer, explanationDepth: 'detailed' }, 'on'), true)

  // It reads as a suggested follow-up, so it disappears with them.
  assert.equal(canExplainSimpler(answer, 'off'), false)
  // Nothing simpler to offer.
  assert.equal(canExplainSimpler({ ...answer, explanationDepth: 'simple' }, 'on'), false)
  // Simplifying a quiz question would defeat the quiz.
  assert.equal(canExplainSimpler({ ...answer, kind: 'quizQuestion' }, 'on'), false)
  assert.equal(canExplainSimpler({ ...answer, kind: 'chat' }, 'on'), true)
  // Questions are not re-explained, answers are.
  assert.equal(canExplainSimpler({ id: 'q1', role: 'user', text: 'A question.' }, 'on'), false)
})

test('a retelling reaches the model as simply as asking on simple mode would have', () => {
  const model = {
    id: 'ollama:llama3', name: 'llama3', engine: 'ollama', role: 'generator',
    status: 'ready', source: 'ollama', ollamaModelName: 'llama3', addedAt: '2026-07-07T00:00:00.000Z'
  }
  const sources = [{ title: 'textbook', locator: 'Page 894', excerpt: 'Two-phase locking does not prevent deadlock.' }]
  const question = 'how does two phase locking prevent deadlock'
  const base = { prompt: question, materials: [], model, settings: {}, modelSettings: {}, retrievedSources: sources }
  const application = { suggestionMode: 'on', followUpSuggestionCount: 4, showSources: true }

  // Asked fresh while the depth picker is on simple.
  const asked = studyChatMessages({
    ...base,
    messages: [],
    applicationSettings: { ...application, explanationDepthEnabled: true, explanationDepth: 'simple' }
  })
  // Asked on standard, then retold: same question, same borrowed evidence, one-shot override.
  const retold = studyChatMessages({
    ...base,
    messages: [{ id: 'q1', role: 'user', text: question }, { id: 'a1', role: 'assistant', text: 'A dense answer.' }],
    applicationSettings: simplerExplanationSettings({ ...application, explanationDepth: 'standard' })
  })

  assert.deepEqual(retold, asked)
  assert.match(retold[0].content, /Explanation depth: the student is meeting this idea for the first time/)
  assert.match(retold.at(-1).content, /Two-phase locking does not prevent deadlock/)
})

test('a retelling is simplified even when the depth picker was never switched on', () => {
  // The chip rides with the follow-ups, so it must not depend on the depth setting.
  const settings = simplerExplanationSettings({ suggestionMode: 'on', explanationDepthEnabled: false, explanationDepth: 'standard' })
  assert.equal(settings.explanationDepthEnabled, true)
  assert.equal(settings.explanationDepth, 'simple')
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
