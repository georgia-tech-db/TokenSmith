import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  answerWithOrderedSources,
  formatFollowUpInstruction,
  parseFollowUpSuggestions,
  questionSuggestionCount,
  questionSuggestionMessages,
  shouldGenerateFollowUps,
  sourceContext,
  studyChatMessages
} = requireTranspiledTs('src/main/engine/study-chat-format.ts')

const addedAt = '2026-07-07T00:00:00.000Z'
const databaseSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 4',
  excerpt: 'Transactions preserve atomicity and durability.'
}
const loggingSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 9',
  excerpt: 'Logging records allow recovery after crashes.'
}
const databaseContext = [
  'Use the context below only when it is relevant to the question.',
  'Answer directly. Do not quote the context before answering. Do not mention context labels, source labels, excerpt labels, locators, or page numbers.',
  'If the context does not contain the answer, say that plainly.',
  '',
  '### Context:',
  'Collection: Database Systems.pdf',
  'Path: Database Systems.pdf',
  'Text: Transactions preserve atomicity and durability.'
].join('\n')
const ollamaChatModel = {
  id: 'ollama:llama3',
  name: 'Ollama llama3',
  engine: 'ollama',
  role: 'generator',
  status: 'ready',
  source: 'ollama',
  ollamaModelName: 'llama3',
  addedAt
}

function suggestionRequest(overrides = {}) {
  return {
    messages: [],
    materials: [],
    model: ollamaChatModel,
    settings: {},
    applicationSettings: {
      suggestionMode: 'on',
      followUpSuggestionCount: 4
    },
    modelSettings: {},
    retrievedSources: [databaseSource],
    ...overrides
  }
}

test('sourceContext uses a neutral context block without model-facing page locators', () => {
  assert.equal(sourceContext([databaseSource]), databaseContext)
  assert.equal(sourceContext([databaseSource]).includes('Locator: Page 4'), false)
})

test('studyChatMessages includes system text, recent conversation, and retrieved source context', () => {
  const messages = Array.from({ length: 14 }, (_, index) => ({
    role: index % 2 === 0 ? 'user' : 'assistant',
    text: index === 3 ? '   ' : `message ${index}`
  }))

  const chatMessages = studyChatMessages({
    prompt: 'What is atomicity?',
    messages,
    materials: [],
    model: ollamaChatModel,
    settings: {},
    applicationSettings: {
      suggestionMode: 'on',
      followUpSuggestionCount: 4
    },
    modelSettings: {
      systemMessage: 'Answer only from PDFs.'
    },
    retrievedSources: [databaseSource]
  })

  assert.equal(chatMessages[0].role, 'system')
  assert.equal(chatMessages[0].content, 'Answer only from PDFs.')
  assert.equal(chatMessages.some((message) => message.content === 'message 0'), false)
  assert.equal(chatMessages.some((message) => message.content === 'message 3'), false)
  assert.equal(chatMessages.at(-1).role, 'user')
  assert.match(chatMessages.at(-1).content, /Transactions preserve atomicity and durability/)
  assert.match(chatMessages.at(-1).content, /Question: What is atomicity\?/)
})

test('answerWithOrderedSources removes source-number wording and moves the cited source first', () => {
  const response = answerWithOrderedSources(
    'According to Source 2, logging records allow recovery after crashes. This should be studied with checkpoints.',
    [databaseSource, loggingSource]
  )

  assert.equal(response.text, 'logging records allow recovery after crashes. This should be studied with checkpoints.')
  assert.deepEqual(response.sources, [loggingSource, databaseSource])
})

test('answerWithOrderedSources strips leaked source-label instruction chatter', () => {
  const response = answerWithOrderedSources(
    'Atomicity is all-or-nothing. The question does not provide information about source numbers or excerpts labelled according to Source 1.',
    [databaseSource]
  )

  assert.equal(response.text, 'Atomicity is all-or-nothing.')
  assert.deepEqual(response.sources, [databaseSource])
})

test('answerWithOrderedSources strips quoted context preambles', () => {
  const response = answerWithOrderedSources(
    '"Logging records allow recovery after crashes." So, according to this excerpt, logs help restore the database after a failure.',
    [databaseSource, loggingSource]
  )

  assert.equal(response.text, 'logs help restore the database after a failure.')
  assert.deepEqual(response.sources, [loggingSource, databaseSource])
})

test('question suggestion count and mode handling stay bounded', () => {
  assert.equal(questionSuggestionCount({ suggestionMode: 'off', followUpSuggestionCount: 4 }), 0)
  assert.equal(questionSuggestionCount({ suggestionMode: 'on', followUpSuggestionCount: 1 }), 2)
  assert.equal(questionSuggestionCount({ suggestionMode: 'on', followUpSuggestionCount: 4 }), 4)
  assert.equal(questionSuggestionCount(), 4)
  assert.equal(shouldGenerateFollowUps({ applicationSettings: { suggestionMode: 'off' } }), false)
})

test('questionSuggestionMessages uses source context before the shared question prompt', () => {
  const messages = questionSuggestionMessages(suggestionRequest({
    modelSettings: {
      suggestedFollowUpPrompt: 'Use my shared question prompt for {count} questions.'
    }
  }))

  assert.deepEqual(messages, [
    { role: 'user', content: databaseContext },
    { role: 'user', content: 'Use my shared question prompt for 4 questions.' }
  ])
})

test('questionSuggestionMessages prefixes prompts without a count placeholder', () => {
  const messages = questionSuggestionMessages(suggestionRequest({
    applicationSettings: {
      suggestionMode: 'on',
      followUpSuggestionCount: 2
    },
    modelSettings: {
      suggestedFollowUpPrompt: 'Ask about adjacent database concepts.'
    }
  }))

  assert.deepEqual(messages, [
    { role: 'user', content: databaseContext },
    { role: 'user', content: 'Generate 2 suggested follow-up questions.\nAsk about adjacent database concepts.' }
  ])
})

test('formatFollowUpInstruction uses explicit placeholders without adding a second prefix', () => {
  assert.equal(formatFollowUpInstruction('Ask {count} PDF questions.', 2), 'Ask 2 PDF questions.')
  assert.equal(
    formatFollowUpInstruction('Ask short questions.', 1),
    'Generate 1 suggested follow-up question.\nAsk short questions.'
  )
})

test('parseFollowUpSuggestions handles JSON, plain text, dedupe, and limits', () => {
  assert.deepEqual(
    parseFollowUpSuggestions(JSON.stringify([
      '1. What is atomicity?',
      'What is atomicity?',
      'Why does durability matter?',
      'This is not a question'
    ]), 2),
    ['What is atomicity?', 'Why does durability matter?']
  )

  assert.deepEqual(
    parseFollowUpSuggestions('Try these: What is a log record? Why does rollback matter?', 4),
    ['What is a log record?', 'Why does rollback matter?']
  )

  assert.deepEqual(
    parseFollowUpSuggestions('- How does recovery use checkpoints?\n- Which files store logs?', 4),
    ['How does recovery use checkpoints?', 'Which files store logs?']
  )
})
