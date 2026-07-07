import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { questionSuggestionMessages, sourceContext } = requireTranspiledTs('src/main/engine/study-chat-format.ts')

const addedAt = '2026-07-07T00:00:00.000Z'
const databaseSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 4',
  excerpt: 'Transactions preserve atomicity and durability.'
}
const databaseContext = [
  'Use these excerpts when they are relevant. If the excerpts do not contain the answer, say that plainly.',
  'Source 1: Database Systems.pdf (Page 4)\nTransactions preserve atomicity and durability.'
].join('\n\n')
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

test('sourceContext matches the v0.1.5 excerpt prompt shape', () => {
  assert.equal(sourceContext([databaseSource]), databaseContext)
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
