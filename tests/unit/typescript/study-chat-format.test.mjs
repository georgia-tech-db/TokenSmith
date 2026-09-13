import assert from 'node:assert/strict'
import test from 'node:test'
import { buzzdbSource } from '../../helpers/buzzdb-fixture.mjs'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  answerWithOrderedSources,
  estimateTokens,
  filterSuggestedQuestions,
  followUpSuggestionMessages,
  modelAwareRuntimeSettings,
  sourceContextBudgetForRequest,
  formatFollowUpInstruction,
  parseFollowUpSuggestions,
  questionSuggestionCount,
  questionSuggestionMessages,
  shouldGenerateFollowUps,
  sourceContext,
  suggestionPromptFor,
  studyChatMessages,
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
const bplusTreeSource = buzzdbSource('ch09.019')
const pinningSource = buzzdbSource('ch05.084')
const databaseContext = [
  'Use the provided context as the factual basis for the answer.',
  'Do not add facts from general knowledge when the context does not support them.',
  'Answer directly for a student with enough detail to teach the concept. Use only relevant evidence and keep the answer scoped to the user\'s question.',
  'Format the answer for readability: use short paragraphs with one idea each; use a compact bullet list or numbered list only for steps, comparisons, or multiple distinct points.',
  'For how, why, definition, or elaboration questions, give a clear explanation instead of a one-sentence answer: name the idea, explain how it works, and include one brief concrete example or mini-walkthrough when the context supports one.',
  'Explain mechanism and consequence; for yes/no, comparison, or judgment questions, start with the conclusion, name the comparison target, and state the workload or condition behind the trade-off.',
  'For "also", "too", or "as well" questions, answer yes only if the context supports the claim for the current target. If the context supports only a related target, say the context does not say.',
  'Do not overstate with words like always, faster, or better unless the context gives that condition.',
  'Do not quote the context before answering. Do not mention context labels, source labels, excerpt labels, locators, or page numbers.',
  'If the context does not contain the answer, say that plainly and do not speculate.',
  'Do not end by asking whether the student wants more detail.',
  '',
  '### Context:',
  'Collection: Database Systems.pdf',
  'Path: Database Systems.pdf',
  'Text: Transactions preserve atomicity and durability.'
].join('\n')
const databaseSuggestionContext = [
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

test('sourceContext prefers full chunk context over the short excerpt', () => {
  const context = sourceContext([
    {
      ...databaseSource,
      excerpt: 'Short excerpt only.',
      context: 'Full chunk context explains why protection prevents unsafe replacement.'
    }
  ])

  assert.match(context, /Full chunk context explains why protection prevents unsafe replacement/)
  assert.doesNotMatch(context, /Short excerpt only/)
})

test('modelAwareRuntimeSettings uses discovered model context with a bounded automatic cap', () => {
  const runtimeSettings = modelAwareRuntimeSettings({
    model: {
      ...ollamaChatModel,
      contextLength: 16384
    },
    modelSettings: {
      contextLength: 2048,
      maxLength: 512
    }
  })

  assert.equal(runtimeSettings.contextLength, 8192)
  assert.equal(runtimeSettings.maxLength, 512)
})

test('studyChatMessages budgets and clips source context around matched terms', () => {
  const request = {
    prompt: 'Why is pinning needed?',
    messages: [],
    materials: [],
    model: {
      ...ollamaChatModel,
      contextLength: 2048
    },
    settings: {},
    applicationSettings: {
      suggestionMode: 'off',
      followUpSuggestionCount: 0
    },
    modelSettings: {
      contextLength: 2048,
      maxLength: 512
    },
    retrievedSources: [
      {
        title: 'BuzzDBBook',
        locator: 'Section 5',
        excerpt: 'Pinning protects a page while it is in use.',
        context: `${'cold filler '.repeat(1400)}Pinning protects a page while it is in use.${'cold filler '.repeat(1400)}`,
        collectionName: 'BuzzDBBook',
        path: '/tmp/buzzdb-book.tokensmith.md',
        sectionHeader: 'Buffer Pool'
      },
      {
        title: 'BuzzDBBook',
        locator: 'Section 6',
        excerpt: 'LRU evicts cold pages.',
        context: 'LRU evicts cold pages after they stop being accessed.',
        collectionName: 'BuzzDBBook',
        path: '/tmp/buzzdb-book.tokensmith.md',
        sectionHeader: 'LRU'
      }
    ]
  }

  const chatMessages = studyChatMessages(request)
  const prompt = chatMessages.at(-1).content
  const budget = sourceContextBudgetForRequest(request)

  assert.equal(budget.modelContextTokens, 2048)
  assert.equal(budget.includedSourceCount, 1)
  assert.equal(budget.truncatedSourceCount, 1)
  assert.ok(budget.estimatedPromptTokens <= budget.modelContextTokens - budget.answerReserveTokens - budget.safetyMarginTokens)
  assert.ok(estimateTokens(prompt) <= budget.estimatedPromptTokens + 20)
  assert.match(prompt, /Pinning protects a page while it is in use/)
  assert.match(prompt, /Text: \.\.\./)
  assert.doesNotMatch(prompt, /LRU evicts cold pages after/)
})

test('studyChatMessages sends standalone source context without raw prior conversation', () => {
  const chatMessages = studyChatMessages({
    prompt: 'What exactly is a B+ tree and how is it different from a binary search tree?',
    messages: [
      {
        id: 'u1',
        role: 'user',
        text: 'What is Phase 3 of the access pattern?'
      },
      {
        id: 'a1',
        role: 'assistant',
        text: 'Phase 3 composes a tuple from fields.'
      }
    ],
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
    retrievedSources: [bplusTreeSource],
    conversationContextMode: 'standalone'
  })

  assert.equal(chatMessages[0].role, 'system')
  assert.match(chatMessages[0].content, /Answer only from PDFs\./)
  assert.match(chatMessages[0].content, /keep the answer scoped/)
  assert.match(chatMessages[0].content, /short paragraphs with one idea each/)
  assert.match(chatMessages[0].content, /include one brief concrete example/)
  assert.equal(chatMessages.length, 2)
  assert.equal(chatMessages.some((message) => /Phase 3/.test(message.content)), false)
  assert.equal(chatMessages.at(-1).role, 'user')
  assert.doesNotMatch(chatMessages.at(-1).content, /Use the context below/)
  assert.match(chatMessages.at(-1).content, /It's a search tree, but it's not a binary tree/)
  assert.match(chatMessages.at(-1).content, /Question: What exactly is a B\+ tree/)
})

test('studyChatMessages uses a resolved contextual prompt without copying raw history', () => {
  const chatMessages = studyChatMessages({
    prompt: 'why is that needed?',
    answerPrompt: 'Previous question: What is pinning a page?\nCurrent question: why is that needed?',
    messages: [
      {
        id: 'u1',
        role: 'user',
        text: 'What is pinning a page?'
      },
      {
        id: 'a1',
        role: 'assistant',
        text: 'The old answer text should not be replayed as a separate model message.'
      }
    ],
    materials: [],
    model: ollamaChatModel,
    settings: {},
    applicationSettings: {
      suggestionMode: 'on',
      followUpSuggestionCount: 4
    },
    modelSettings: {},
    retrievedSources: [pinningSource],
    conversationContextMode: 'contextual'
  })

  assert.equal(chatMessages.length, 2)
  assert.equal(chatMessages[0].role, 'system')
  assert.match(chatMessages[0].content, /Answer directly for a student/)
  assert.equal(chatMessages.at(-1).role, 'user')
  assert.match(chatMessages.at(-1).content, /Previous question: What is pinning a page\?/)
  assert.match(chatMessages.at(-1).content, /Current question: why is that needed\?/)
  assert.doesNotMatch(chatMessages.at(-1).content, /old answer text/)
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
    { role: 'user', content: databaseSuggestionContext },
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
    { role: 'user', content: databaseSuggestionContext },
    { role: 'user', content: 'Generate 2 suggested follow-up questions.\nAsk about adjacent database concepts.' }
  ])
})

test('questionSuggestionMessages uses starter wording for new chats', () => {
  const messages = questionSuggestionMessages(suggestionRequest())
  const prompt = messages.at(-1).content

  assert.deepEqual(messages[0], { role: 'user', content: databaseSuggestionContext })
  assert.match(prompt, /first questions a curious undergraduate student/i)
  assert.match(prompt, /serious study session/i)
  assert.match(prompt, /Spread the questions across different sections/i)
  assert.match(prompt, /Prefer named concepts, mechanisms, code structures/i)
  assert.match(prompt, /Avoid trivial definition-only questions/i)
  assert.match(prompt, /Avoid bare nouns like page, data, query/i)
  assert.doesNotMatch(prompt, /very short factual/i)
  assert.doesNotMatch(prompt, /cannot be found/i)
  assert.doesNotMatch(prompt, /Answer directly for a student/i)
})

test('questionSuggestionMessages uses follow-up wording once a chat has history', () => {
  const messages = questionSuggestionMessages(suggestionRequest({
    messages: [
      { id: 'u1', role: 'user', text: 'What is atomicity?' },
      { id: 'a1', role: 'assistant', text: 'Atomicity means all-or-nothing execution.' }
    ]
  }))
  const prompt = messages.at(-1).content

  assert.match(prompt, /natural next questions a curious undergraduate student/i)
  assert.match(prompt, /after this answer/i)
  assert.match(prompt, /concrete phrase, mechanism, trade-off, or claim/i)
  assert.match(prompt, /latest answer the student just saw/i)
  assert.match(prompt, /Keep each question short/i)
  assert.doesNotMatch(prompt, /first questions/)
  assert.doesNotMatch(prompt, /What part of this usually confuses people/)
})

test('followUpSuggestionMessages uses only the current question and latest answer', () => {
  const messages = followUpSuggestionMessages({
    prompt: "Isn't contention also a problem in 2Q?",
    answerPrompt: 'Previous question: Why does an operating system use LRU?\nCurrent question: Is contention also a problem?',
    messages: [
      { id: 'u1', role: 'user', text: 'Why does an operating system use LRU?' },
      { id: 'a1', role: 'assistant', text: 'Older answer about operating systems.' }
    ],
    materials: [],
    model: ollamaChatModel,
    settings: {},
    applicationSettings: {
      suggestionMode: 'on',
      followUpSuggestionCount: 4
    },
    modelSettings: {
      systemMessage: 'Stay grounded.'
    },
    retrievedSources: [
      {
        title: 'BuzzDBBook',
        locator: 'Section 6.5',
        excerpt: 'MRU and scans are discussed elsewhere.',
        context: 'MRU and scans are discussed elsewhere.'
      }
    ],
    conversationContextMode: 'contextual'
  }, 'No. The answer explains 2Q using FIFO and LRU lists.')

  assert.equal(messages.length, 2)
  assert.deepEqual(messages[0], { role: 'system', content: 'Stay grounded.' })
  assert.match(messages[1].content, /Current question:\nIsn't contention also a problem in 2Q\?/)
  assert.match(messages[1].content, /Latest answer:\nNo\. The answer explains 2Q using FIFO and LRU lists\./)
  assert.doesNotMatch(messages[1].content, /Previous question/)
  assert.doesNotMatch(messages[1].content, /MRU and scans/)
})

test('suggestionPromptFor treats saved built-in prompts as defaults', () => {
  const oldPrompt = 'Suggest {count} very short factual follow-up questions that have not been answered yet or cannot be found inspired by the previous conversation and excerpts.'

  assert.match(suggestionPromptFor({ suggestedFollowUpPrompt: oldPrompt }, 'starter'), /first questions/)
  assert.match(suggestionPromptFor({ suggestedFollowUpPrompt: oldPrompt }, 'followUp'), /natural next questions/)
  assert.equal(suggestionPromptFor({ suggestedFollowUpPrompt: 'Ask friendly questions.' }, 'starter'), 'Ask friendly questions.')
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

  assert.deepEqual(
    parseFollowUpSuggestions([
      'What happens when the log fills up?',
      'How does recovery behave after a crash? Does it replay every record?',
      'Can you show a tiny transaction example?'
    ].join('\n'), 4),
    [
      'What happens when the log fills up?',
      'How does recovery behave after a crash?',
      'Can you show a tiny transaction example?'
    ]
  )

  assert.deepEqual(
    parseFollowUpSuggestions([
      'What problem does 2Q solve in the context of sequential flooding?',
      'From the provided source, what is a dirty page?',
      'What kind of performance trade-offs can we expect in terms of hit rate and cache size if we increase the number of pages in the buffer pool for 2Q?'
    ].join('\n'), 4),
    ['What problem does 2Q solve in the context of sequential flooding?']
  )

  assert.deepEqual(
    parseFollowUpSuggestions([
      'Based on the context, what is atomicity?',
      'What part of this usually confuses people?',
      'Can you think of another recovery scenario?',
      'Can you show a tiny transaction example?',
      'Can you show a tiny transaction example?'
    ].join('\n'), 4),
    ['Can you show a tiny transaction example?']
  )
})

test('suggestions contain only model questions, with no padding for short or empty output', () => {
  assert.deepEqual(filterSuggestedQuestions(parseFollowUpSuggestions('[]', 8), [], 4), [])
  assert.deepEqual(filterSuggestedQuestions(parseFollowUpSuggestions('I cannot suggest a question.', 8), [], 4), [])
  const modelText = '["Why must lookups wait during rehashing?"]'
  assert.deepEqual(filterSuggestedQuestions(parseFollowUpSuggestions(modelText, 8), [], 4),
    ['Why must lookups wait during rehashing?'])
  assert.deepEqual(filterSuggestedQuestions(['Why must lookups wait during rehashing?'], [], 0), [])
  assert.deepEqual(parseFollowUpSuggestions(modelText, 0), [])
})

test('filterSuggestedQuestions removes repeated current and recent questions', () => {
  assert.deepEqual(
    filterSuggestedQuestions([
      'What is an example of a real-world application where LRU would significantly outperform 2Q?',
      'How does 2Q handle cache misses during large scans?',
      'What trade-off does 2Q make compared with LRU?'
    ], [
      'What is an example of a real-world application where 2Q would significantly outperform LRU?'
    ], 4),
    [
      'How does 2Q handle cache misses during large scans?',
      'What trade-off does 2Q make compared with LRU?'
    ]
  )
})
