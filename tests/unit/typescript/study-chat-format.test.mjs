import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  answerWithOrderedSources,
  questionSuggestionMessages,
  sourceContext
} = requireTranspiledTs('src/main/engine/study-chat-format.ts')

test('sourceContext avoids citation-shaped excerpt labels', () => {
  const context = sourceContext([
    {
      title: 'Annita Demetriou - Wikipedia.pdf',
      locator: 'Page 1',
      excerpt: 'Annita Demetriou is a Cypriot politician.'
    }
  ])

  assert.match(context, /TokenSmith shows evidence separately/)
  assert.match(context, /PDF: Annita Demetriou - Wikipedia\.pdf \(Page 1\)/)
  assert.doesNotMatch(context, /Source 1:/)
  assert.doesNotMatch(context, /Excerpt:/)
  assert.doesNotMatch(context, /source numbers|excerpt labels/i)
})

test('answerWithOrderedSources removes citation-label prose and moves cited source first', () => {
  const sources = [
    { title: 'Unrelated 1', locator: 'Page 1', excerpt: 'One.' },
    { title: 'Unrelated 2', locator: 'Page 2', excerpt: 'Two.' },
    { title: 'Unrelated 3', locator: 'Page 3', excerpt: 'Three.' },
    {
      title: 'Annita Demetriou - Wikipedia.pdf',
      locator: 'Page 1',
      excerpt: 'Annita Demetriou is a Cypriot politician.'
    }
  ]

  const answer = answerWithOrderedSources(
    'According to Excerpt 4, Annita Demetriou is a Cypriot politician. The question does not provide information about source numbers or excerpts labelled "according to Source 1".',
    sources
  )

  assert.equal(answer.text, 'Annita Demetriou is a Cypriot politician.')
  assert.equal(answer.sources[0].title, 'Annita Demetriou - Wikipedia.pdf')
  assert.deepEqual(answer.sources.slice(1).map((source) => source.title), [
    'Unrelated 1',
    'Unrelated 2',
    'Unrelated 3'
  ])
})

test('questionSuggestionMessages uses the configured follow-up prompt for first-turn questions', () => {
  const messages = questionSuggestionMessages({
    messages: [],
    materials: [
      {
        id: 'material-1',
        title: 'Database Systems.pdf',
        detail: 'Ready',
        status: 'ready',
        kind: 'document',
        addedAt: '2026-07-07T00:00:00.000Z',
        pageCount: 12,
        chunkCount: 30
      }
    ],
    model: {
      id: 'ollama:llama3',
      name: 'Ollama llama3',
      engine: 'ollama',
      role: 'generator',
      status: 'ready',
      source: 'ollama',
      ollamaModelName: 'llama3',
      addedAt: '2026-07-07T00:00:00.000Z'
    },
    settings: {},
    applicationSettings: {
      suggestionMode: 'on',
      followUpSuggestionCount: 4
    },
    modelSettings: {
      suggestedFollowUpPrompt: 'Use my shared question prompt for {count} questions.'
    }
  })

  const userContent = messages.at(-1).content

  assert.match(userContent, /has not asked a first question yet/i)
  assert.match(userContent, /Database Systems \(12 pages, 30 chunks\)/)
  assert.match(userContent, /Use my shared question prompt for 4 questions\./)
})
