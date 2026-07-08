import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  buildRetrievalContext,
  isContextualFollowUpPrompt,
  mergeChatSources
} = requireTranspiledTs('src/shared/chat-context.ts')

const databaseServerSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 4',
  excerpt: 'Database servers are dedicated multiprocessor computers with memory and RAID storage.',
  path: '/course/db.pdf',
  pageStart: 4,
  chunkRowid: 41
}

const parallelDatabaseSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 5',
  excerpt: 'Parallel databases improve performance by partitioning work across processors.',
  path: '/course/db.pdf',
  pageStart: 5,
  chunkRowid: 42
}

test('isContextualFollowUpPrompt recognizes short follow-up requests without treating fresh questions as follow-ups', () => {
  assert.equal(isContextualFollowUpPrompt('be more specific'), true)
  assert.equal(isContextualFollowUpPrompt('What about memory?'), true)
  assert.equal(isContextualFollowUpPrompt('How does it work?'), true)
  assert.equal(isContextualFollowUpPrompt('What is CC BY-SA?'), false)
})

test('buildRetrievalContext uses the previous completed turn for k=1 follow-ups', () => {
  const context = buildRetrievalContext(
    'be more specific',
    [
      {
        id: 'u1',
        role: 'user',
        text: 'What is the typical hardware configuration of a database server?'
      },
      {
        id: 'a1',
        role: 'assistant',
        text: 'A database server is usually a dedicated multiprocessor computer with generous memory and RAID storage.',
        sources: [databaseServerSource]
      }
    ],
    { turnCount: 1 }
  )

  assert.equal(context.isContextualFollowUp, true)
  assert.match(context.query, /Previous question: What is the typical hardware configuration/)
  assert.match(context.query, /Previous answer: A database server is usually/)
  assert.match(context.query, /Follow-up: be more specific/)
  assert.deepEqual(context.carriedSources, [databaseServerSource])
})

test('buildRetrievalContext leaves standalone questions alone', () => {
  const context = buildRetrievalContext(
    'What is CC BY-SA?',
    [
      {
        id: 'u1',
        role: 'user',
        text: 'What is a database server?'
      },
      {
        id: 'a1',
        role: 'assistant',
        text: 'A database server stores databases.',
        sources: [databaseServerSource]
      }
    ]
  )

  assert.equal(context.isContextualFollowUp, false)
  assert.equal(context.query, 'What is CC BY-SA?')
  assert.deepEqual(context.carriedSources, [])
})

test('mergeChatSources keeps carried context while deduping retrieved results', () => {
  assert.deepEqual(
    mergeChatSources([databaseServerSource], [databaseServerSource, parallelDatabaseSource], 4),
    [databaseServerSource, parallelDatabaseSource]
  )
})
