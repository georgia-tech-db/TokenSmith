import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { followUpSuggestionMessages, questionSuggestionMessages, filterSuggestedQuestions } =
  requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { legacySuggestionPrompts, defaultSuggestedFollowUpPrompt, normalizeSuggestedFollowUpPrompt } =
  requireTranspiledTs('src/shared/model-defaults.ts')

const request = (engine = 'ollama', overrides = {}) => ({
  prompt: 'How does the buffer manager pick a page to evict?', messages: [], materials: [], settings: {},
  model: { engine, contextLength: 8192 },
  modelSettings: { contextLength: 8192, contextLengthMode: 'manual', maxLength: 1536 },
  applicationSettings: { suggestionMode: 'on', followUpSuggestionCount: 4 },
  retrievedSources: [{ title: 'Buffer manager', excerpt: 'Pinned pages cannot be evicted. Dirty pages must be written back before their frames are reused.' }],
  ...overrides
})

test('follow-ups receive evidence separately from an unsupported answer premise', () => {
  const text = followUpSuggestionMessages(request(), 'FIFO is always better than LRU.')[0].content
  assert.match(text, /Pinned pages cannot be evicted/)
  assert.match(text, /Latest answer:\nFIFO is always better than LRU/)
  assert.match(text, /may contain mistakes; do not build on a premise/)
  assert.match(text, /Worked examples and contrasting conditions may be derived/)
})

for (const engine of ['ollama', 'remote']) {
  test(`${engine}: an 8K follow-up budgets sources, answer and recent questions together`, () => {
    const sources = Array.from({ length: 12 }, (_, i) => ({ title: `Unit ${i}`, excerpt:
      'An unpinned page can be evicted. Dirty pages must be written back before their frame is reused. '.repeat(45) }))
    const messages = followUpSuggestionMessages(request(engine, {
      messages: Array.from({ length: 8 }, (_, i) => ({ role: 'user', text: `Question ${i}: ` +
        'Explain a buffer pool access sequence with pinned pages and repeated requests. '.repeat(6) })),
      retrievedSources: sources
    }), 'The buffer manager checks whether frames are pinned before choosing an eviction victim. '.repeat(48))
    const text = messages.map(m => m.content).join('\n')
    const estimatedTokens = Math.ceil(text.length / (engine === 'ollama' ? 2.5 : 4))
    assert.ok(estimatedTokens + 384 <= 8192, `${estimatedTokens} input tokens must leave output room`)
    assert.ok(text.includes(sources[0].excerpt.trim()), 'Keep included passages intact')
    assert.ok((text.match(/### Source unit/g) || []).length < sources.length, 'Drop passages that do not fit')
    assert.match(text, /Current question:\nHow does the buffer manager/)
    assert.match(text, /Latest answer:\nThe buffer manager/)
  })
}

test('cloud follow-ups respect a separate input limit', () => {
  const messages = followUpSuggestionMessages(request('remote', {
    model: { engine: 'remote', inputTokenLimit: 1800, maxOutputTokens: 256 },
    modelSettings: { contextLengthMode: 'auto', maxLength: 1536 },
    retrievedSources: Array.from({ length: 20 }, (_, i) => ({ title: `Unit ${i}`, excerpt:
      'Pinned pages cannot be evicted while in use. '.repeat(20) }))
  }), 'Pins protect pages while an operator reads them.')
  assert.ok(Math.ceil(messages.map(m => m.content).join('\n').length / 4) <= 1800)
})

test('an oversized question or source fails before sending an ungrounded follow-up request', () => {
  assert.throws(() => followUpSuggestionMessages(request('ollama', { prompt: 'Long question. '.repeat(4000) }), 'Answer.'),
    /too little room for suggested questions/)
  assert.throws(() => followUpSuggestionMessages(request('ollama', {
    retrievedSources: [{ title: 'Oversized unit', excerpt: 'Evidence. '.repeat(5000) }]
  }), 'Answer.'), /No complete source passage fits/)
})

test('general chat still gets follow-ups when no documents were used', () => {
  const text = followUpSuggestionMessages(request('ollama', { retrievedSources: [] }), 'A buffer holds pages.')[0].content
  assert.match(text, /If no excerpts are supplied, avoid assuming undocumented details/)
  assert.doesNotMatch(text, /### Context:/)
})

test('Practice-to-Chat follow-ups keep the visible exchange, false-premise instruction and evidence', () => {
  const text = questionSuggestionMessages(request('ollama', { messages: [
    { role: 'user', text: 'True/False: Pinned pages may be evicted.\nMy answer: True' },
    { role: 'assistant', text: 'Incorrect. Pinning protects pages in use.' }
  ] })).map(m => m.content).join('\n')
  assert.match(text, /My answer: True/)
  assert.match(text, /Incorrect. Pinning protects pages/)
  assert.match(text, /practice statement may be false/)
  assert.match(text, /Pinned pages cannot be evicted/)
  assert.match(text, /may contain mistakes/)
})

test('opposite comparisons and source identifiers are not removed as duplicates or unfamiliar names', () => {
  const suggestions = ['When does an index scan outperform a table scan?',
    'When does a table scan outperform an index scan?', 'How does `page_id` preserve a record reference?']
  assert.deepEqual(filterSuggestedQuestions([...suggestions, suggestions[0]], [], 4), suggestions)
})

test('the shipped answer-only prompt migrates, while a custom prompt survives', () => {
  const previous = legacySuggestionPrompts.find(text => text.includes('Use subjects the answer has already introduced.'))
  assert.ok(previous)
  assert.equal(normalizeSuggestedFollowUpPrompt(previous), defaultSuggestedFollowUpPrompt)
  assert.equal(normalizeSuggestedFollowUpPrompt('Ask about worked examples.'), 'Ask about worked examples.')
})
