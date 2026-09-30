import assert from 'node:assert/strict'
import test from 'node:test'
import { utsRows, utsSaveMessage, utsRecallQuestion, utsUnrelatedQuestion } from '../../helpers/uts-fixture.mjs'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { messagesForModel, pinAfterHistoryChange, resolveRunningExample, runningExampleSavedText, saveRunningExample } =
  requireTranspiledTs('src/renderer/src/chat-interactions.ts')
const { studyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')

const model = { id: 't', name: 't', engine: 'ollama', source: 'ollama', ollamaModelName: 't', status: 'ready', contextLength: 8192 }
const classroom = { title: 'Book', locator: 'p9', excerpt: 'classroom(building, room_number, capacity) and branch(customer) are example relations.' }
const allRowValues = Object.values(utsRows).flat().map((row) => JSON.stringify(row).slice(1, -1))

const saved = saveRunningExample(utsSaveMessage, { user: 'uq', assistant: 'ua' })
const chat = [...saved.messages]

test('saving stores the text verbatim with a confirmation and no model input', () => {
  assert.equal(saved.messages.length, 2)
  assert.deepEqual(saved.pin, { messageId: 'uq' })
  assert.equal(saved.messages[0].text, utsSaveMessage.trim())
  assert.equal(saved.messages[0].kind, 'runningExample')
  assert.equal(saved.messages[1].role, 'assistant')
  assert.equal(saved.messages[1].text, runningExampleSavedText)
  assert.equal(saved.messages[1].kind, 'runningExampleSaved')
})

test('saved examples are hidden from the rewriter history but still resolve', () => {
  const ordinary = [{ id: 'q', role: 'user', text: 'Hi' }, { id: 'a', role: 'assistant', text: 'Hello' }]
  assert.deepEqual(messagesForModel([...ordinary, ...saved.messages]), ordinary)
  assert.deepEqual(messagesForModel(chat), [])
  assert.equal(resolveRunningExample(chat, saved.pin), utsSaveMessage.trim())
})

test('the pin fails closed for other conversations, tampering, and edits', () => {
  const other = [{ id: 'x1', role: 'user', text: 'Hi' }, { id: 'x2', role: 'assistant', text: 'Hello' }]
  assert.equal(resolveRunningExample(other, saved.pin), undefined)
  assert.equal(resolveRunningExample(chat), undefined)
  assert.equal(resolveRunningExample([{ ...chat[0], kind: undefined }, chat[1]], saved.pin), undefined)
  assert.equal(resolveRunningExample([{ ...chat[0], text: ' ' }], saved.pin), undefined)
  assert.equal(resolveRunningExample([chat[1]], saved.pin), undefined)
  assert.equal(pinAfterHistoryChange(saved.pin, chat), saved.pin)
  assert.equal(pinAfterHistoryChange(saved.pin, chat, 'uq'), undefined)
  assert.equal(pinAfterHistoryChange(saved.pin, [], 'earlier'), undefined)
  assert.equal(pinAfterHistoryChange(undefined, chat), undefined)
})

// Mirrors submitPrompt: hide saved examples from the rewriter, retrieve, then attach the resolved pin.
async function turn(prompt, mode, pin, messages = chat) {
  let searched
  const prepared = await prepareRewrittenStudyChat({
    prompt, messages: messagesForModel(messages), materials: [], model, settings: {},
    modelSettings: { contextLength: 8192, maxLength: 512 }
  }, {
    resolve: async () => ({ mode, query: prompt, clarification: '' }),
    search: async (query) => { searched = query; return [classroom] }
  })
  const request = { ...prepared.request, pinnedRunningExample: resolveRunningExample(messages, pin) }
  return { text: studyChatMessages(request).at(-1).content, searched, prepared }
}

test('UTS fixture has nine consistent rows with numeric foreign keys', () => {
  assert.equal(allRowValues.length, 9)
  const deptIds = new Set(utsRows.Department.map((row) => row[0]))
  assert.ok(utsRows.Course.every((row) => deptIds.has(row[3])))
  assert.ok(utsRows.Instructor.every((row) => deptIds.has(row[2])))
  for (const value of allRowValues) assert.ok(utsSaveMessage.includes(value), value)
})

test('UTS baseline: without the pin the recall turn loses the example', async () => {
  const { text, searched } = await turn(utsRecallQuestion, 'standalone')
  assert.equal(searched, utsRecallQuestion)
  assert.ok(!text.includes('University of TokenSmith') && !text.includes('BUS301'))
  assert.ok(text.includes('classroom'))
})

test('saved-example turns are not a previous exchange, so the first real question needs no rewrite', async () => {
  let resolved = 0
  const prepared = await prepareRewrittenStudyChat({
    prompt: utsRecallQuestion, messages: messagesForModel(chat), materials: [], model, settings: {}
  }, { resolve: async () => { resolved += 1; return { mode: 'contextual', query: 'x', clarification: '' } }, search: async () => [] })
  assert.equal(resolved, 0)
  assert.equal(prepared.request.referenceExchange, undefined)
})

for (const mode of ['standalone', 'contextual']) {
  test(`UTS pinned recall (${mode}) carries the full example once, without changing retrieval`, async () => {
    const history = [...chat, { id: 'q2', role: 'user', text: 'What is a key?' }, { id: 'a2', role: 'assistant', text: 'A key identifies rows.' }]
    const { text, searched } = await turn(utsRecallQuestion, mode, saved.pin, history)
    assert.equal(searched, utsRecallQuestion)
    assert.equal(text.split('### Pinned running example').length, 2)
    for (const value of allRowValues) assert.equal(text.split(value).length, 2, `${value} appears exactly once`)
    for (const key of ['dept_id PRIMARY KEY', 'course_id PRIMARY KEY', 'instructor_id PRIMARY KEY',
      'Course.dept_id      REFERENCES Department.dept_id', 'Instructor.dept_id  REFERENCES Department.dept_id']) assert.ok(text.includes(key), key)
    assert.ok(!text.includes(runningExampleSavedText))
    assert.ok(text.indexOf('### Pinned running example') < text.indexOf('### Source unit'))
    assert.ok(text.endsWith(`Question: ${utsRecallQuestion}`))
  })
}

test('UTS unrelated control keeps normal retrieval and does not make UTS textbook evidence', async () => {
  const { text, searched } = await turn(utsUnrelatedQuestion, 'standalone', saved.pin)
  assert.equal(searched, utsUnrelatedQuestion)
  assert.match(text, /For unrelated questions, ignore it/)
  assert.match(text, /never cite the pinned text as a source/)
})

test('no pin leaves the prompt unchanged; selected-passage and simplification ignore it', () => {
  const base = { prompt: 'Q?', messages: [], materials: [], model, settings: {}, modelSettings: { contextLength: 8192, maxLength: 512 }, retrievedSources: [classroom] }
  const text = (extra) => studyChatMessages({ ...base, ...extra }).at(-1).content
  assert.ok(!text({}).includes('Pinned running example'))
  assert.equal(text({ pinnedRunningExample: undefined }), text({}))
  const passage = { messageId: 'a', role: 'assistant', text: 'rows', question: 'Q' }
  assert.ok(!text({ selectedPassage: passage, pinnedRunningExample: utsSaveMessage }).includes('Pinned running example'))
  assert.ok(!text({ answerToSimplify: 'old', pinnedRunningExample: utsSaveMessage }).includes('Pinned running example'))
})

test('an oversized pin is never truncated and fails with a clear error', () => {
  assert.throws(() => studyChatMessages({
    prompt: 'Q?', messages: [], materials: [], settings: {}, retrievedSources: [classroom],
    pinnedRunningExample: 'row '.repeat(4000), model: { ...model, contextLength: 2048 },
    modelSettings: { contextLength: 2048, maxLength: 512 }
  }), /Pinned example is too large/)
})
