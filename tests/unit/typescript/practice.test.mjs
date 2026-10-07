import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { studyDocumentKey } = requireTranspiledTs('src/shared/study-scope.ts')
const { emptyPracticeState, normalizePracticeState, practiceSourceKey, practiceAssisted, practiceReference,
  practiceDiscussionPassage, practiceReviewLabel } = requireTranspiledTs('src/shared/practice.ts')
const { parsePracticeQuestion } = requireTranspiledTs('src/shared/quiz.ts')
const { createPracticeClient } = requireTranspiledTs('src/renderer/src/practice-client.ts')

const source = { materialId: 'c1', documentId: 8, chunkId: 'ch.1', sourceUnitId: 'unit.1', excerpt: 'Ancestor paths let insertion propagate splits to parents.' }
const packageData = { question: 'Why keep a path?', objective: 'Explain split propagation.', assumptions: [], criteria: [
  { id: 'c1', description: 'Propagation to parents.', evidence: [{ sourceKey: 'c1:8:unit.1', quote: source.excerpt }] }
], explanation: 'The path lets insertion revisit parents after a split.', hint: 'Consider which node changes next.' }
const question = { id: 'q1', ...parsePracticeQuestion(JSON.stringify(packageData), [source]), draft: 'To revisit ancestors.', attempts: [], reviewing: false }
const feedback = { questionAssessment: 'The question is supported and answerable.', checks: [{ criterionId: 'c1', status: 'met', feedback: 'You identified where updates propagate.' }], improvement: '', nextStep: '', questionIssue: '' }
const model = { id: 'model1', status: 'ready' }
const settings = { application: { suggestionMode: 'on', explanationDepthEnabled: true } }
const session = { id: 's1', documents: [{ materialId: 'c1', documentId: 8 }], modelId: model.id,
  modelSettings: { contextLength: 8192, maxLength: 4096 }, questions: [], currentIndex: 0, totalQuestions: 5 }

function harness(overrides = {}) {
  const calls = []
  const bridge = {
    practiceSources: async (...args) => { calls.push(['sources', ...args]); return [source] },
    sendChatMessage: async request => { calls.push(['model', request]); return {
      text: JSON.stringify(request.practiceTask === 'question' ? packageData : feedback), sources: [source] } },
    ...overrides
  }
  return { calls, client: createPracticeClient(bridge, model, settings) }
}

test('practice state preserves saved attempts, scopes, and old transcripts across reloads', () => {
  assert.deepEqual(normalizePracticeState(), emptyPracticeState())
  assert.deepEqual(normalizePracticeState({ sessions: null }), emptyPracticeState())
  const saved = { mode: 'practice', activeSessionId: session.id, sessions: [session] }
  assert.deepEqual(normalizePracticeState(JSON.parse(JSON.stringify(saved))), saved)
  assert.equal(normalizePracticeState({ ...saved, activeSessionId: 'missing' }).activeSessionId, undefined)
})

test('document and evidence identity includes collection and document', () => {
  assert.notEqual(studyDocumentKey(source), studyDocumentKey({ ...source, materialId: 'c2' }))
  assert.notEqual(studyDocumentKey(source), studyDocumentKey({ ...source, documentId: 9 }))
  assert.equal(practiceSourceKey(source), 'c1:8:unit.1')
})

test('hints, explanations, and revisions are not mistaken for independent attempts', () => {
  assert.equal(practiceAssisted(question), false)
  for (const change of [{ hintUsed: true }, { answerRevealed: true }, { attempts: [{ answer: 'first' }] }]) {
    assert.equal(practiceAssisted({ ...question, ...change }), true)
  }
  assert.equal(practiceReviewLabel(question), 'Not answered')
  assert.equal(practiceReviewLabel({ ...question, answerRevealed: true }), 'Explanation viewed')
  assert.equal(practiceReviewLabel({ ...question, attempts: [{ feedback: { verdict: 'partial' } }] }), 'Revisit this concept')
  assert.equal(practiceReviewLabel({ ...question, attempts: [{ feedback: { verdict: 'correct' }, assisted: true }] }), 'Correct with help or revision')
})

test('discussion uses the submitted answer and feedback, never unrevealed explanations or hints', () => {
  assert.equal(practiceReference(session, question).feedback, '')
  const answered = { ...question, draft: 'Unsubmitted revision', attempts: [{ answer: 'Submitted answer', feedback: 'Actual feedback' }] }
  const reference = practiceReference(session, answered)
  assert.equal(reference.answer, 'Submitted answer')
  assert.deepEqual(reference.sources, [source])
  assert.deepEqual(reference.documents, session.documents)
  assert.deepEqual(practiceDiscussionPassage(reference), { messageId: 'q1', role: 'assistant',
    question: 'Practice question: Why keep a path?\n\nMy answer: Submitted answer', text: 'Actual feedback' })
})

test('one question call prepares the question, saved criteria, hint and explanation', async () => {
  const { calls, client } = harness()
  const result = await client.question({ ...session, questions: [question] }, 'r1', new AbortController().signal)
  assert.deepEqual(calls[0], ['sources', session.documents, ['c1:8:unit.1'], 1])
  const request = calls[1][1]
  assert.equal(request.practiceTask, 'question')
  assert.deepEqual(request.messages, [])
  assert.deepEqual(request.materials, [])
  assert.deepEqual(request.retrievedSources, [source])
  assert.equal(request.applicationSettings.suggestionMode, 'off')
  assert.equal(request.modelSettings.thinking, false)
  assert.equal(request.modelSettings.maxLength, 2048)
  assert.equal(result.text, packageData.question)
  assert.deepEqual(result.rubric, question.rubric)
  assert.equal(result.referenceAnswer, packageData.explanation)
  assert.equal(result.hint, packageData.hint)
  assert.deepEqual(result.attempts, [])
  assert.equal(calls.length, 2)
  assert.equal(client.help, undefined)
})

test('grading reuses saved criteria, evidence, and previous submitted attempt without new retrieval', async () => {
  const { calls, client } = harness()
  const previous = { answer: 'First answer.', feedback: { verdict: 'partial', checks: [], improvement: '', nextStep: 'Explain the path.', questionIssue: '' } }
  const result = await client.feedback(session, { ...question, attempts: [previous] }, 'My revision.', 'grade')
  assert.equal(calls.length, 1)
  const request = calls[0][1]
  assert.equal(request.practiceTask, 'feedback')
  assert.equal(request.modelSettings.maxLength, 1024)
  assert.deepEqual(request.retrievedSources, question.sources)
  const input = JSON.parse(request.prompt)
  assert.deepEqual(input.rubric, question.rubric)
  assert.deepEqual(input.previousAttempt, previous)
  assert.equal(input.answer, 'My revision.')
  assert.equal(result.verdict, 'correct')
})

test('cancellation during passage loading prevents model calls', async () => {
  let release
  const { client, calls } = harness({ practiceSources: () => new Promise(resolve => { release = resolve }) })
  const controller = new AbortController()
  const pending = client.question(session, 'cancel', controller.signal)
  controller.abort(); release([source])
  await assert.rejects(pending, { name: 'AbortError' })
  assert.equal(calls.length, 0)
})

test('invalid model output fails explicitly; older questions never get invented rubrics', async () => {
  const empty = harness({ practiceSources: async () => [] })
  await assert.rejects(empty.client.question(session, 'r', new AbortController().signal), /No unused passages/)
  for (const reply of [{ text: '', sources: [source] }, { text: 'Question?', sources: [] }]) {
    const { client } = harness({ sendChatMessage: async () => reply })
    await assert.rejects(client.question(session, 'r', new AbortController().signal), /valid question/)
    await assert.rejects(client.feedback(session, question, 'answer', 'r'), /usable feedback/)
  }
  const { client, calls } = harness()
  await assert.rejects(client.feedback(session, { ...question, rubric: undefined }, 'answer', 'r'), /older question/)
  await assert.rejects(client.feedback({ ...session, modelId: 'other' }, question, 'answer', 'r'), /unavailable/)
  assert.equal(calls.length, 0)
})
