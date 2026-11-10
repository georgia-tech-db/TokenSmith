import test from 'node:test'
import assert from 'node:assert/strict'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { practicePerformance, practiceConceptKey } = requireTranspiledTs('src/shared/practice-study.ts')
const { normalizePracticeState } = requireTranspiledTs('src/shared/practice.ts')

function session(id, correct, { course = 'book', topic = 'trees', title = 'Trees', assisted = false,
  at = '2026-11-10T12:00:00Z' } = {}) {
  return { id, title, createdAt: at, documents: [], modelId: 'fixture', modelSettings: { contextLength: 8192 }, currentIndex: 0,
    questions: [{ id: 'q1', text: 'An internal node stores separator keys.', format: 'true_false', correctAnswer: true,
      concept: { courseId: course, id: topic, title }, sources: [], draft: '', reviewing: false,
      attempts: [{ id: `${id}:first`, answer: correct ? 'True' : 'False', feedback: { kind: 'true_false', correct, explanation: 'Fixture feedback.' },
        assisted, submittedAt: at, observationVersion: 1, durationMs: 1000 }] }] }
}
const first = s => s.questions[0].attempts[0]

test('topic progress combines first answers across sessions and retains raw counts for every estimator', () => {
  const a = session('a', false, { at: '2026-11-09T12:00:00Z' })
  const b = session('b', true, { assisted: true })
  const before = JSON.stringify([a, b])
  for (const method of ['first_response', 'recency', 'assistance']) {
    const [row] = practicePerformance([b, a], method)
    assert.equal(row.observations, 2)
    assert.equal(row.correctResponses, 1)
    assert.equal(row.assisted, 1)
    assert.equal(row.lastPracticedAt, first(b).submittedAt)
    assert.equal(row.recentMistake, true, 'An assisted success does not clear an independent mistake')
  }
  assert.equal(JSON.stringify([a, b]), before)
})

test('revisions and duplicate saved question records do not inflate progress', () => {
  const s = session('a', false)
  s.questions[0].attempts.push({ ...first(s), id: 'revision', feedback: { kind: 'true_false', correct: true }, assisted: true })
  s.questions.push(structuredClone(s.questions[0]))
  const [row] = practicePerformance([s, structuredClone(s)])
  assert.equal(row.observations, 1)
  assert.equal(row.correctResponses, 0)
  assert.equal(row.assisted, 0)
  assert.equal(row.recentMistake, true)
})

test('the same item answered in another session is a new first response', () => {
  const a = session('a', false), b = session('b', true)
  a.questions[0].item = b.questions[0].item = { id: 'shared-item', version: '1', keyVersion: '1', origin: 'reviewed' }
  assert.equal(practicePerformance([a, b])[0].observations, 2)
})

test('matching titles and local topic IDs never join different collections or different topics', () => {
  const rows = practicePerformance([session('a', true), session('b', false, { course: 'other-book' }), session('c', false, { topic: 'new-trees' })])
  const map = new Map(rows.map(r => [practiceConceptKey(r.concept), r]))
  assert.equal(map.size, 3)
  assert.equal(map.get(practiceConceptKey({ courseId: 'book', id: 'trees' })).correctResponses, 1)
  assert.equal(map.get(practiceConceptKey({ courseId: 'other-book', id: 'trees' })).correctResponses, 0)
  assert.equal(map.get(practiceConceptKey({ courseId: 'book', id: 'new-trees' })).correctResponses, 0)
  assert.equal(map.has(practiceConceptKey({ courseId: 'book', id: 'regenerated-trees' })), false)
})

test('renaming a topic with the same identity preserves results without joining on title', () => {
  const rows = practicePerformance([session('a', true), session('b', false, { title: 'Tree insertion' })])
  assert.equal(rows.length, 1)
  assert.equal(rows[0].observations, 2)
})

test('unscored, invalid, untracked and incomplete assessment answers do not create progress', () => {
  const changes = [
    s => { s.questions[0].attempts = [] },
    s => { first(s).feedback = 'Awaiting independent review.' },
    s => { first(s).feedback.correct = 'true' },
    s => { delete first(s).submittedAt },
    s => { first(s).submittedAt = 'invalid-date' },
    s => { delete first(s).observationVersion },
    s => { delete s.questions[0].concept },
    s => { s.questions[0].text = 'Why does a node split?' },
    s => { s.assessment = { packId: 'pack', version: '1' } },
    s => { s.questions[0].format = 'short_answer'; first(s).feedback = { verdict: 'correct', checks: [], improvement: '', nextStep: '', questionIssue: '' } }
  ]
  for (const change of changes) {
    const s = session('a', true); change(s)
    assert.deepEqual(practicePerformance([s]), [])
  }
  const reviewed = session('b', true)
  reviewed.assessment = { packId: 'pack', version: '1', completedAt: '2026-11-10T14:00:00Z' }
  assert.equal(practicePerformance([reviewed])[0].observations, 1)
})

test('recent mistakes follow chronological independent answers, not array or timezone string order', () => {
  const mistake = session('a', false, { at: '2026-11-10T08:00:00-05:00' })
  const earlierSuccess = session('b', true, { at: '2026-11-10T12:00:00Z' })
  const assisted = session('c', true, { assisted: true, at: '2026-11-10T14:00:00Z' })
  assert.equal(practicePerformance([assisted, mistake, earlierSuccess])[0].recentMistake, true)
  const recovered = session('d', true, { at: '2026-11-10T15:00:00Z' })
  assert.equal(practicePerformance([recovered, assisted, mistake, earlierSuccess])[0].recentMistake, false)
  assert.equal(practicePerformance([session('help-only', false, { assisted: true })])[0].recentMistake, false)
})

test('progress reflects deleted sessions and survives the app save/reload normalization', () => {
  const sessions = [session('a', false), session('b', true, { assisted: true })]
  const saved = JSON.parse(JSON.stringify({ mode: 'practice', sessions }))
  const restored = normalizePracticeState(saved)
  assert.deepEqual(practicePerformance(restored.sessions), practicePerformance(sessions))
  assert.equal(practicePerformance(sessions.slice(1))[0].recentMistake, false)
  assert.deepEqual(practicePerformance([]), [])
})

test('review corrections update progress and recent mistakes without modifying attempts', () => {
  const s = session('a', false)
  const [row] = practicePerformance([s], 'first_response', [{ sessionId: 'a', questionId: 'q1', correct: true, keyVersion: '2', reason: 'Reviewed answer key' }])
  assert.equal(row.correctResponses, 1)
  assert.equal(row.recentMistake, false)
  assert.equal(first(s).feedback.correct, false)
})
