import test from 'node:test'
import assert from 'node:assert/strict'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { practiceMastery, updateMastery, chapterMastery, removePracticeHistory } = requireTranspiledTs('src/shared/practice-mastery.ts')
const { practiceConceptKey, exportPractice } = requireTranspiledTs('src/shared/practice-study.ts')
const { normalizePracticeState } = requireTranspiledTs('src/shared/practice.ts')
const concept = { courseId: 'book', id: 'trees', title: 'Trees' }
const key = practiceConceptKey(concept)
function session(id, correct = true, options = {}) {
  id = String(id)
  const c = options.concept ?? concept
  return { id, title: c.title, createdAt: '2026-11-10T12:00:00Z', documents: [], modelId: 'fixture', modelSettings: { contextLength: 8192 }, currentIndex: 0,
    topic: { title: c.title, concept: c }, questions: [{ id: 'q1', text: `Claim ${id} is true.`, format: 'true_false', correctAnswer: true,
      concept: c, masteryCheck: { version: 1, conceptKey: practiceConceptKey(c) }, sources: [], draft: '', reviewing: false,
      rubric: { objective: 'A distinct claim.', assumptions: [], criteria: [{ id: 'c1', description: 'Understand the claim.', evidence: [{ sourceKey: 'fixture', quote: `Evidence for ${id}.` }] }] },
      attempts: [{ id: `${id}:first`, answer: correct ? 'True' : 'False', feedback: { kind: 'true_false', correct, explanation: 'Checked.' },
        assisted: options.assisted ?? false, observationVersion: 1, submittedAt: options.at ?? `2026-11-10T12:00:${String(id).padStart(2, '0')}Z`, durationMs: 1000 }] }] }
}
const q = s => s.questions[0]
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-10, `${a} differs from ${b}`)

test('source-view assistance flags are repaired without changing answers or genuine help', () => {
  const reading = session(1, true, { assisted: true })
  const help = session(2, true, { assisted: true })
  for (const s of [reading, help]) {
    s.observations = [{ id: `source-read:${s.id}`, questionId: 'q1', kind: 'help_shown', at: s.createdAt }]
    q(s).attempts[0].assistanceIds = [`source-read:${s.id}`]
  }
  help.observations.push({ id: 'help:chat', questionId: 'q1', kind: 'help_shown', at: help.createdAt })
  q(help).attempts[0].assistanceIds.push('help:chat')
  const before = structuredClone([reading, help])
  const state = normalizePracticeState({ mode: 'practice', sessions: [reading, help] })
  assert.equal(q(state.sessions[0]).attempts[0].assisted, false)
  assert.equal(q(state.sessions[0]).attempts[0].answer, 'True')
  assert.deepEqual(state.sessions[0].observations, [])
  assert.equal(q(state.sessions[1]).attempts[0].assisted, true)
  assert.deepEqual(q(state.sessions[1]).attempts[0].assistanceIds, ['help:chat'])
  const mastery = practiceMastery(state.sessions).get(key)
  assert.equal(mastery.answers, 1)
  assert.equal(mastery.withHelp, 1)
  assert.deepEqual(normalizePracticeState(state), state)
  assert.deepEqual([reading, help], before)
})

test('source flag repair preserves genuine help recorded by an older version', () => {
  const s = session(1, true, { assisted: true })
  q(s).hintUsed = true
  q(s).attempts[0].assistanceIds = ['source-read:1']
  const restored = normalizePracticeState({ mode: 'practice', sessions: [s] }).sessions[0]
  assert.equal(q(restored).attempts[0].assisted, true)
  assert.equal(practiceMastery([restored]).get(key).answers, 0)
})

test('mastery reproduces the proposed answer sequence and remains unknown until evidence exists', () => {
  assert.equal(practiceMastery([]).size, 0)
  const sessions = []
  const expected = [0.3103448275862069, 0.43302992261417494, 0.12019159405608854, 0.2071159619143899, 0.31886317807040093]
  for (const [i, correct] of [true, true, false, true, true].entries()) {
    sessions.push(session(i, correct))
    const result = practiceMastery(sessions).get(key)
    close(result.score, expected[i]); assert.equal(result.answers, i + 1)
  }
})

test('five and ten independent successes reach the documented estimates, never 100%', () => {
  for (const [n, expected] of [[5, 0.7351188224118086], [10, 0.9199808529464086]]) {
    close(practiceMastery(Array.from({ length: n }, (_, i) => session(i))).get(key).score, expected)
  }
  for (const correct of [true, false]) {
    let e = 0, score
    for (let i = 0; i < 10000; i++) ({ evidence: e, score } = updateMastery(e, correct))
    assert.ok(Number.isFinite(score) && score > 0 && score < 1)
  }
})

test('old failures lose influence as new independent answers demonstrate improvement', () => {
  let evidence = 0
  for (let i = 0; i < 30; i++) evidence = updateMastery(evidence, false).evidence
  let score
  for (let i = 0; i < 30; i++) ({ evidence, score } = updateMastery(evidence, true))
  assert.ok(score > .95)
})

test('help and revisions neither update nor decay mastery and cannot erase an independent mistake', () => {
  const wrong = session(0, false), help = session(1, true, { assisted: true })
  q(wrong).attempts.push({ ...q(wrong).attempts[0], id: 'revision', feedback: { kind: 'true_false', correct: true } })
  const result = practiceMastery([wrong, help]).get(key)
  close(result.score, practiceMastery([wrong]).get(key).score)
  assert.equal(result.answers, 1); assert.equal(result.withHelp, 1); assert.equal(result.recentMistake, true)
  assert.equal(practiceMastery([help]).get(key).score, null)
})

test('reused claims and normalized duplicate statements do not inflate mastery across sessions', () => {
  const first = session(0), paraphrase = session(1), repeated = session(2)
  q(paraphrase).rubric = structuredClone(q(first).rubric)
  q(repeated).text = 'CLAIM 0 IS TRUE!'
  const row = practiceMastery([first, paraphrase, repeated, structuredClone(first)]).get(key)
  assert.equal(row.answers, 1); assert.equal(row.repeated, 2)
  close(row.score, practiceMastery([first]).get(key).score)
  q(first).attempts[0].assisted = true
  assert.equal(practiceMastery([first, paraphrase]).get(key).score, null, 'A repeated claim after seeing help is not independent')
})

test('collection and topic identity isolate scores, regardless of matching labels', () => {
  const other = { ...concept, courseId: 'other-book' }, revised = { ...concept, id: 'different-topic' }
  const rows = practiceMastery([session(0), session(1, false, { concept: other }), session(2, false, { concept: revised })])
  assert.equal(rows.size, 3)
  close(rows.get(key).score, .3103448275862069)
  assert.ok(rows.get(practiceConceptKey(other)).score < .1)
})

test('untracked, mismatched, incomplete and unscored responses never earn mastery', () => {
  for (const change of [s => delete q(s).masteryCheck, s => q(s).masteryCheck.version = 2,
    s => q(s).masteryCheck.conceptKey = 'wrong-topic', s => q(s).format = 'short_answer',
    s => q(s).rubric = undefined, s => q(s).attempts = [],
    s => delete q(s).attempts[0].observationVersion, s => q(s).attempts[0].feedback = 'Awaiting review.',
    s => s.assessment = { packId: 'p', version: '1' }]) {
    const s = session(0); change(s); assert.equal(practiceMastery([s]).size, 0)
  }
})

test('single-call questions use the generated key without a hidden wording or verifier gate', () => {
  const s = session(0)
  q(s).text = 'Why do nodes split?'
  q(s).masteryCheck.verification = 'none'
  assert.equal(practiceMastery([s]).get(key).answers, 1)
  assert.equal(exportPractice([s]).sessions[0].questions[0].masteryCheck.verification, 'none')
})

test('parent progress includes unassessed topics as zero and does not double count identities', () => {
  const topics = [concept, { ...concept, id: 'hash' }, { ...concept, id: 'pages' }].map(c => ({ title: c.title, concept: c }))
  const rows = practiceMastery([session(0)])
  const parent = chapterMastery([...topics, topics[0]], rows)
  assert.deepEqual([parent.total, parent.assessed], [3, 1])
  close(parent.score, rows.get(key).score / 3)
  assert.deepEqual(chapterMastery([], rows), { total: 0, assessed: 0, score: 0 })
})

test('replay is chronological, survives save/export/reload, and responds to key correction', () => {
  const a = session(0, false, { at: '2026-11-10T08:00:00-05:00' }), b = session(1, true, { at: '2026-11-10T12:00:00Z' })
  const expected = practiceMastery([b, a])
  assert.deepEqual(practiceMastery([a, b]), expected)
  assert.equal(expected.get(key).recentMistake, true)
  assert.deepEqual(practiceMastery(normalizePracticeState(JSON.parse(JSON.stringify({ mode: 'practice', sessions: [a, b] }))).sessions), expected)
  assert.deepEqual(practiceMastery(exportPractice([a, b]).sessions), expected)
  assert.equal(practiceMastery([b]).get(key).answers, 1)
  const corrections = [{ sessionId: a.id, questionId: q(a).id, keyVersion: '2', correct: true, reason: 'Reviewed correction' }]
  assert.ok(practiceMastery([a, b], corrections).get(key).score > expected.get(key).score)
  assert.equal(q(a).attempts[0].feedback.correct, false)
})

const masteryOf = state => practiceMastery(state.sessions, [], state.masteryRecords)

test('clearing conversations keeps mastery, assistance, recent mistakes and topic identity after reload', () => {
  const original = { mode: 'practice', activeSessionId: '2', sessions: [session(0), session(1, true, { assisted: true }), session(2, false),
    session(3, true, { concept: { ...concept, courseId: 'another-book' } })] }
  const expected = masteryOf(original)
  const cleared = removePracticeHistory(original)
  assert.deepEqual(cleared.sessions, [])
  assert.equal(cleared.activeSessionId, undefined)
  assert.equal(original.sessions.length, 4, 'Do not mutate the current snapshot before saving')
  assert.deepEqual(masteryOf(cleared), expected)
  const restored = normalizePracticeState(JSON.parse(JSON.stringify(cleared)))
  assert.deepEqual(masteryOf(restored), expected)
  assert.deepEqual(removePracticeHistory(restored), restored, 'Repeated clear is a no-op')
  const serialized = JSON.stringify(restored.masteryRecords)
  assert.doesNotMatch(serialized, /Claim 0|Evidence for|Checked\.|modelSettings|sources|attempts|referenceAnswer/)
})

test('individual deletion preserves chronology even when deleted and retained answers are interleaved', () => {
  let state = { mode: 'practice', activeSessionId: '2', sessions: [session(0, false), session(1), session(2, false), session(3)] }
  const expected = masteryOf(state)
  state = removePracticeHistory(state, '1')
  assert.equal(state.activeSessionId, '2')
  assert.deepEqual(masteryOf(state), expected)
  state = removePracticeHistory(state, '2')
  assert.equal(state.activeSessionId, undefined)
  assert.deepEqual(masteryOf(state), expected)
  state.sessions.push(session(4))
  assert.deepEqual(masteryOf(state), practiceMastery([session(0, false), session(1), session(2, false), session(3), session(4)]))
  assert.deepEqual(masteryOf(removePracticeHistory(state)), masteryOf(state))
})

test('deleted questions and supporting claims cannot earn repeated mastery credit', () => {
  const first = session(0)
  const cleared = removePracticeHistory({ mode: 'practice', sessions: [first] })
  const repeated = session(1), paraphrase = session(2)
  q(repeated).text = 'CLAIM 0 IS TRUE!'
  q(paraphrase).rubric = structuredClone(q(first).rubric)
  const result = masteryOf({ ...cleared, sessions: [first, repeated, paraphrase] }).get(key)
  assert.equal(result.answers, 1, 'Also deduplicate a restored original session against its saved record')
  assert.equal(result.repeated, 2)
  close(result.score, masteryOf(cleared).get(key).score)
})
