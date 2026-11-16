import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { validateEnvironment, evidenceChunk, corruptRecency, regressionGate, sha256 } from './buzzdb-regression-policy.mjs'

const suite = JSON.parse(readFileSync(new URL('../fixtures/buzzdb/regression/suite.json', import.meta.url)))
const controls = JSON.parse(readFileSync(new URL('../fixtures/buzzdb/regression/judge-controls.json', import.meta.url)))
const book = readFileSync(new URL('../fixtures/buzzdb/buzzdb-book.tokensmith.md', import.meta.url), 'utf8')
const cleanReport = () => ({ completed: true,
  controls: controls.map(c => ({ id: c.id, decision: { verdict: c.expected } })),
  conditions: ['baseline', 'retrieval-front-back-swapped'].map(condition => ({ condition,
    turns: suite.cases.map(c => ({ id: c.id, changedSources: condition === 'baseline' ? 0 : 1,
      decision: { verdict: condition === 'baseline' ? 'correct' : 'incorrect' } })) })) })

test('public fixture, evidence IDs and authored rubrics remain consistent', () => {
  validateEnvironment(suite, book, Object.values(suite.models))
  for (const c of suite.cases) {
    for (const id of c.evidenceChunks) assert.ok(evidenceChunk(book, id))
    for (const fact of c.requiredFacts) assert.ok(c.idealAnswer.includes(fact.referenceQuote))
  }
  for (const c of controls) for (const fact of c.requiredFacts) assert.ok(c.idealAnswer.includes(fact.referenceQuote))
})

test('environment changes cannot silently reuse the baseline', () => {
  const tiny = { bookSha256: sha256('synthetic'), models: { judge: { name: 'local', digest: 'v1' } } }
  assert.throws(() => validateEnvironment(tiny, 'changed', [{ name: 'local', digest: 'v1' }]), /fixture changed/)
  assert.throws(() => validateEnvironment(tiny, 'synthetic', []), /not installed/)
  assert.throws(() => validateEnvironment(tiny, 'synthetic', [{ name: 'local', digest: 'v2' }]), /digest changed/)
})

test('new answer failures and unresolved judgments fail the gate', () => {
  const r = cleanReport(); r.conditions[0].turns[1].decision.verdict = 'incorrect'
  assert.equal(regressionGate(suite, controls, r).passed, false)
  r.conditions[0].turns[1].decision.verdict = 'manual_review'
  assert.equal(regressionGate(suite, controls, r).passed, false)
})

test('known failure stays visible and a genuine improvement is accepted', () => {
  const r = cleanReport()
  assert.deepEqual(regressionGate(suite, controls, r).improvements, ['list-ends'])
  r.conditions[0].turns[0].decision.verdict = 'incorrect'
  const gate = regressionGate(suite, controls, r)
  assert.equal(gate.passed, true); assert.deepEqual(gate.knownFailures, ['list-ends'])
})

test('both false acceptance and false rejection by the judge fail calibration', () => {
  for (const verdict of ['correct', 'incorrect']) {
    const r = cleanReport(); r.controls.find(c => c.decision.verdict === verdict).decision.verdict = verdict === 'correct' ? 'incorrect' : 'correct'
    assert.equal(regressionGate(suite, controls, r).passed, false)
  }
})

test('missing turns, unfinished runs and ineffective faults cannot pass', () => {
  const missing = cleanReport(); missing.conditions[0].turns.pop()
  assert.equal(regressionGate(suite, controls, missing).passed, false)
  const unfinished = cleanReport(); unfinished.completed = false
  assert.equal(regressionGate(suite, controls, unfinished).passed, false)
  const noFault = cleanReport(); noFault.conditions[1].turns.forEach(t => t.changedSources = 0)
  assert.equal(regressionGate(suite, controls, noFault).passed, false)
})

test('recovery under corrupted retrieval is an improvement, not a regression', () => {
  const r = cleanReport(); r.conditions[1].turns.forEach(t => t.decision.verdict = 'correct')
  assert.equal(regressionGate(suite, controls, r).passed, true)
})

test('fault injection changes directional terms without mutating source objects', () => {
  const source = { context: 'Front of list, back of list. Feedback remains unchanged.' }
  const delivered = { ...source, context: corruptRecency(source.context) }
  assert.equal(delivered.context, 'Back of list, front of list. Feedback remains unchanged.')
  assert.equal(source.context, 'Front of list, back of list. Feedback remains unchanged.')
})
