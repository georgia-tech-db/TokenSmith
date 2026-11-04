import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { spawnSync } from 'node:child_process'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { parseAssessmentPack, assessmentSession, assessmentLocked, assessmentDisplayQuestion, exportPractice, practicePerformance } = requireTranspiledTs('src/shared/practice-study.ts')
const { normalizePracticeState } = requireTranspiledTs('src/shared/practice.ts')
const { createStateWriter } = requireTranspiledTs('src/main/atomic-state.ts')
const pack = JSON.parse(readFileSync('tests/fixtures/assessment-pack.json', 'utf8'))
const session = () => assessmentSession(parseAssessmentPack(pack), 's', { contextLength: 8192, temperature: 0 }, '2026-11-04T00:00:00Z')

test('pack validation rejects duplicate IDs, bad keys, empty rubrics and unsupported versions', () => {
  for (const change of [p => p.version = 2, p => p.items.push(p.items[0]), p => p.items[0].correctAnswer = 'false',
    p => p.items[1].criteria = [], p => p.items[0].concept.id = '', p => p.items = []]) {
    const p = structuredClone(pack); change(p); assert.throws(() => parseAssessmentPack(p))
  }
  const p = parseAssessmentPack({ ...pack, apiKey: 'secret' }); assert.equal(p.apiKey, undefined)
})
test('assessment display excludes keys, criteria, hints, sources and recorded grades', () => {
  const s = session(), q = s.questions[0]
  q.attempts.push({ id: 'a', answer: 'False', feedback: { kind: 'true_false', correct: true, explanation: 'SECRET_KEY' }, assisted: false, durationMs: 0 })
  assert.equal(assessmentLocked(s), true)
  const display = assessmentDisplayQuestion(q)
  assert.equal(display.correctAnswer, undefined); assert.equal(display.referenceAnswer, undefined); assert.equal(display.rubric, undefined)
  assert.deepEqual(display.sources, []); assert.doesNotMatch(JSON.stringify(display), /SECRET_KEY/)
  assert.equal(display.attempts[0].answer, 'False')
})
test('saved assessments retain completion rules; legacy sessions retain unknown timing and concept', () => {
  const s = session(); const q = s.questions[0]; delete q.concept; delete s.observations
  const normalized = normalizePracticeState({ mode: 'practice', activeSessionId: s.id, sessions: [s] })
  assert.equal(assessmentLocked(normalized.sessions[0]), true)
  assert.equal(normalized.sessions[0].observations, undefined)
  assert.equal(normalized.sessions[0].questions[0].concept, undefined)
})
function answered(course, correct, assisted = false, submittedAt = '2026-11-04T00:01:00Z') {
  const s = session(); s.id = course; s.assessment.completedAt = submittedAt
  s.questions = [s.questions[0]]; s.questions[0].concept.courseId = course
  s.questions[0].attempts = [{ id: course, answer: 'False', feedback: { kind: 'true_false', correct, explanation: '' },
    submittedAt, observationVersion: 1, assisted, durationMs: 2000, visibleMs: 1000 }]
  return s
}
test('performance uses scoped concepts, first responses, and excludes ungraded or incomplete assessments', () => {
  const a = answered('db', true), b = answered('algorithms', false)
  a.questions[0].attempts.push({ ...a.questions[0].attempts[0], id: 'revision', feedback: { kind: 'true_false', correct: false } })
  assert.deepEqual(practicePerformance([a,b]).map(r => [r.concept.courseId,r.performance,r.observations]), [['algorithms',0,1],['db',1,1]])
  delete b.assessment.completedAt; assert.equal(practicePerformance([a,b]).length, 1)
  a.questions[0].attempts[0].feedback = 'Awaiting independent review.'; assert.deepEqual(practicePerformance([a,b]), [])
})
test('assistance weighting and corrections are replayable without mutating observations', () => {
  const a = answered('db', false), b = answered('db', true, true, '2026-11-04T00:02:00Z'); b.id = 'later'
  const before = JSON.stringify([a,b])
  assert.equal(practicePerformance([a,b])[0].performance, .5)
  assert.ok(practicePerformance([a,b], 'assistance')[0].performance < .5)
  const corrected = practicePerformance([a,b], 'first_response', [{ sessionId: a.id, questionId: a.questions[0].id, keyVersion: '2', correct: true, reason: 'Reviewer correction' }])
  assert.equal(corrected[0].performance, 1); assert.equal(JSON.stringify([a,b]), before)
})
test('export excludes credentials and filesystem paths while retaining response and version provenance', () => {
  const s = answered('db', true); s.modelSettings.apiKey = 'SECRET'; s.documents = [{ materialId:'m',documentId:1,title:'Book',path:'/private/student' }]
  const result = exportPractice([s], 'P001')
  assert.equal(result.participantId, 'P001'); assert.doesNotMatch(JSON.stringify(result), /SECRET|\/private\/student/)
  assert.equal(result.sessions[0].questions[0].item.keyVersion, '1')
  assert.equal(result.sessions[0].questions[0].attempts[0].submittedAt, '2026-11-04T00:01:00Z')
})
test('incomplete assessment exports preserve responses but cannot leak keys or rubrics', () => {
  const s = session()
  const q = exportPractice([s]).sessions[0].questions[0]
  assert.equal(q.correctAnswer, undefined); assert.equal(q.referenceAnswer, undefined); assert.equal(q.rubric, undefined)
  assert.deepEqual(q.sources, []); assert.equal(q.text, s.questions[0].text)
})
test('offline review and dry-run comparison retain failures/missing data boundaries without inference', async () => {
  const dir = await mkdtemp(join(tmpdir(),'tokensmith-study-cli-'))
  try {
    const input = join(dir,'export.json'); await writeFile(input, JSON.stringify(exportPractice([answered('db', true)])))
    const run = (script,args) => {
      const result = spawnSync(process.execPath,[script,...args],{encoding:'utf8'})
      assert.equal(result.status,0,result.stderr); return result
    }
    const reviewOut = join(dir,'review')
    run('tests/experiments/practice-study.mjs',['--input',input,'--out',reviewOut])
    const review = JSON.parse(await readFile(join(reviewOut,'review.json'),'utf8'))
    assert.equal(review[0].response,'False'); assert.equal(review[0].modelId,undefined); assert.equal(review[0].correctAnswer,undefined)
    const summary = JSON.parse(await readFile(join(reviewOut,'summary.json'),'utf8'))
    assert.equal(summary.estimates.first_response[0].performance,1)
    const source = { materialId:'m', documentId:1, chunkId:'c', excerpt:'A reviewed source.' }
    const manifest = {version:1,cases:[{id:'case',prompt:'Explain.',evaluatorSecret:'NEVER_SEND',
      model:{engine:'ollama',ollamaModelName:'fixture'},modelSettings:{contextLength:8192,maxLength:1024},
      evidence:{baseline:[source],expanded:[source],reviewed:[source]}}]}
    const cases = join(dir,'cases.json'); await writeFile(cases,JSON.stringify(manifest))
    run('tests/experiments/proposal-comparison.mjs',['--input',cases,'--out',join(dir,'requests'),'--dry-run'])
    const requests = JSON.parse(await readFile(join(dir,'requests/requests.json'),'utf8'))
    assert.equal(requests.length,6); assert.doesNotMatch(JSON.stringify(requests),/NEVER_SEND/)
    assert.deepEqual(new Set(requests.map(r=>r.request.modelSettings.reasoningMode)),new Set(['off','on']))
  } finally { await rm(dir,{recursive:true,force:true}) }
})
test('atomic state saves keep the last complete snapshot and recover from failed writes', async () => {
  const dir = await mkdtemp(join(tmpdir(),'tokensmith-state-test-')), path = join(dir,'state.json'), save = createStateWriter()
  try {
    await Promise.all([save(path, JSON.stringify({ value:'a'.repeat(100000) })), save(path, '{"value":"last"}')])
    assert.deepEqual(JSON.parse(await readFile(path,'utf8')), { value:'last' })
    await writeFile(join(dir,'blocked'), 'file')
    await assert.rejects(save(join(dir,'blocked','state.json'), '{}'))
    await save(path, '{"recovered":true}')
    assert.deepEqual(JSON.parse(await readFile(path,'utf8')), { recovered:true })
  } finally { await rm(dir,{recursive:true,force:true}) }
})
