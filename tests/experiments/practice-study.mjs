// Offline export/replay. Never contacts a model or edits the original observations.
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { createHash, randomUUID, randomInt } from 'node:crypto'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'
const { values } = parseArgs({ options: { input: { type: 'string' }, out: { type: 'string' }, corrections: { type: 'string' } } })
if (!values.input || !values.out) throw new Error('--input export.json --out NEW_DIRECTORY are required')
const raw = readFileSync(values.input, 'utf8'), data = JSON.parse(raw)
if (data.version !== 1 || !Array.isArray(data.sessions)) throw new Error('Expected a version 1 Practice export.')
const corrections = values.corrections ? JSON.parse(readFileSync(values.corrections, 'utf8')) : []
if (!Array.isArray(corrections) || corrections.some(c => typeof c.correct !== 'boolean' || !c.keyVersion || !c.reason ||
    !data.sessions.some(s => s.id === c.sessionId && s.questions.some(q => q.id === c.questionId)))) throw new Error('Corrections must name an existing question, a key version, a boolean score, and a reason.')
const { practicePerformance } = requireTranspiledTs('src/shared/practice-study.ts')
const out = resolve(values.out)
mkdirSync(out, { recursive: true })
if (readdirSync(out).length) throw new Error('Choose an empty output directory.')
const write = (name, value) => writeFileSync(resolve(out, name), JSON.stringify(value, null, 2)+'\n')
const rows = data.sessions.flatMap(s => s.questions.map(q => ({ s, q })))
for (let i = rows.length - 1; i > 0; i--) { const j = randomInt(i + 1); [rows[i], rows[j]] = [rows[j], rows[i]] }
const key = [], packet = []
for (const { s, q } of rows) {
  const reviewId = randomUUID()
  packet.push({ reviewId, question: q.text, criteria: q.rubric?.criteria.map(c => c.description) ?? [],
    response: q.attempts[0]?.answer ?? null })
  key.push({ reviewId, sessionId: s.id, questionId: q.id, modelId: s.modelId, concept: q.concept,
    item: q.item, correctAnswer: q.correctAnswer, referenceAnswer: q.referenceAnswer })
}
write('review.json', packet); write('key.json', key)
write('summary.json', { version: 1, sessions: data.sessions.length, questions: rows.length,
  unanswered: rows.filter(r => !r.q.attempts.length).length,
  timingUnknownOrIncomplete: rows.filter(r => !r.q.attempts[0]?.timingComplete).length,
  corrections, estimates: Object.fromEntries(['first_response', 'recency', 'assistance'].map(method => [method, practicePerformance(data.sessions, method, corrections)])) })
write('manifest.json', { version: 1, inputSha256: createHash('sha256').update(raw).digest('hex'),
  estimatorSha256: createHash('sha256').update(readFileSync('src/shared/practice-study.ts')).digest('hex'),
  note: 'Blinding removes identifiers, model settings and prior grades, not clues inherent in question/response text. Keep key.json separate from reviewers.' })
console.log(out)
