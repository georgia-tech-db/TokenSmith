import { readFileSync } from 'node:fs'
import { evaluateAnswer } from '../experiments/local-eval-judge.mjs'
import { gradeDecision } from '../experiments/local-eval-policy.mjs'
import { evidenceChunk, regressionGate, validateEnvironment } from '../experiments/buzzdb-regression-policy.mjs'

export function loadGradedProfile(installedModels, runtimeVersion) {
  const profile = JSON.parse(readFileSync('tests/fixtures/buzzdb/regression/suite.json'))
  const controls = JSON.parse(readFileSync('tests/fixtures/buzzdb/regression/judge-controls.json'))
  const book = readFileSync('tests/fixtures/buzzdb/buzzdb-book.tokensmith.md', 'utf8')
  validateEnvironment(profile, book, installedModels)
  if (runtimeVersion !== profile.ollamaVersion) throw new Error(`Graded benchmark requires Ollama ${profile.ollamaVersion}; received ${runtimeVersion}.`)
  const turns = profile.cases.map(c => ({
    phase: c.id, question: c.question, referenceAnswer: c.idealAnswer,
    requiredFacts: c.requiredFacts, criteria: c.requiredFacts.map(r => r.requirement),
    keyEvidenceQuote: c.evidenceChunks.map(id => evidenceChunk(book, id)).join('\n\n')
  }))
  const families = ['baseline', 'retrieval-front-back-swapped'].map(condition => ({ id: condition, condition, turns }))
  return { profile, controls, families }
}

export async function judgeAnswer(input, profile, fetchImpl = globalThis.fetch) {
  const settings = profile.settings
  const judgment = await evaluateAnswer(input, async ({ messages, schema }) => {
    if (Math.ceil(messages.map(m => m.content).join('\n').length / 3) + settings.judgeTokens + 256 > settings.contextLength) throw new Error('Judge prompt exceeds the pinned context budget.')
    const response = await fetchImpl('http://127.0.0.1:11434/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, redirect: 'error', signal: AbortSignal.timeout(180_000),
      body: JSON.stringify({ model: profile.models.judge.name, think: false, messages, format: schema, stream: false,
        options: { temperature: 0, seed: settings.seed, num_ctx: settings.contextLength, num_predict: settings.judgeTokens } })
    })
    if (!response.ok) throw new Error(`Local judge failed: HTTP ${response.status}`)
    const raw = await response.json()
    if (raw.done_reason === 'length') throw new Error('Local judge exhausted its output budget.')
    return JSON.parse(raw.message.content)
  })
  return { judgment, decision: gradeDecision(judgment, 1) }
}

export async function reviewLiveAnswer(row, grade) {
  row.reviewStatus = 'graded'
  if (row.status !== 'completed') {
    row.decision = { passed: false, verdict: 'manual_review', score: 0 }
    row.passed = false
    return
  }
  const input = { question: row.question, priorUserQuestions: row.priorUserQuestions,
    idealAnswer: row.referenceAnswer, requiredFacts: row.requiredFacts,
    keyEvidenceQuote: row.keyEvidenceQuote, generatedAnswer: row.answer.text }
  try {
    Object.assign(row, await grade(input))
    row.passed = row.decision.passed
    row.failures = row.passed ? [] : [row.judgment.rationale]
  } catch (error) {
    row.passed = false
    row.decision = { passed: false, verdict: 'manual_review', score: 0 }
    row.failures = [`Grading failed: ${error.message}`]
  }
}

export function liveRegressionGate(grading, suite) {
  return regressionGate(grading.profile, grading.controls, {
    completed: suite.completed, error: suite.error, controls: suite.controls,
    conditions: grading.families.map(family => ({ condition: family.condition,
      turns: suite.cases.filter(row => row.condition === family.condition).map(row => ({
        id: row.phase, decision: row.decision, changedSources: row.changedSources ?? 0
      })) }))
  })
}
