import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { classifyAnswerStatus, gradeClaims } = requireTranspiledTs('src/shared/confidence.ts')

// Error budgets for the BuzzDB NLI cases. A "missed low confidence" rates an
// answer that should read Not supported above it; a "wrong low confidence" rates one that
// should be trusted as Not supported; a "false supported" labels an unsupported answer
// Supported. Lower these as the check improves; never raise them silently.
const budgets = { missedLowConfidence: 2, wrongLowConfidence: 0, falseSupported: 0 }

function statusFor(verification) {
  return classifyAnswerStatus(gradeClaims(verification.available ? verification.claims : []))
}

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))]
}

export function runBuzzdbNliBenchmark() {
  const python = process.env.TOKENSMITH_BENCHMARK_PYTHON || 'app_runtime/python/bin/python'
  const run = spawnSync(python, ['tests/benchmarks/buzzdb_nli_verify.py'], {
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
    env: process.env
  })
  assert.equal(run.status, 0, run.stderr)
  const { modelLoadMs, results } = JSON.parse(run.stdout)

  const errors = { missedLowConfidence: [], wrongLowConfidence: [], falseSupported: [] }
  const byKind = {}
  for (const result of results) {
    const status = statusFor(result.verification)
    const label = `${result.caseId}/${result.kind} -> ${status}`
    const kind = (byKind[result.kind] ??= { total: 0, correct: 0 })
    let correct = true

    if (result.expect === 'abstain' && status !== 'unsupported') {
      errors.missedLowConfidence.push(label)
      correct = false
    }
    if (result.expect === 'answer' && status === 'unsupported') {
      errors.wrongLowConfidence.push(label)
      correct = false
    }
    if (result.expect !== 'answer' && status === 'supported') {
      errors.falseSupported.push(label)
      correct = false
    }
    if (result.expect === 'answer' && status === 'unverified') {
      errors.wrongLowConfidence.push(label)
      correct = false
    }

    kind.total += 1
    kind.correct += correct ? 1 : 0
  }

  const durations = results.map((result) => result.durationMs)
  const correct = Object.values(byKind).reduce((total, kind) => total + kind.correct, 0)
  console.log(`NLI grounding: ${correct}/${results.length} answers classified correctly`)
  for (const [kind, { total, correct: right }] of Object.entries(byKind)) {
    console.log(`  ${kind.padEnd(12)} ${right}/${total}`)
  }
  for (const [name, list] of Object.entries(errors)) {
    console.log(`  ${name}: ${list.length}${list.length ? ` (${list.join(', ')})` : ''}`)
  }
  console.log(
    `  latency: model load ${Math.round(modelLoadMs)} ms; per answer median ${Math.round(percentile(durations, 0.5))} ms, ` +
      `p90 ${Math.round(percentile(durations, 0.9))} ms, max ${Math.round(Math.max(...durations))} ms`
  )

  for (const [name, budget] of Object.entries(budgets)) {
    assert.ok(errors[name].length <= budget, `${name}: ${errors[name].length} exceeds budget ${budget}: ${errors[name].join(', ')}`)
  }

  return { name: 'buzzdb-nli', total: results.length, correct, errors }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  runBuzzdbNliBenchmark()
}
