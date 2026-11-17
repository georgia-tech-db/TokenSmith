import { spawnSync } from 'node:child_process'
import { appendFileSync, mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { runBuzzdbMultiturnBenchmark } from './test_buzzdb_multiturn.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')

function failureSuite(name, label, error) {
  return {
    name,
    label,
    unit: 'checks',
    passed: 0,
    total: 1,
    cases: [{
      id: name,
      passed: false,
      failures: [error instanceof Error ? error.message : String(error)]
    }]
  }
}

function parseJsonOutput(output) {
  const lines = output.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  if (lines.length === 0) {
    throw new Error('Benchmark produced no JSON output.')
  }
  return JSON.parse(lines.at(-1))
}

function prepareGradedEmbeddingCache() {
  if (process.env.TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH) return
  const python = process.env.TOKENSMITH_BENCHMARK_PYTHON ?? 'python3'
  const builder = 'tests/benchmarks/build_buzzdb_embedding_cache.py'
  const key = spawnSync(python, [builder, '--cache-key'], { cwd: root, encoding: 'utf8' })
  if (key.status !== 0 || !/^[a-f0-9]{64}$/.test(key.stdout.trim())) throw new Error(key.stderr || 'Cannot calculate public embedding cache identity.')
  const path = resolve(root, `tmp/buzzdb-${key.stdout.trim()}.npz`)
  if (!existsSync(path)) {
    const build = spawnSync(python, [builder, path], { cwd: root, stdio: 'inherit' })
    if (build.status !== 0) throw new Error('Public embedding cache build failed.')
  }
  // Existing Python validation checks every identity and all vectors on reuse.
  process.env.TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH = path
}

function runBuzzdbHybridBenchmark() {
  const python = process.env.TOKENSMITH_BENCHMARK_PYTHON ?? 'python3'
  const result = spawnSync(python, ['-m', 'tests.benchmarks.test_buzzdb_embeddings', '--json'], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, TOKENSMITH_RUN_EMBEDDING_BENCHMARK: '1' },
    maxBuffer: 16 * 1024 * 1024
  })

  if (result.error) {
    throw result.error
  }

  let parsed
  try {
    parsed = parseJsonOutput(result.stdout)
  } catch (error) {
    const details = [error.message, result.stderr.trim(), result.stdout.trim()].filter(Boolean).join('\n')
    throw new Error(details)
  }

  if (result.status !== 0 && parsed.passed === parsed.total) {
    parsed.passed = 0
    parsed.total = Math.max(1, parsed.total)
    parsed.cases = [{
      id: 'hybrid_retrieval',
      passed: false,
      failures: [result.stderr.trim() || `Python exited with status ${result.status}.`]
    }]
  }

  return parsed
}

function printFailures(suites) {
  for (const suite of suites.filter(s => s.gate)) for (const failure of suite.gate.failures) console.error(`- ${suite.label}: ${failure}`)
  const failedCases = suites.filter(s => !s.gate).flatMap((suite) =>
    (suite.cases ?? [])
      .filter((testCase) => !testCase.passed)
      .map((testCase) => ({ suite, testCase }))
  )

  if (failedCases.length === 0) {
    return
  }

  console.error('Failures:')
  for (const { suite, testCase } of failedCases) {
    const details = (testCase.failures ?? ['failed']).join('; ')
    console.error(`- ${suite.label}: ${testCase.id}: ${details}`)
  }
}

export function benchmarkPassed(suites) {
  return suites.every(suite => suite.gate ? suite.gate.passed : suite.total > 0 && suite.passed === suite.total)
}

function printSummary(suites) {
  console.log(suites.some(s => s.reviewStatus === 'graded')
    ? 'Study benchmark (retrieval, conversation contracts, and graded live answers):'
    : 'Study benchmark (evidence coverage and conversation contracts; not generated-answer accuracy):')
  for (const suite of suites) {
    console.log(`- ${suite.label}: ${suite.passed}/${suite.total} ${suite.unit}`)
    if (suite.gate) console.log(`  Regression gate: ${suite.gate.passed ? 'PASS' : 'FAIL'}; known baseline failures: ${suite.gate.knownFailures.join(', ') || 'none'}`)
  }
  return benchmarkPassed(suites)
}

function writeReport(suites, passed) {
  const reportPath = process.env.TOKENSMITH_BENCHMARK_REPORT_PATH
  if (reportPath) {
    mkdirSync(dirname(resolve(reportPath)), { recursive: true })
    writeFileSync(reportPath, `${JSON.stringify({ passed, suites }, null, 2)}\n`)
  }
  if (!process.env.GITHUB_STEP_SUMMARY) return
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, formatBenchmarkSummary(suites))
}

export function formatBenchmarkSummary(suites) {
  const escapeCell = (value) => String(value).replaceAll('&', '&amp;').replaceAll('|', '&#124;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replace(/\r?\n/g, ' ')
  const details = (item) => {
    const parts = []
    if (item.referenceExchange) {
      const origin = item.reviewStatus ? 'live conversation' : 'fixture'
      parts.push(`**Prior question (${origin}):** ${escapeCell(item.referenceExchange.question)}`,
        `**Prior answer (${origin}):** ${escapeCell(item.referenceExchange.answer)}`)
    }
    if (item.question) parts.push(`**Question:** ${escapeCell(item.question)}`)
    if (item.retrievalQuery) parts.push(`**Reference search query (cleanup model not run):** ${escapeCell(item.retrievalQuery)}`)
    if (item.answer) parts.push(`**Actual model answer (${item.reviewStatus === 'graded' ? 'graded' : 'ungraded'}):** ${escapeCell(item.answer.text)}`)
    if (item.referenceAnswer) parts.push(`**Reference answer (not generated):** ${escapeCell(item.referenceAnswer)}`)
    if (item.evidence) {
      parts.push(`**${escapeCell(item.evidenceLabel)}:**`)
      if (!item.evidence.length) parts.push('No evidence returned.')
      for (const [index, source] of item.evidence.entries()) {
        const text = source.context.replace(/\s+/g, ' ').trim()
        const excerpt = text.slice(0, 320) + (text.length > 320 ? '...' : '')
        parts.push(`${index + 1}. ${escapeCell(source.chunkIds.join(', '))}: ${escapeCell(source.section)} - ${escapeCell(excerpt)}`)
      }
    }
    if (item.failures?.length) parts.push(`**Failures:** ${escapeCell(item.failures.join('; '))}`)
    return parts.join('<br>')
  }
  const lines = [
    ...suites.filter(s => s.gate).map(s => `Graded answer regression gate: ${s.gate.passed ? 'PASS' : 'FAIL'}. ${s.gate.failures.join(' ')} Known failures: ${s.gate.knownFailures.join(', ') || 'none'}.`),
    '## BuzzDB Benchmark', '',
    suites.some(s => s.reviewStatus === 'graded')
      ? '**Scope:** real hybrid retrieval, mocked conversation contracts, and opt-in live answers graded against source-backed rubrics. Calibration and regression gates are reported separately from acceptance counts.'
      : '**Scope:** app-matched Ollama Nomic embeddings + production hybrid retrieval. Conversation contracts use a mocked resolver/search. No generated-answer accuracy is claimed.', '',
    'Reference answers are authored expectations, not model output or automatically graded answers. Evidence excerpts are shortened for display; the JSON report retains full selected context.', '',
    '| Suite | Passed / Completed | Rate |', '| --- | --- | --- |',
    ...suites.map((suite) => `| ${escapeCell(suite.label)} | ${suite.passed}/${suite.total} | ${suite.total ? (100 * suite.passed / suite.total).toFixed(1) : '0.0'}% |`),
    '', ...suites.filter((suite) => suite.model && suite.name !== 'reasoning_answers').map((suite) =>
      `Embedder: \`${escapeCell(suite.model)}\`, digest \`${escapeCell(suite.modelDigest)}\`, Ollama \`${escapeCell(suite.ollamaVersion)}\`. Fresh query embedding time: ${(suite.measurements?.fresh_query_embedding_seconds ?? 0).toFixed(2)}s.`
    ),
    '', '| Case | Result | Details |', '| --- | --- | --- |',
    ...suites.flatMap((suite) => (suite.cases ?? []).map((item) =>
      `| ${escapeCell(item.id)} | ${suite.name === 'reasoning_answers' ? (item.reviewStatus === 'graded' ? item.decision?.verdict?.toUpperCase() ?? 'ERROR' : item.passed ? 'COMPLETED (ungraded)' : 'ERROR / BLOCKED') : item.passed ? 'PASS' : 'FAIL'} | ${details(item)} |`
    )), ''
  ]
  return lines.join('\n')
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const suites = []
  if (process.argv.includes('--graded')) {
    const profile = JSON.parse(readFileSync('tests/fixtures/buzzdb/regression/suite.json'))
    if (process.env.TOKENSMITH_BENCHMARK_OLLAMA_URL && process.env.TOKENSMITH_BENCHMARK_OLLAMA_URL !== 'http://127.0.0.1:11434') throw new Error('The graded profile requires local Ollama at http://127.0.0.1:11434.')
    // Use the calibrated local profile's runtime for both retrieval and grading.
    // The ordinary retrieval CI command keeps its existing separate runtime pin.
    process.env.TOKENSMITH_BENCHMARK_OLLAMA_VERSION = profile.ollamaVersion
    process.env.TOKENSMITH_BENCHMARK_OLLAMA_URL = 'http://127.0.0.1:11434'
    prepareGradedEmbeddingCache()
  }

  try {
    suites.push(runBuzzdbHybridBenchmark())
  } catch (error) {
    suites.push(failureSuite('hybrid_retrieval', 'hybrid evidence coverage', error))
  }

  try {
    suites.push(await runBuzzdbMultiturnBenchmark())
  } catch (error) {
    suites.push(failureSuite('multi_turn', 'multi-turn', error))
  }

  if (process.argv.includes('--live') || process.argv.includes('--graded')) {
    try {
      const { runBuzzdbReasoningBenchmark } = await import('./buzzdb_reasoning.mjs')
      suites.push(await runBuzzdbReasoningBenchmark({ graded: process.argv.includes('--graded') }))
    } catch (error) {
      suites.push(failureSuite('reasoning_answers', 'live reasoning execution', error))
    }
  }

  const passed = printSummary(suites)
  printFailures(suites)
  writeReport(suites, passed)

  if (!passed) {
    process.exitCode = 1
  }
}
