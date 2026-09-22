import { spawnSync } from 'node:child_process'
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
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

function runBuzzdbHybridBenchmark() {
  const python = process.env.TOKENSMITH_BENCHMARK_PYTHON ?? 'python3'
  const result = spawnSync(python, ['-m', 'tests.benchmarks.test_buzzdb_fastembed', '--json'], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, TOKENSMITH_RUN_FASTEMBED_BENCHMARK: '1' },
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
  const failedCases = suites.flatMap((suite) =>
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

function printSummary(suites) {
  console.log('BuzzDB benchmark (evidence coverage and conversation contracts; not generated-answer accuracy):')
  for (const suite of suites) {
    console.log(`- ${suite.label}: ${suite.passed}/${suite.total} ${suite.unit}`)
  }
  return suites.every((suite) => suite.total > 0 && suite.passed === suite.total)
}

function writeReport(suites, passed) {
  const reportPath = process.env.TOKENSMITH_BENCHMARK_REPORT_PATH
  if (reportPath) {
    mkdirSync(dirname(resolve(reportPath)), { recursive: true })
    writeFileSync(reportPath, `${JSON.stringify({ passed, suites }, null, 2)}\n`)
  }
  if (!process.env.GITHUB_STEP_SUMMARY) return
  const escapeCell = (value) => String(value).replaceAll('|', '\\|').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replace(/\r?\n/g, ' ')
  const lines = [
    '## BuzzDB Benchmark', '',
    '**Scope:** real CPU embeddings + production hybrid retrieval. Conversation contracts use a mocked resolver/search. No generated-answer accuracy is claimed.', '',
    '| Suite | Passed | Rate |', '| --- | --- | --- |',
    ...suites.map((suite) => `| ${escapeCell(suite.label)} | ${suite.passed}/${suite.total} | ${suite.total ? (100 * suite.passed / suite.total).toFixed(1) : '0.0'}% |`),
    '', ...suites.filter((suite) => suite.model).map((suite) =>
      `Embedder: \`${escapeCell(suite.model)}\`. Fresh query embedding time: ${(suite.measurements?.fresh_query_embedding_seconds ?? 0).toFixed(2)}s.`
    ),
    '', '| Case | Result | Details |', '| --- | --- | --- |',
    ...suites.flatMap((suite) => (suite.cases ?? []).map((item) =>
      `| ${escapeCell(item.id)} | ${item.passed ? 'PASS' : 'FAIL'} | ${escapeCell((item.failures ?? []).join('; '))} |`
    )), ''
  ]
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, lines.join('\n'))
}

const suites = []

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

const passed = printSummary(suites)
printFailures(suites)
writeReport(suites, passed)

if (!passed) {
  process.exitCode = 1
}
