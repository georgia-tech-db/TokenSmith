import { spawnSync } from 'node:child_process'
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

function runBuzzdbGroundingBenchmark() {
  const python = process.env.TOKENSMITH_BENCHMARK_PYTHON ?? 'python3'
  const result = spawnSync(python, ['tests/benchmarks/test_buzzdb_grounding.py', '--json'], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: process.env
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
      id: 'grounding',
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
  const passed = suites.reduce((sum, suite) => sum + suite.passed, 0)
  const total = suites.reduce((sum, suite) => sum + suite.total, 0)
  console.log(`BuzzDB retrieval benchmark: ${passed}/${total} checks passed.`)
  for (const suite of suites) {
    console.log(`- ${suite.label}: ${suite.passed}/${suite.total} ${suite.unit}`)
  }
  return passed === total
}

const suites = []

try {
  suites.push(runBuzzdbGroundingBenchmark())
} catch (error) {
  suites.push(failureSuite('grounding', 'grounding', error))
}

try {
  suites.push(await runBuzzdbMultiturnBenchmark())
} catch (error) {
  suites.push(failureSuite('multi_turn', 'multi-turn', error))
}

const passed = printSummary(suites)
printFailures(suites)

if (!passed) {
  process.exitCode = 1
}
