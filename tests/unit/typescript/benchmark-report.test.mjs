import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

test('a missing embedder runtime fails the benchmark and still reports the failure', () => {
  const directory = mkdtempSync(join(tmpdir(), 'tokensmith-benchmark-report-'))
  try {
    const reportPath = join(directory, 'report.json')
    const summaryPath = join(directory, 'summary.md')
    const result = spawnSync(process.execPath, ['tests/benchmarks/test_buzzdb_retrieval.mjs'], {
      encoding: 'utf8',
      env: {
        ...process.env,
        TOKENSMITH_BENCHMARK_PYTHON: join(directory, 'missing-python'),
        TOKENSMITH_BENCHMARK_REPORT_PATH: reportPath,
        GITHUB_STEP_SUMMARY: summaryPath
      }
    })
    assert.equal(result.status, 1, result.stderr)
    const report = JSON.parse(readFileSync(reportPath, 'utf8'))
    assert.equal(report.passed, false)
    assert.equal(report.suites[0].passed, 0)
    assert.equal(report.suites[0].cases[0].passed, false)
    assert.equal(report.suites[1].passed, report.suites[1].total)
    const summary = readFileSync(summaryPath, 'utf8')
    assert.match(summary, /FAIL/)
    assert.match(summary, /No generated-answer accuracy is claimed/)
    assert.match(summary, /conversation contracts/)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
