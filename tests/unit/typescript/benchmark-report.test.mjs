import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { formatBenchmarkSummary } from '../../benchmarks/test_buzzdb_retrieval.mjs'

test('benchmark details show reference answers and ordered evidence without claiming generation', () => {
  const summary = formatBenchmarkSummary([{
    label: 'retrieval', passed: 1, total: 1, cases: [{
      id: 'example', passed: true, question: 'Why A | B < C?', referenceAnswer: 'An authored answer.',
      evidenceLabel: 'Retrieved evidence (rank order)',
      evidence: [
        { chunkIds: ['ch01.001'], section: 'First', context: '<script>not HTML</script> ' + 'x'.repeat(400) },
        { chunkIds: ['ch01.002'], section: 'Second', context: 'Second passage.' }
      ], failures: []
    }]
  }])
  assert.match(summary, /Question:\*\* Why A &#124; B &lt; C\?/)
  assert.match(summary, /Reference answer \(not generated\):\*\* An authored answer/)
  assert.ok(summary.indexOf('ch01.001') < summary.indexOf('ch01.002'))
  assert.match(summary, /&lt;script&gt;/)
  assert.doesNotMatch(summary, /<script>|x{321}/)
})

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
    for (const item of report.suites[1].cases) {
      assert.ok(item.question)
      assert.ok(item.referenceAnswer)
      assert.ok(item.evidence.length)
    }
    const summary = readFileSync(summaryPath, 'utf8')
    assert.match(summary, /FAIL/)
    assert.match(summary, /No generated-answer accuracy is claimed/)
    assert.match(summary, /conversation contracts/)
    assert.match(summary, /Prior answer \(fixture\)/)
    assert.match(summary, /Fixture evidence \(mock search, not live retrieval\)/)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
