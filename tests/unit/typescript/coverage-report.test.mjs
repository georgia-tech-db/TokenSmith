import assert from 'node:assert/strict'
import test from 'node:test'
import { checkInventory, formatReport, summarizeCoverage } from '../../../scripts/coverage-report.mjs'

function fixture() {
  return summarizeCoverage({
    total: { lines: { covered: 90, total: 100 } },
    'src/test.ts': { lines: { covered: 90, total: 100 } },
    'src/untested.tsx': { lines: { covered: 0, total: 800 } }
  }, { files: { 'python_engine/test.py': { summary: { covered_lines: 40, num_statements: 100, percent_covered: 70 } } } })
}

test('coverage reports languages separately, including untested files, not branch percentages', () => {
  const report = fixture()
  assert.equal(report.typescript.percent, 10)
  assert.equal(report.typescript.total, 900)
  assert.equal(report.badges.typescript.message, '10.0%')
  assert.equal(report.badges.python.message, '40.0%')
  assert.equal(report.python.percent, 40)
  assert.doesNotMatch(formatReport(report), /Combined/)
  assert.match(formatReport(report), /src\/untested.tsx/)
})

test('coverage fails closed for empty, invalid, incomplete or unexpected reports', () => {
  assert.throws(() => summarizeCoverage({}, { files: {} }), /empty/)
  assert.throws(() => summarizeCoverage({ a: { lines: { covered: 10, total: 2 } } }, { files: {} }), /Invalid/)
  const report = fixture()
  const paths = report.files.map(file => file.path)
  checkInventory(report, paths)
  assert.throws(() => checkInventory(report, [...paths, 'src/missing.ts']), /missing.ts/)
  assert.throws(() => checkInventory(report, paths.slice(1)), /unexpected/)
  assert.throws(() => checkInventory({ files: [...report.files, report.files[0]] }, paths), /mismatch/)
})
