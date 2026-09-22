import { appendFileSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

function count(covered, total) {
  if (!Number.isInteger(covered) || !Number.isInteger(total) || covered < 0 || total < covered) {
    throw new Error('Invalid coverage counts')
  }
  return { covered, total, percent: total ? 100 * covered / total : 0 }
}

function sum(files) {
  const result = count(files.reduce((n, file) => n + file.covered, 0), files.reduce((n, file) => n + file.total, 0))
  if (!result.total) throw new Error('Coverage report is empty')
  return result
}

export function summarizeCoverage(ts, python) {
  const tsFiles = Object.entries(ts).filter(([path]) => path !== 'total').map(([path, data]) => ({
    path: relative(process.cwd(), resolve(path)).replaceAll('\\', '/'),
    ...count(data.lines.covered, data.lines.total)
  }))
  const pythonFiles = Object.entries(python.files).map(([path, data]) => ({
    path: path.replaceAll('\\', '/'), ...count(data.summary.covered_lines, data.summary.num_statements)
  }))
  const typescript = sum(tsFiles)
  const py = sum(pythonFiles)
  const badge = (label, value) => ({ schemaVersion: 1, label, message: `${value.percent.toFixed(1)}%`,
    color: value.percent >= 80 ? 'brightgreen' : value.percent >= 60 ? 'yellow' : 'orange' })
  return {
    typescript, python: py,
    files: [...tsFiles, ...pythonFiles].sort((a, b) => a.path.localeCompare(b.path)),
    badges: {
      typescript: badge('frontend TS coverage', typescript),
      python: badge('backend Python coverage', py)
    }
  }
}

function sourceFiles(directory, matches) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name)
    return entry.isDirectory() ? sourceFiles(path, matches) : matches(path) ? [path.replaceAll('\\', '/')] : []
  })
}

export function checkInventory(report, expected) {
  const paths = report.files.map(file => file.path)
  const missing = expected.filter(path => !paths.includes(path))
  const extra = paths.filter(path => !expected.includes(path))
  if (missing.length || extra.length || new Set(paths).size !== paths.length) {
    throw new Error(`Coverage scope mismatch: missing ${missing.join(', ')}; unexpected ${extra.join(', ')}`)
  }
}

export function formatReport(report) {
  const row = (label, data) => `| ${label} | ${data.covered}/${data.total} | ${data.percent.toFixed(1)}% |`
  return [
    '## Unit-Test Line Coverage', '',
    '| Scope | Covered / total lines | Coverage |', '| --- | ---: | ---: |',
    row('Frontend / Electron (TypeScript / TSX)', report.typescript), row('Backend (Python)', report.python), '',
    'Includes untested files in `src/` and `python_engine/`; excludes TypeScript declaration files.',
    'TypeScript coverage includes the UI, Electron main/preload, and shared modules, not only the renderer.',
    'Measured with c8 (V8 line coverage) and coverage.py (executable Python lines).',
    'This measures unit-test execution, not answer accuracy or full end-to-end coverage.', '',
    '<details><summary>Per-file coverage</summary>', '',
    '| File | Covered / total lines | Coverage |', '| --- | ---: | ---: |',
    ...report.files.map(file => row(`\`${file.path}\``, file)), '', '</details>', ''
  ].join('\n')
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const read = path => JSON.parse(readFileSync(path, 'utf8'))
  const report = summarizeCoverage(read('.coverage-reports/typescript/coverage-summary.json'), read('.coverage-reports/python.json'))
  checkInventory(report, [
    ...sourceFiles('src', path => /\.tsx?$/.test(path) && !path.endsWith('.d.ts')),
    ...sourceFiles('python_engine', path => path.endsWith('.py'))
  ])
  const summary = formatReport(report)
  writeFileSync('.coverage-reports/summary.md', summary)
  for (const [language, badge] of Object.entries(report.badges)) {
    writeFileSync(`.coverage-reports/${language}-coverage.json`, JSON.stringify(badge))
  }
  if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, summary)
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `report=${JSON.stringify({ badges: report.badges, summary })}\n`)
  console.log(summary.split('<details>')[0])
}
