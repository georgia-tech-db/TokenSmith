import { createHash } from 'node:crypto'

export const sha256 = value => createHash('sha256').update(value).digest('hex')

export function validateEnvironment(suite, book, installedModels) {
  if (sha256(book) !== suite.bookSha256) throw new Error('Public BuzzDB fixture changed. Review the book and rubrics before updating suite.json.')
  for (const spec of Object.values(suite.models)) {
    const model = installedModels.find(m => m.name === spec.name)
    if (!model) throw new Error(`Required local model is not installed: ${spec.name}. No models will be downloaded automatically.`)
    if (model.digest !== spec.digest) throw new Error(`Model digest changed: ${spec.name}. Review and recalibrate the baseline before changing suite.json.`)
  }
}

export function evidenceChunk(book, id) {
  const marker = `<!-- tokensmith:chunk id="${id}"`
  const start = book.indexOf(marker)
  if (start < 0) throw new Error(`Missing public evidence chunk: ${id}`)
  const end = book.indexOf('<!-- tokensmith:chunk', start + marker.length)
  return book.slice(book.indexOf('-->', start) + 3, end < 0 ? undefined : end).trim()
}

export function corruptRecency(text) {
  return typeof text === 'string' ? text.replace(/\b(front|back)\b/gi, word => {
    const opposite = word.toLowerCase() === 'front' ? 'back' : 'front'
    return word[0] === word[0].toUpperCase() ? opposite[0].toUpperCase() + opposite.slice(1) : opposite
  }) : text
}

export function regressionGate(suite, controls, report) {
  const failures = [], knownFailures = [], improvements = []
  if (!report.completed || report.error) failures.push(report.error || 'Run did not complete.')
  for (const control of controls) {
    const rows = report.controls.filter(r => r.id === control.id)
    if (rows.length !== 1 || rows[0].decision.verdict !== control.expected) failures.push(`Judge calibration failed: ${control.id}`)
  }
  for (const condition of ['baseline', 'retrieval-front-back-swapped']) {
    const runs = report.conditions.filter(r => r.condition === condition)
    if (runs.length !== 1) { failures.push(`Missing or duplicate condition: ${condition}`); continue }
    for (const c of suite.cases) {
      const turns = runs[0].turns.filter(t => t.id === c.id)
      if (turns.length !== 1) { failures.push(`Missing or duplicate turn: ${condition}/${c.id}`); continue }
      const verdict = turns[0].decision.verdict
      if (!['correct', 'incorrect'].includes(verdict)) failures.push(`Unresolved judgment: ${condition}/${c.id}`)
      if (condition === 'baseline') {
        if (verdict === 'incorrect') {
          if (Object.hasOwn(suite.knownFailures, c.id)) knownFailures.push(c.id)
          else failures.push(`New answer failure: ${c.id}`)
        } else if (verdict === 'correct' && Object.hasOwn(suite.knownFailures, c.id)) improvements.push(c.id)
      }
    }
    if (condition !== 'baseline' && !runs[0].turns.some(t => t.changedSources > 0)) failures.push('Retrieval fault did not change any evidence.')
  }
  // Recovery from corrupted retrieval is allowed. Known-answer controls independently
  // ensure the judge rejects false answers, rather than demanding the generator fail.
  return { passed: failures.length === 0, failures, knownFailures, improvements }
}
