import { readFileSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { requireTranspiledTs } from '../../unit/typescript/ts-module-loader.mjs'
const base = 'tests/experiments/practice-topic-audit-2026-10-14/'
const bookPath = process.env.BUZZDB_BOOK || join(homedir(), 'Desktop/TokenSmith/BuzzDBBook/buzzdb-book.tokensmith.md')
const book = readFileSync(bookPath, 'utf8'), lines = book.split('\n')
const report = JSON.parse(readFileSync(base + 'results.json', 'utf8'))
const { hasSupportingQuote } = requireTranspiledTs('src/shared/practice-evidence.ts')
const { practiceSourceKey } = requireTranspiledTs('src/shared/practice.ts')
const result = { bookPath, bookSha256: createHash('sha256').update(book).digest('hex'),
  note: 'This checks citation provenance and grading consistency only. Semantic judgments are in review.json.', questions: [] }
for (const topic of report.topics) for (const row of topic.questions) {
  if (!row.question) continue
  const q = row.question
  const citations = q.rubric.criteria.flatMap(c => c.evidence.map(e => {
    const source = q.sources.find(s => practiceSourceKey(s) === e.sourceKey)
    const original = lines.slice(source.lineFrom - 1, source.lineTo).join('\n')
    return { criterion: c.id, sourceKey: e.sourceKey, quote: e.quote,
      lineFrom: source.lineFrom, lineTo: source.lineTo,
      foundInOriginalBookLines: hasSupportingQuote(original, e.quote) }
  }))
  result.questions.push({ topic: topic.title, number: row.number, citations,
    gradesConsistentWithSavedKey: row.trueFeedback.correct === q.correctAnswer && row.falseFeedback.correct === !q.correctAnswer })
}
writeFileSync(base + 'verification.json', JSON.stringify(result, null, 2))
const citations = result.questions.flatMap(q => q.citations)
console.log(JSON.stringify({ questions: result.questions.length, citations: citations.length,
  verifiedInOriginalBook: citations.filter(c => c.foundInOriginalBookLines).length,
  gradingChecksPassed: result.questions.every(q => q.gradesConsistentWithSavedKey) }))
if (citations.some(c => !c.foundInOriginalBookLines) || result.questions.some(q => !q.gradesConsistentWithSavedKey)) process.exitCode = 1
