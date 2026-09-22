import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { parseArgs } from 'node:util'

const { values } = parseArgs({ options: {
  fixed: { type: 'string' }, review: { type: 'string' }, out: { type: 'string' }
} })
if (!values.fixed || !values.review || !values.out) throw new Error('--fixed, --review and --out are required')
const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'))
const report = readJson(values.fixed)
const review = readJson(values.review)
const models = Object.keys(report.models)
const lines = [
  '# Gemma Final Answer Quality', '',
  '## Method', '',
  review.method, '',
  'Ten cases were run with both Llama 3 and Gemma 4 E4B using identical questions, current app prompts, and the same four previously hybrid-retrieved sources in the same order. No sources were dropped or clipped. Both used 8192 context tokens, a 1024-token output limit, temperature 0, seed 42, and thinking disabled. Every answer completed without hitting its output limit.', '',
  'Two additional live conversations used real nomic query embeddings and production hybrid retrieval against a read-only snapshot of the installed BuzzDBBook index. Follow-ups were selected after reading the actual preceding Gemma answer. These used the experimental Gemma rewrite pipeline, not the installed app router or GUI.', '',
  'These are one-sample diagnostic cases, deliberately including earlier failures. The manual counts below are not representative accuracy percentages or a CI regression guarantee. One acceptable outcome is an appropriate evidence-gap acknowledgment, not a substantive answer.', '',
  '## Summary', '',
  '| Model | Acceptable | Partial | Incorrect | Median answer time |',
  '| --- | ---: | ---: | ---: | ---: |'
]
for (const model of models) {
  const counts = { acceptable: 0, partial: 0, incorrect: 0 }
  for (const item of review.cases) counts[item[model].verdict]++
  const times = report.answers.filter((answer) => answer.model === model).map((answer) => answer.seconds).sort((a, b) => a - b)
  const middle = Math.floor(times.length / 2)
  const median = times.length % 2 ? times[middle] : (times[middle - 1] + times[middle]) / 2
  lines.push(`| ${model} | ${counts.acceptable} | ${counts.partial} | ${counts.incorrect} | ${median.toFixed(2)} s |`)
}
lines.push('', 'Warm-up excluded. These are end-to-end generation request times, not matched-length token throughput.', '')
for (const [label, description] of Object.entries(review.labels)) lines.push(`- **${label}:** ${description}`)
lines.push('', '## Fixed-Evidence Answers', '')
for (const item of report.cases) {
  const assessment = review.cases.find((entry) => entry.id === item.id)
  lines.push(`### ${item.id}`, '', `**Question:** ${item.question}`, '', `**Prewritten review criterion:** ${item.review}`, '', '**Sources in input order:**', '')
  for (const source of item.sources) lines.push(`- ${source.sectionHeader} (row ${source.chunkRowid ?? source.id})`)
  for (const model of models) {
    const answer = report.answers.find((entry) => entry.caseId === item.id && entry.model === model)
    lines.push('', `#### ${model}`, '', `**Review: ${assessment[model].verdict}.** ${assessment[model].notes}`, '',
      `Time: ${answer.seconds.toFixed(2)} s. Actual raw answer:`, '', answer.rawText, '')
  }
}
lines.push('## Live Conversations', '')
for (const item of review.live) {
  const session = readJson(resolve(dirname(values.fixed), item.file))
  lines.push(`### ${item.file}`, '', item.review, '')
  for (const [index, turn] of session.turns.entries()) {
    lines.push(`#### Turn ${index + 1}`, '', `**Student:** ${turn.question}`, '', `**Retrieval question:** ${turn.query}`, '',
      `Hybrid vector hits: ${turn.vectorHits}. Answer time: ${turn.answer.seconds.toFixed(2)} s. Total including rewriting/search: ${turn.totalSeconds.toFixed(2)} s.`, '', '**Sources:**', '')
    for (const source of turn.sources) lines.push(`- ${source.sectionHeader} (row ${source.chunkRowid ?? source.id})`)
    lines.push('', '**Actual raw answer:**', '', turn.answer.text, '')
  }
}
lines.push('## App Formatting', '', review.formatting, '',
  '## Artifacts', '',
  `[Fixed-source requests and responses](${resolve(values.fixed)})`, '',
  `[Manual assessments](${resolve(values.review)})`, '',
  'No production prompts, routing, app model selection, or answer formatting were changed. Nothing was committed or pushed.', '')
writeFileSync(values.out, lines.join('\n'))
console.log(values.out)
