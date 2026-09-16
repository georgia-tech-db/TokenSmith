import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const directory = resolve(process.argv[2] || 'tmp/db50-comparison')
const read = path => JSON.parse(readFileSync(path, 'utf8'))
const definitions = read('tests/experiments/db50-cases.json')
const aiId = existsSync(`${directory}/ai-ids/results.json`) ? 'ai-ids' : 'ai'
const arms = ['basic', aiId].map(id => ({
  id, index: read(`${directory}/${id}/results.json`), answers: read(`${directory}/${id}/answers.json`)
}))
const median = numbers => {
  const sorted = [...numbers].sort((a, b) => a - b)
  return sorted.length % 2 ? sorted[Math.floor(sorted.length / 2)]
    : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
}
const output = [
  '# DB Book: 50-Page Preparation Experiment', '',
  'These are actual local model responses, not reference answers. They can contain errors.', '',
  'Database System Concepts, 7th Edition: printed pages 591-640, physical PDF pages 620-669.',
  'Basic and AI use the same extracted pages (verified SHA-256), Gemma 4 E4B, Nomic hybrid retrieval, four sources, an 8,192-token context, and a 1,024-token answer reserve.',
  'Both use the current production answer/rewrite/suggestion services: temperature 0.7, top-p 0.4, top-k 40, repeat penalty 1.18. No keyword-only search, model downloads, source edits, or changes to the live library.',
  aiId === 'ai-ids'
    ? 'The historical AI arm is an experiment-only numbered-boundary variant plus production parent expansion. The exact-quote implementation failed preparation and has no answer score.'
    : 'The AI arm uses production numbered-line preparation and linked source-unit retrieval.',
  'This compares preparation pipelines, not only boundary selection: Basic applies its existing cleaning; AI retains raw extracted text. Neither changes the generator prompt between arms.', '',
  '## Timing', '',
  '| Arm | Preparation | Embeddings | To Searchable Index | Chunks | Median Response | Median Prompt Tokens |',
  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'
]
const statistics = {}
for (const arm of arms) {
  const turns = arm.answers.cases.flatMap(c => c.turns)
  const seeds = arm.answers.cases.flatMap((c, i) => c.turns.slice(0, definitions.groups[i].questions.length))
  statistics[arm.id] = {
    answerCount: turns.length,
    medianResponseSeconds: median(turns.map(t => t.seconds)),
    seedMedianResponseSeconds: median(seeds.map(t => t.seconds)),
    medianPromptTokens: median(turns.map(t => t.budget.estimatedPromptTokens)),
    seedMedianPromptTokens: median(seeds.map(t => t.budget.estimatedPromptTokens)),
    truncatedSourceCount: turns.reduce((n, t) => n + t.budget.truncatedSourceCount, 0),
    retrievalModes: [...new Set(turns.flatMap(t => t.sources.map(s => s.retrievalMode)))],
    embeddingModels: [...new Set(turns.flatMap(t => t.sources.map(s => s.embeddingModel)))],
    timing: arm.index.timing
  }
  const timing = arm.index.timing
  output.push(`| ${arm.id} | ${timing.preparationSeconds.toFixed(2)}s | ${timing.embeddingSeconds.toFixed(2)}s | ${timing.totalSeconds.toFixed(2)}s | ${arm.index.chunkCount} | ${statistics[arm.id].medianResponseSeconds.toFixed(2)}s | ${statistics[arm.id].medianPromptTokens} |`)
}
output.push('', 'Response timings include rewriting when needed, retrieval, generation, and suggested questions. Import timings exclude GUI work, thumbnail generation, and process startup. One measured run per successful arm; model sampling is not seeded. These are not statistically established accuracy rates.', '',
  '## Matched Seed Questions', '', 'Expected facts below were defined before answer generation. They are a review guide, not a keyword-match grade.', '')

function appendTurn(turn) {
  output.push(turn.text, '', `Mode: ${turn.resolution.mode}. Retrieval query: ${turn.resolution.query}.`,
    `Elapsed: ${turn.seconds.toFixed(2)}s. Estimated prompt: ${turn.budget.estimatedPromptTokens} tokens. Truncated sources: ${turn.budget.truncatedSourceCount}.`, '', 'Sources, in supplied order:', '')
  for (const source of turn.sources) {
    const pages = source.pageStart === source.pageEnd ? source.pageStart - 29 : `${source.pageStart - 29}-${source.pageEnd - 29}`
    output.push(`- Printed page(s) ${pages}; ${source.sectionHeader || 'Basic text chunk'}; source ID: ${source.sourceId}`)
  }
  output.push('', 'Model-generated suggestions:', '', ...(turn.followUpSuggestions || []).map(q => `- ${q}`), '')
}

for (const [i, group] of definitions.groups.entries()) {
  for (const [j, question] of group.questions.entries()) {
    output.push(`### ${question.id}`, '', question.question, '', 'Expected facts:', '', ...question.requiredFacts.map(f => `- ${f}`), '')
    for (const arm of arms) {
      output.push(`#### ${arm.id}`, '')
      appendTurn(arm.answers.cases[i].turns[j])
    }
  }
}
output.push('## Organic Follow-Ups', '', 'Each chain follows the first suggestion generated from that arm\'s own preceding answer. Questions differ between arms, so these are qualitative conversation checks, not matched accuracy comparisons.', '')
for (const [i, group] of definitions.groups.entries()) {
  for (const arm of arms) {
    output.push(`### ${group.id}: ${arm.id}`, '', `Chain starts from: ${group.questions.at(-1).question}`, '')
    for (const turn of arm.answers.cases[i].turns.slice(group.questions.length)) {
      output.push(`#### ${turn.question}`, '')
      appendTurn(turn)
    }
  }
}
const repeatIds = new Set(process.argv.slice(3))
for (const arm of arms) {
  if (repeatIds.size) {
    const cases = definitions.groups.map((group, i) => ({
      ...arm.index.cases[i], questions: group.questions.filter(q => repeatIds.has(q.id)).map(q => q.question)
    })).filter(c => c.questions.length)
    writeFileSync(`${directory}/${arm.id}/repeat-input.json`, JSON.stringify({ ...arm.index, cases }, null, 2) + '\n')
  }
  const repeatPath = `${directory}/${arm.id}/repeat-answers.json`
  if (existsSync(repeatPath)) {
    output.push(`## Repeated Seed Checks: ${arm.id}`, '', 'A second generation with unchanged settings and sources. These cases were selected after the first run to probe apparent gains and remaining weaknesses.', '')
    for (const group of read(repeatPath).cases) {
      for (const turn of group.turns) {
        output.push(`### ${turn.question}`, '')
        appendTurn(turn)
      }
    }
  }
}
writeFileSync(`${directory}/answers.md`, output.join('\n') + '\n')
writeFileSync(`${directory}/statistics.json`, JSON.stringify(statistics, null, 2) + '\n')
console.log(JSON.stringify(statistics, null, 2))
