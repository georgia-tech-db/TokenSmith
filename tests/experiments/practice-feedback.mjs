// Opt-in live diagnostic. Uses real book passages and the production Practice
// client/provider. Passage selection is fixed here, not a retrieval benchmark.
import { execFileSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { createPracticeClient } = requireTranspiledTs('src/renderer/src/practice-client.ts')
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { parsePracticeQuestion } = requireTranspiledTs('src/shared/quiz.ts')
const { practiceSourceKey } = requireTranspiledTs('src/shared/practice.ts')
const modelName = process.env.PRACTICE_MODEL || 'gemma4:26b'
const model = { id: 'live-practice', name: modelName, engine: 'ollama', status: 'ready', ollamaModelName: modelName, contextLength: 8192 }
const modelSettings = { contextLength: 8192, maxLength: 4096, reasoningMode: 'off', temperature: 0.3, topP: 0.95 }
const session = { modelId: model.id, modelSettings, questions: [], documents: [], totalQuestions: 5 }
const chunks = JSON.parse(execFileSync(process.env.PRACTICE_PYTHON || 'app_runtime/python/bin/python', ['-E', '-s', '-B', '-c',
  'import json; from tests.benchmarks.test_buzzdb_embeddings import all_chunks; print(json.dumps(all_chunks()))'], { maxBuffer: 10_000_000 }))
const sources = prefix => chunks.filter(chunk => chunk.sectionHeader.startsWith(prefix)).slice(0, 3).map(chunk => ({
  title: 'BuzzDBBook', documentTitle: 'buzzdb-book.tokensmith.md', materialId: 'book', documentId: 1,
  chunkId: chunk.tokensmithChunkId, sectionHeader: chunk.sectionHeader, excerpt: chunk.text, context: chunk.text
}))
const rows = []
const calls = []
const output = process.env.PRACTICE_OUTPUT || '/tmp/tokensmith-practice-feedback-live.json'
const save = () => writeFileSync(output, JSON.stringify({ model: modelName, mode: 'fixed real book passages, production model pipeline', rows, calls }, null, 2))
const client = evidence => createPracticeClient({ practiceSources: async () => evidence,
  sendChatMessage: async request => {
    const reply = await runOllamaStudyEngine(request)
    calls.push({ id: request.requestId, task: request.practiceTask, prompt: request.prompt, reply })
    save()
    return reply
  } }, model, { application: { suggestionMode: 'off' } })
async function run(id, action) {
  if (process.env.PRACTICE_CASE_FILTER && !id.includes(process.env.PRACTICE_CASE_FILTER)) return null
  const start = performance.now()
  try {
    const result = await action()
    rows.push({ id, seconds: (performance.now() - start) / 1000, result })
    save(); console.log(JSON.stringify(rows.at(-1)))
    return result
  } catch (error) {
    rows.push({ id, seconds: (performance.now() - start) / 1000, error: error.message })
    save(); console.log(JSON.stringify(rows.at(-1)))
    return null
  }
}

const heap = sources('4.4.4 ')
const heapClient = client(heap)
const generated = await run('generated.heap', () => heapClient.question(session, 'heap-question', new AbortController().signal))
if (generated) {
  await run('generated.heap.reference-answer', () => heapClient.feedback(session, generated, generated.referenceAnswer, 'reference'))
}
// A fixed, explicitly qualified question lets us compare the motivating answer,
// a misconception, and a revision against one independently authored rubric.
const fixed = { id: 'fixed-heap', attempts: [], draft: '', reviewing: false, ...parsePracticeQuestion(JSON.stringify({
  question: 'Why can finding a record by key in a heap file require scanning every page when no usable index or record location is available?',
  objective: 'Connect unordered storage to the worst-case search cost.',
  assumptions: ['No usable index or known record location.'],
  criteria: [
    { id: 'c1', description: 'No usable index or known location provides a direct route to the record.',
      evidence: [{ sourceKey: practiceSourceKey(heap[0]), quote: 'the database has no idea which page it might be on.' }] },
    { id: 'c2', description: 'Records are unordered, so any page might hold the record; in the worst case all pages must be checked.',
      evidence: [{ sourceKey: practiceSourceKey(heap[0]), quote: 'Because records are not stored in any particular order, there are no shortcuts to finding a specific tuple.' }] }
  ],
  explanation: 'Without a usable index or known record location, the key does not identify a page. Heap records are unordered, so the record could be on any page. In the worst case the search examines every page, though a unique match may be found earlier.',
  hint: 'Think about what the storage order tells you about the page holding the key.'
}), heap) }
const firstAnswer = 'Because there is no index to accelerate the search'
const first = await run('fixed.heap.incomplete', () => heapClient.feedback(session, fixed, firstAnswer, 'first'))
await run('fixed.heap.misconception', () => heapClient.feedback(session, fixed, 'Heap records are sorted by key, so binary search can jump to the right page.', 'wrong'))
await run('fixed.heap.revision', () => heapClient.feedback(session, { ...fixed,
  attempts: first ? [{ answer: firstAnswer, feedback: first }] : [] },
  'Without an index we have no direct route to the record. Heap records are unordered, so the key does not identify a page; we may need to inspect all pages. A unique match can be found before reaching the last page.', 'revision'))
await run('fixed.heap.ambiguous-question', () => heapClient.feedback(session, { ...fixed,
  text: 'Why must every lookup in a heap file always read every page?' },
  'That is not always true. An index or a known record location can avoid the scan, and finding a unique key early lets us stop.', 'ambiguous'))
const vector = sources('15.4.1 ')
const vectorClient = client(vector)
const vectorQuestion = await run('generated.vector', () => vectorClient.question(session, 'vector-question', new AbortController().signal))
if (vectorQuestion) await run('generated.vector.reference-answer', () => vectorClient.feedback(session, vectorQuestion, vectorQuestion.referenceAnswer, 'vector-reference'))
console.log(`Saved ${rows.length} real model results to ${output}. Semantic feedback still requires human review.`)
