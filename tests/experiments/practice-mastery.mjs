// Real-model diagnostic of the unverified, single-call Practice pipeline.
// Reads a prepared inventory; never modifies the app's books or student history.
import { readFileSync, writeFileSync } from 'node:fs'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { createPracticeClient } = requireTranspiledTs('src/renderer/src/practice-client.ts')
const { hasSupportingQuote } = requireTranspiledTs('src/shared/practice-evidence.ts')
const { trueFalseStatementIssue } = requireTranspiledTs('src/shared/true-false-statement.ts')
if (!process.env.PRACTICE_INVENTORY) throw new Error('Set PRACTICE_INVENTORY to a prepared topic-inventory.json.')
const inventory = JSON.parse(readFileSync(process.env.PRACTICE_INVENTORY, 'utf8'))
const modelName = process.env.PRACTICE_MODEL || 'gemma4:e4b'
const model = { id: `ollama:${modelName}`, name: modelName, engine: 'ollama', status: 'ready', ollamaModelName: modelName, contextLength: 8192 }
const modelSettings = { contextLength: 8192, contextLengthMode: 'manual', maxLength: 2048, reasoningMode: 'off', thinking: false, temperature: 0.3, topP: 0.95 }
const settings = { application: { suggestionMode: 'off', searchMode: 'hybrid' } }
const cases = JSON.parse(process.env.PRACTICE_TOPICS || '[{"title":"All-in-Memory Strategy","count":8},{"title":"Slotted Page","count":4},{"title":"Node Splitting","count":4}]')
const output = process.env.PRACTICE_OUTPUT || '/tmp/tokensmith-practice-single-call.json'
const rows = [], calls = [], history = []
const save = () => writeFileSync(output, JSON.stringify({ model: modelName, mode: 'single_call_without_quality_validation', rows, calls }, null, 2))
const bridge = {
  loadTopicInventory: async () => inventory,
  searchLibrary: async () => { throw new Error('A saved topic must not trigger retrieval.') },
  sendChatMessage: async request => {
    const start = performance.now()
    const reply = await runOllamaStudyEngine(request)
    calls.push({ task: request.practiceTask, prompt: request.prompt, seconds: (performance.now() - start) / 1000, sources: request.retrievedSources, text: reply.text })
    save(); return reply
  }
}
for (const item of cases) {
  const topic = inventory.topics.find(t => t.title === item.title)
  if (!topic) throw new Error(`Saved topic missing: ${item.title}`)
  const documents = [...new Map(topic.sources.map(s => [`${s.materialId}:${s.documentId}`, { materialId: s.materialId, documentId: s.documentId, title: s.documentTitle }])).values()]
  const materials = [...new Set(documents.map(d => d.materialId))].map(id => ({ id, collectionRevision: inventory.indexRevision }))
  let session = { id: `experiment-${history.length}`, title: topic.title, topic, modelId: model.id, modelSettings, documents, questions: [], currentIndex: 0, createdAt: new Date().toISOString() }
  const client = createPracticeClient(bridge, model, settings, { materials, embeddingModels: [] }, history)
  session = await client.prepareTopic(session, new AbortController().signal)
  for (let i = 0; i < item.count; i++) {
    const before = calls.length, start = performance.now()
    try {
      const question = await client.question(session, `${session.id}-${i}`, new AbortController().signal, (cursor, _sources, target) => {
        session.questionSourceCursor = cursor; session.questionTargetAnswer = target
      })
      if (!question) throw new Error('No mapped source')
      session.questions.push(question); session.questionTargetAnswer = undefined
      // Measurements only: never reject, repair or hide a generated question.
      const quoteMatches = question.rubric.criteria.every(c => c.evidence.every(e => question.sources.some(s => hasSupportingQuote(s.context || s.excerpt, e.quote))))
      rows.push({ topic: topic.title, number: i + 1, seconds: (performance.now() - start) / 1000, calls: calls.length - before,
        chunk: question.sources[0].chunkId, requestedAnswer: question.requestedAnswer, answer: question.correctAnswer,
        statement: question.text, explanation: question.referenceAnswer, quoteMatches,
        targetMatches: question.requestedAnswer === question.correctAnswer, statementIssue: trueFalseStatementIssue(question.text),
        repeatedText: session.questions.slice(0, -1).some(q => q.text.toLowerCase() === question.text.toLowerCase()) })
    } catch (error) { rows.push({ topic: topic.title, number: i + 1, error: error.message, calls: calls.length - before }) }
    save(); console.log(JSON.stringify(rows.at(-1)))
  }
  history.push(session)
}
console.log(`Saved real outputs to ${output}. Answer correctness and topic fit require review against the logged passages.`)
