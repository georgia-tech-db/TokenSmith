import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { buzzdbSource } from '../helpers/buzzdb-fixture.mjs'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
const { studyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { parseQuestionRewrite } = requireTranspiledTs('src/main/engine/question-rewrite.ts')
const cases = JSON.parse(readFileSync('tests/benchmarks/buzzdb_multiturn_cases.json', 'utf8'))
for (const definition of JSON.parse(readFileSync('tests/benchmarks/buzzdb_regression_cases.json', 'utf8'))) {
  if (!definition.retrievalQuery) continue
  cases.push({ id: definition.id, prompt: definition.question, referenceAnswer: definition.referenceAnswer,
    referenceRetrievalQuery: definition.retrievalQuery, expectedMode: 'standalone', limit: 4,
    candidateChunkIds: definition.evidenceGroups.map(group => group.chunkIds[0]) })
}

// CI contract checks, not a model-quality score. Real answers are evaluated in
// the app; fixed fixture answers must not masquerade as organic conversations.
export async function runBuzzdbMultiturnBenchmark() {
  const results = []
  for (const definition of cases) {
    let previousQuestion = definition.seedQuestion ?? definition.priorQuestion
    let previousAnswer = definition.seedAnswer ?? definition.priorAnswer
    for (const turn of definition.turns ?? [definition]) {
      const id = turn.id === definition.id ? definition.id : `${definition.id}/${turn.id}`
      const failures = []
      const messages = previousQuestion && previousAnswer ? [
        { role: 'user', text: previousQuestion },
        { role: 'assistant', text: previousAnswer }
      ] : []
      const sources = turn.candidateChunkIds.slice(0, turn.limit ?? definition.limit).map(buzzdbSource)
      const request = {
        prompt: turn.prompt, messages, materials: [], settings: {},
        model: { engine: 'ollama', contextLength: 8192 },
        modelSettings: { contextLength: 8192, maxLength: 1024,
          ...(turn.referenceRetrievalQuery ? { reasoningMode: 'auto' } : {}) }
      }
      try {
        let calls = 0
        let plans = 0
        const query = turn.referenceRetrievalQuery ?? (turn.expectedMode === 'standalone' ? turn.prompt : `Fixture retrieval query: ${id}`)
        const prepared = await prepareRewrittenStudyChat(request, {
          resolve: async () => {
            plans += 1
            return parseQuestionRewrite(JSON.stringify({ mode: turn.expectedMode, query, clarification: '', reasoning: false }), turn.prompt)
          },
          search: async (actualQuery) => {
            calls += 1
            assert.equal(actualQuery, messages.length || turn.referenceRetrievalQuery ? query : turn.prompt)
            return { sources }
          }
        })
        assert.equal(calls, 1, 'Exactly one retrieval, without a heuristic probing pass')
        if (turn.referenceRetrievalQuery) assert.equal(plans, 1, 'First-turn cleanup uses the existing model call')
        assert.equal(prepared.request.prompt, turn.prompt)
        assert.equal(prepared.request.answerPrompt, turn.prompt)
        assert.deepEqual(prepared.request.retrievedSources, sources, 'No hidden source reranking')
        const packed = studyChatMessages(prepared.request).at(-1).content
        assert.ok(packed.endsWith(`Question: ${turn.prompt}`))
        if (prepared.resolution.mode === 'contextual') {
          assert.deepEqual(prepared.request.referenceExchange, { question: previousQuestion, answer: previousAnswer })
          assert.ok(packed.includes('not factual evidence'))
          assert.ok(!packed.includes('Fixture retrieval query:'))
          assert.equal(packed.split('\n\nQuestion:').at(-1).trim(), turn.prompt)
        } else {
          assert.equal(prepared.request.referenceExchange, undefined)
          assert.ok(!packed.includes('### Previous exchange'))
        }
      } catch (error) { failures.push(error.message) }
      results.push({
        id, question: turn.prompt, referenceAnswer: turn.referenceAnswer ?? turn.assistantAnswer,
        retrievalQuery: turn.referenceRetrievalQuery,
        referenceExchange: messages.length ? { question: previousQuestion, answer: previousAnswer } : undefined,
        evidenceLabel: 'Fixture evidence (mock search, not live retrieval)',
        evidence: sources.map(source => ({ chunkIds: [source.chunkId], section: source.sectionHeader, context: source.context })),
        passed: failures.length === 0, failures
      })
      previousQuestion = turn.prompt
      previousAnswer = turn.assistantAnswer
    }
  }
  return {
    name: 'conversation_contracts', label: 'conversation contracts (mock resolver/search; not accuracy)',
    unit: 'checks', passed: results.filter((result) => result.passed).length,
    total: results.length, cases: results
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const result = await runBuzzdbMultiturnBenchmark()
  console.log(process.argv.includes('--json') ? JSON.stringify(result) : `${result.label}: ${result.passed}/${result.total}`)
  for (const item of result.cases.filter((item) => !item.passed)) console.error(item)
  if (result.passed !== result.total) process.exitCode = 1
}
