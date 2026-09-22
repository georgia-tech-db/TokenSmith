import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { buzzdbSource } from '../helpers/buzzdb-fixture.mjs'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
const { studyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const cases = JSON.parse(readFileSync('tests/benchmarks/buzzdb_multiturn_cases.json', 'utf8'))

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
        modelSettings: { contextLength: 8192, maxLength: 1024 }
      }
      try {
        let calls = 0
        const query = turn.expectedMode === 'standalone' ? turn.prompt : `Fixture retrieval query: ${id}`
        const prepared = await prepareRewrittenStudyChat(request, {
          resolve: async () => ({ mode: turn.expectedMode, query, clarification: '' }),
          search: async (actualQuery) => {
            calls += 1
            assert.equal(actualQuery, messages.length ? query : turn.prompt)
            return sources
          }
        })
        assert.equal(calls, 1, 'Exactly one retrieval, without a heuristic probing pass')
        assert.equal(prepared.request.prompt, turn.prompt)
        assert.equal(prepared.request.answerPrompt, turn.prompt)
        assert.deepEqual(prepared.request.retrievedSources, sources, 'No hidden source reranking')
        const packed = studyChatMessages(prepared.request).at(-1).content
        assert.ok(packed.endsWith(`Question: ${turn.prompt}`))
        if (prepared.resolution.mode === 'contextual') {
          assert.deepEqual(prepared.request.referenceExchange, { question: previousQuestion, answer: previousAnswer })
          assert.ok(packed.includes('not factual evidence'))
          assert.ok(!packed.includes('Fixture retrieval query:'))
        } else {
          assert.equal(prepared.request.referenceExchange, undefined)
          assert.ok(!packed.includes('### Previous exchange'))
        }
      } catch (error) { failures.push(error.message) }
      results.push({
        id, question: turn.prompt, referenceAnswer: turn.referenceAnswer ?? turn.assistantAnswer,
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
