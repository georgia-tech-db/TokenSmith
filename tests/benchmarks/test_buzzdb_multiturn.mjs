import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { buzzdbSource, searchBuzzdbFixture } from '../helpers/buzzdb-fixture.mjs'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const {
  routeRetrievalContext
} = requireTranspiledTs('src/shared/chat-context.ts')
const {
  studyChatMessages
} = requireTranspiledTs('src/main/engine/study-chat-format.ts')

const casesPath = resolve('tests/benchmarks/buzzdb_multiturn_cases.json')
const cases = JSON.parse(readFileSync(casesPath, 'utf8'))
const model = {
  id: 'ollama:llama3',
  name: 'Ollama llama3',
  engine: 'ollama',
  role: 'generator',
  status: 'ready',
  source: 'ollama',
  ollamaModelName: 'llama3',
  addedAt: '2026-09-10T00:00:00.000Z'
}

function normalizeText(value) {
  return String(value).toLowerCase().replaceAll('*', '').replaceAll('`', '').replace(/\s+/g, ' ').trim()
}

function completedTurn(userText, assistantText, sourceIds, idPrefix = '1') {
  return [
    {
      id: `u${idPrefix}`,
      role: 'user',
      text: userText
    },
    {
      id: `a${idPrefix}`,
      role: 'assistant',
      text: assistantText,
      sources: sourceIds.map((id) => buzzdbSource(id))
    }
  ]
}

function appendCompletedTurn(messages, userText, assistantText, sources, idPrefix) {
  messages.push({
    id: `u${idPrefix}`,
    role: 'user',
    text: userText
  })
  messages.push({
    id: `a${idPrefix}`,
    role: 'assistant',
    text: assistantText,
    sources
  })
}

function modelPromptFor(testId, prompt, messages, choice) {
  const modelMessages = studyChatMessages({
    prompt,
    answerPrompt: choice.answerPrompt,
    retrievalQuery: choice.query,
    conversationContextMode: choice.mode,
    messages,
    materials: [],
    model,
    settings: {},
    applicationSettings: {
      suggestionMode: 'off',
      followUpSuggestionCount: 2
    },
    modelSettings: {},
    retrievedSources: choice.sources
  })

  assert.equal(modelMessages.at(-1).role, 'user', `${testId} should end with the packed user message`)
  return modelMessages.at(-1).content
}

function assertIncludesAll(haystack, needles, label) {
  const normalizedHaystack = normalizeText(haystack)
  for (const needle of needles ?? []) {
    assert.ok(
      normalizedHaystack.includes(normalizeText(needle)),
      `${label} should include ${JSON.stringify(needle)}`
    )
  }
}

function assertExcludesAll(haystack, needles, label) {
  const normalizedHaystack = normalizeText(haystack)
  for (const needle of needles ?? []) {
    assert.ok(
      !normalizedHaystack.includes(normalizeText(needle)),
      `${label} should not include ${JSON.stringify(needle)}`
    )
  }
}

function firstIndexOfAny(values, candidates) {
  return values.findIndex((value) => candidates.includes(value))
}

export async function runBuzzdbMultiturnBenchmark() {
  let passed = 0
  let total = 0
  const results = []

  for (const caseDefinition of cases) {
    const initialQuestion = caseDefinition.seedQuestion ?? caseDefinition.priorQuestion
    const initialAnswer = caseDefinition.seedAnswer ?? caseDefinition.priorAnswer
    const initialSourceIds = caseDefinition.seedSourceIds ?? caseDefinition.priorSourceIds
    const messages = initialQuestion
      ? completedTurn(initialQuestion, initialAnswer, initialSourceIds, `${caseDefinition.id}-seed`)
      : []
    const turns = caseDefinition.turns ?? [caseDefinition]

    for (const [turnIndex, turnDefinition] of turns.entries()) {
      const testId = turnDefinition.id ? `${caseDefinition.id}/${turnDefinition.id}` : caseDefinition.id
      const failures = []
      const calls = []
      let choice = null
      let selectedChunkIds = []

      try {
        const limit = turnDefinition.limit ?? caseDefinition.limit
        const carriedSourceLimit = turnDefinition.carriedSourceLimit ?? caseDefinition.carriedSourceLimit
        const candidateSources = turnDefinition.candidateChunkIds.map((id) => buzzdbSource(id))
        choice = await routeRetrievalContext(turnDefinition.prompt, messages, {
          limit,
          turnCount: turnDefinition.turnCount ?? caseDefinition.turnCount ?? 1,
          carriedSourceLimit,
          search: (query, searchLimit = limit) =>
            searchBuzzdbFixture(query, candidateSources, calls, searchLimit)
        })
        selectedChunkIds = choice.sources.map((source) => source.chunkId)
        const packedPrompt = modelPromptFor(testId, turnDefinition.prompt, messages, choice)

        assert.equal(choice.mode, turnDefinition.expectedMode, `${testId} selected the wrong mode`)
        assert.ok(calls.length >= 1, `${testId} did not perform a standalone search`)
        if (turnDefinition.expectedMode === 'contextual') {
          assert.equal(calls.length, 2, `${testId} should perform standalone and contextual searches`)
        }

        assertIncludesAll(choice.query, turnDefinition.expectedQueryIncludes, `${testId} contextual query`)
        assertExcludesAll(choice.query, turnDefinition.expectedQueryExcludes, `${testId} contextual query`)

        if (turnDefinition.expectedFirstChunkIds) {
          assert.ok(
            turnDefinition.expectedFirstChunkIds.includes(selectedChunkIds[0]),
            `${testId} selected ${selectedChunkIds[0]} first; expected one of ${turnDefinition.expectedFirstChunkIds}`
          )
        }

        for (const chunkId of turnDefinition.expectedChunkIds ?? []) {
          assert.ok(
            selectedChunkIds.includes(chunkId),
            `${testId} should include ${chunkId}; selected ${selectedChunkIds}`
          )
        }

        for (const chunkId of turnDefinition.forbiddenChunkIds ?? []) {
          assert.ok(
            !selectedChunkIds.includes(chunkId),
            `${testId} should not include ${chunkId}; selected ${selectedChunkIds}`
          )
        }

        if (turnDefinition.expectedAnyChunkIdWithinRank) {
          const index = firstIndexOfAny(selectedChunkIds, turnDefinition.expectedAnyChunkIdWithinRank.chunkIds)
          assert.ok(
            index >= 0 && index < turnDefinition.expectedAnyChunkIdWithinRank.within,
            `${testId} expected one of ${turnDefinition.expectedAnyChunkIdWithinRank.chunkIds} within rank ` +
              `${turnDefinition.expectedAnyChunkIdWithinRank.within}; selected ${selectedChunkIds}`
          )
        }

        if (turnDefinition.rankAnyBeforeAny) {
          const earlierIndex = firstIndexOfAny(selectedChunkIds, turnDefinition.rankAnyBeforeAny.earlierChunkIds)
          const laterIndex = firstIndexOfAny(selectedChunkIds, turnDefinition.rankAnyBeforeAny.laterChunkIds)
          assert.ok(
            earlierIndex >= 0 && (laterIndex < 0 || earlierIndex < laterIndex),
            `${testId} expected ${turnDefinition.rankAnyBeforeAny.earlierChunkIds} before ` +
              `${turnDefinition.rankAnyBeforeAny.laterChunkIds}; selected ${selectedChunkIds}`
          )
        }

        if (turnDefinition.maxSelectedFromChunkIds) {
          const selectedCount = selectedChunkIds.filter((chunkId) =>
            turnDefinition.maxSelectedFromChunkIds.chunkIds.includes(chunkId)
          ).length
          assert.ok(
            selectedCount <= turnDefinition.maxSelectedFromChunkIds.max,
            `${testId} selected too many distractor chunks; selected ${selectedChunkIds}`
          )
        }

        assertIncludesAll(packedPrompt, turnDefinition.requiredPromptContext, `${testId} packed prompt`)
        assertExcludesAll(packedPrompt, turnDefinition.forbiddenPromptContext, `${testId} packed prompt`)
      } catch (error) {
        failures.push(error instanceof Error ? error.message : String(error))
      }

      if (choice) {
        appendCompletedTurn(
          messages,
          turnDefinition.prompt,
          turnDefinition.assistantAnswer ?? 'The selected sources answer the follow-up question.',
          choice.sources,
          `${caseDefinition.id}-${turnIndex + 1}`
        )
      }

      total += 1
      if (failures.length === 0) {
        passed += 1
      }
      results.push({
        id: testId,
        passed: failures.length === 0,
        mode: choice?.mode,
        query: choice?.query,
        selectedChunkIds,
        failures
      })
    }
  }

  return {
    name: 'multi_turn',
    label: 'multi-turn',
    unit: 'turns',
    passed,
    total,
    cases: results
  }
}

function printFailures(result) {
  for (const testCase of result.cases.filter((item) => !item.passed)) {
    for (const failure of testCase.failures) {
      console.error(`- ${testCase.id}: ${failure}`)
    }
  }
}

async function main() {
  const result = await runBuzzdbMultiturnBenchmark()
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(result))
  } else {
    console.log(`BuzzDB multi-turn benchmark: ${result.passed}/${result.total} turns passed.`)
    printFailures(result)
  }

  if (result.passed !== result.total) {
    process.exitCode = 1
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main()
}
