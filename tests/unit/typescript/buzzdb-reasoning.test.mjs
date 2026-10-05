import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { formatReasoningAnswers, liveBenchmarkOptions, packedEvidenceFromCalls, runReasoningTurns, seededBenchmarkFetch } from '../../benchmarks/buzzdb_reasoning.mjs'
import { formatBenchmarkSummary } from '../../benchmarks/test_buzzdb_retrieval.mjs'
import { buzzdbChunk } from '../../helpers/buzzdb-fixture.mjs'

const families = JSON.parse(readFileSync('tests/benchmarks/buzzdb_reasoning_cases.json', 'utf8'))

test('live model and seed are explicit without changing the default or accepting invalid seeds', () => {
  assert.equal(liveBenchmarkOptions({}).modelName, 'gemma4:e4b')
  assert.equal(liveBenchmarkOptions({}).seed, undefined)
  assert.equal(liveBenchmarkOptions({ TOKENSMITH_BENCHMARK_MODEL: 'gemma4:26b', TOKENSMITH_BENCHMARK_SEED: '19' }).modelName, 'gemma4:26b')
  assert.equal(liveBenchmarkOptions({ TOKENSMITH_BENCHMARK_SEED: '0' }).seed, 0)
  for (const seed of ['', ' ', 'abc', '-1', '1.5', '2147483648']) {
    assert.throws(() => liveBenchmarkOptions({ TOKENSMITH_BENCHMARK_SEED: seed }), /SEED/)
  }
  assert.throws(() => liveBenchmarkOptions({ TOKENSMITH_BENCHMARK_MODEL: ' ' }), /MODEL/)
})

test('paired seeds apply to actual chat requests, not embedding requests or production settings', () => {
  const args = ['http://localhost:11434/api/chat', { body: JSON.stringify({ options: { temperature: 0.7 }, messages: [] }) }]
  const seeded = seededBenchmarkFetch(args, 19)
  assert.deepEqual(JSON.parse(seeded[1].body).options, { temperature: 0.7, seed: 19 })
  assert.equal(JSON.parse(args[1].body).options.seed, undefined)
  assert.equal(seededBenchmarkFetch(args, undefined), args)
  const embed = ['http://localhost:11434/api/embed', args[1]]
  assert.equal(seededBenchmarkFetch(embed, 19), embed)
})

test('six reasoning families have source-backed rubrics for all eighteen turns', () => {
  assert.equal(families.length, 6)
  assert.equal(new Set(families.map(family => family.id)).size, 6)
  for (const family of families) {
    assert.deepEqual(family.turns.map(turn => turn.phase), ['initial', 'followup', 'transfer'])
    assert.ok(family.evidenceGroups.length)
    for (const group of family.evidenceGroups) {
      assert.ok(group.chunkIds.length)
      for (const id of group.chunkIds) buzzdbChunk(id)
    }
    for (const turn of family.turns) {
      assert.ok(turn.question.trim())
      assert.ok(turn.referenceAnswer.trim())
      assert.equal(turn.criteria.length, 4)
    }
  }
})

test('new diagnostic chains have distinct questions, valid passages, and evaluator-only criteria', () => {
  const variants = JSON.parse(readFileSync('tests/benchmarks/buzzdb_reasoning_variants.json', 'utf8'))
  assert.equal(variants.length, 3)
  const originalQuestions = new Set(families.flatMap(family => family.turns.map(turn => turn.question)))
  for (const family of variants) {
    assert.deepEqual(family.turns.map(turn => turn.phase), ['initial', 'followup'])
    for (const group of family.evidenceGroups) for (const id of group.chunkIds) buzzdbChunk(id)
    for (const turn of family.turns) {
      assert.ok(!originalQuestions.has(turn.question))
      assert.ok(turn.referenceAnswer.trim())
      assert.equal(turn.criteria.length, 4)
    }
  }
})

test('live conversations carry their own answers and never expose evaluator fields', async () => {
  const inputs = []
  const results = await runReasoningTurns(families.slice(0, 2), async request => {
    inputs.push(request)
    assert.deepEqual(Object.keys(request).sort(), ['messages', 'prompt'])
    assert.ok(!JSON.stringify(request).includes('referenceAnswer'))
    return { answer: { text: `Actual response ${inputs.length}` } }
  })
  assert.equal(results.length, 6)
  assert.deepEqual(inputs[0].messages, [])
  assert.equal(inputs[1].messages.at(-1).text, 'Actual response 1')
  assert.equal(inputs[2].messages.at(-1).text, 'Actual response 2')
  assert.deepEqual(inputs[3].messages, [], 'New family starts with no old conversation')
  assert.equal(results[1].referenceExchange.answer, 'Actual response 1')
  assert.ok(results.every(row => row.passed && row.reviewStatus === 'ungraded'))
})

test('a failed turn is not retried or replaced by a reference answer', async () => {
  let calls = 0
  const results = await runReasoningTurns(families.slice(0, 2), async () => {
    calls += 1
    if (calls === 1) return { error: 'Output limit', calls: [{ response: { message: { content: 'Incomplete' } } }] }
    return { answer: { text: 'Actual answer' } }
  })
  assert.equal(calls, 4)
  assert.deepEqual(results.map(row => row.status), ['error', 'blocked', 'blocked', 'completed', 'completed', 'completed'])
  assert.equal(results[0].calls[0].response.message.content, 'Incomplete')
  assert.equal(results[1].answer, undefined)
})

test('packed evidence is taken from the final API request, not candidate or history text', () => {
  const sources = [{ context: 'included code' }, { context: 'removed code' }]
  const messages = [{ role: 'user', content: '### Previous exchange\nremoved code\n### Context:\n### Source unit\nText: included code\n### End source unit\n\nQuestion: Q?' }]
  const calls = [
    { request: { messages: [{ role: 'user', content: 'Text: removed code\n### End source unit' }], options: { num_predict: 1 } } },
    { request: { messages, options: { num_predict: 1536 } } }
  ]
  assert.deepEqual(packedEvidenceFromCalls(calls, sources), [sources[0]])
  assert.deepEqual(packedEvidenceFromCalls(calls.slice(0, 1), sources), [])
})

test('reports distinguish completion and retrieved evidence from answer correctness', async () => {
  const rows = await runReasoningTurns(families.slice(0, 1), async () => ({ answer: { text: 'An actual response.' } }))
  const suite = { name: 'reasoning_answers', label: 'Live reasoning execution', model: 'gemma4:e4b',
    passed: 3, total: 3, cases: rows }
  const report = formatReasoningAnswers(suite)
  assert.match(report, /Not an answer-accuracy score/)
  assert.match(report, /Actual answer/)
  assert.match(report, /Reference answer \(not generated\)/)
  assert.match(report, /ungraded/)
  const summary = formatBenchmarkSummary([suite])
  assert.match(summary, /Actual model answer \(ungraded\)/)
  assert.match(summary, /Prior answer \(live conversation\)/)
  assert.doesNotMatch(summary, /Prior answer \(fixture\)/)
  assert.match(summary, /COMPLETED \(ungraded\)/)
  assert.doesNotMatch(summary, /\| PASS \|/)
})
