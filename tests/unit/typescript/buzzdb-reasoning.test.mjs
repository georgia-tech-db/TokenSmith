import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { formatReasoningAnswers, packedEvidenceFromCalls, runReasoningTurns } from '../../benchmarks/buzzdb_reasoning.mjs'
import { formatBenchmarkSummary } from '../../benchmarks/test_buzzdb_retrieval.mjs'
import { buzzdbChunk } from '../../helpers/buzzdb-fixture.mjs'

const families = JSON.parse(readFileSync('tests/benchmarks/buzzdb_reasoning_cases.json', 'utf8'))

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
