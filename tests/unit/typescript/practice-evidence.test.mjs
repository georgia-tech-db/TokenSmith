import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { hasSupportingQuote } = requireTranspiledTs('src/shared/practice-evidence.ts')
const { parsePracticeQuestion } = requireTranspiledTs('src/shared/quiz.ts')
const passage = 'An optimizer would choose an index scan if *Cost<sub>IndexScan</sub>* < *Cost<sub>FileScan</sub>*.'
const quote = 'An optimizer would choose an index scan if $Cost_{IndexScan} < Cost_{FileScan}$.'

test('the reported selectivity question accepts the same formula in HTML and LaTeX formatting', () => {
  assert.equal(hasSupportingQuote(passage, quote), true)
  const data = { question: 'An index scan is chosen when its estimated cost is lower than the file scan cost.',
    objective: 'Compare access path costs.', assumptions: [], correctAnswer: true, questionAssessment: 'The rule matches the evidence.', questionIssue: '',
    criteria: [{ id: 'c1', description: 'Cost comparison.', evidence: [{ sourceKey: '5:7:selectivity', quote }] }],
    hint: 'Compare the costs.', explanation: 'Choose the lower estimated cost.' }
  const result = parsePracticeQuestion(JSON.stringify(data), [{ materialId: '5', documentId: 7, chunkId: 'selectivity', excerpt: passage }], 'true_false')
  assert.equal(result.correctAnswer, true)
  assert.equal(result.sources.length, 1)
})

test('presentation tolerance still rejects changed operators, variables, powers, numbers, and claims', () => {
  for (const changed of [quote.replace(' < ', ' > '), quote.replace(' < ', ' <= '), quote.replace('IndexScan', 'IndexSeek'),
    quote.replace('Cost_{IndexScan}', 'Cost^{IndexScan}'), quote.replace('would choose', 'would not choose'),
    quote.replace('Cost_{IndexScan}', 'CostIndexScan'), quote.replace('Cost_{IndexScan}', '2Cost_{IndexScan}')]) {
    assert.equal(hasSupportingQuote(passage, changed), false, changed)
  }
  assert.equal(hasSupportingQuote('Cost = 11.', 'Cost = 10.'), false)
  assert.equal(hasSupportingQuote('`a_b < c`', '`ab < c`'), false)
  assert.equal(hasSupportingQuote('`a * b`', '`a b`'), false)
  assert.equal(hasSupportingQuote(passage, '$$'), false)
})

test('exact code quotes and whitespace differences still work, without paraphrase matching', () => {
  assert.equal(hasSupportingQuote('if (a < b) {\n return false;\n}', 'if (a < b) { return false; }'), true)
  assert.equal(hasSupportingQuote('The *page* contains 4 records.', 'The page contains 4 records.'), true)
  assert.equal(hasSupportingQuote('The page contains 4 records.', 'There are four records on the page.'), false)
})
