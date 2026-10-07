import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  verifiedCalculationFromQuestion,
  verifiedCalculationOutputMatches,
  verifiedCalculationAnswerMatches,
  verifiedCalculationTeachingFacts
} = requireTranspiledTs('src/main/engine/verified-calculation.ts')
const { runJavascriptInterpret } = requireTranspiledTs('src/main/engine/javascript-interpret.ts')
const { executeIsolatedJavascript } = requireTranspiledTs('src/main/engine/javascript-executor.ts')
const cases = [
  { id: 'invoice-total', prompt: 'Using the invoice record, state the quantity and unit price for each item, list each line total, then calculate the subtotal, 7% tax, and final total. Round monetary values to cents.',
    sources: [{ excerpt: 'The invoice has three lines: 3 laboratory notebooks at $19.95 each; 2 replacement filters at $20.50 each; and 5 sample vials at $3.10 each. Sales tax is 7%.' }], expected: { calculation: [59.85, 41, 15.5, 116.35, 8.14, 124.49] } },
  { id: 'weighted-course-score', prompt: 'From the section results, state the section sizes and scores, list each weighted contribution, and calculate the overall weighted average score.',
    sources: [{ excerpt: 'Section A has 30 students with an average score of 80. Section B has 45 students with an average score of 70. Section C has 25 students with an average score of 90.' }], expected: { calculation: [2400, 3150, 2250, 78] } },
  { id: 'quarterly-percent-changes', prompt: 'Use the quarterly subscriber counts. State the four counts, list the three consecutive percentage changes in order, and calculate their arithmetic mean. Report percentages to two decimal places.',
    sources: [{ excerpt: 'Subscriber counts were Q1: 1,000; Q2: 1,100; Q3: 990; Q4: 1,188.' }], expected: { calculation: [10, -10, 20, 6.67] } },
  { id: 'average-speed', prompt: "Using the trip log, state each segment's distance and time, calculate total distance and total time, then calculate the trip's average speed in miles per hour.",
    sources: [{ excerpt: 'Segment 1 covered 60 miles in 1.0 hour. Segment 2 covered 45 miles in 0.75 hour. Segment 3 covered 45 miles in 1.25 hours.' }], expected: { calculation: [150, 3, 50] } },
  { id: 'break-even', prompt: 'Use the product plan. State the fixed cost, price, and variable cost; calculate contribution margin per unit; and calculate the whole number of units needed to break even.',
    sources: [{ excerpt: 'Monthly fixed cost is $1,200. The selling price is $30 per unit and variable cost is $18 per unit.' }], expected: { calculation: [12, 100] } },
  { id: 'temperature-conversion', prompt: 'Convert 32°F, 68°F, and 95°F to Celsius. List the three Celsius values in the same order, then calculate their average in Celsius.', expected: { calculation: [18.33] } },
  { id: 'expected-value', prompt: "Using the payoff table, state each payoff and probability, list each payoff's contribution to expected value, and calculate the total expected value in dollars.",
    sources: [{ excerpt: 'The three possible payoffs are $0 with probability 0.20, $50 with probability 0.50, and $100 with probability 0.30.' }], expected: { calculation: [0, 25, 30, 55] } },
  { id: 'incorrect-average-claim', prompt: 'A student claims that the arithmetic mean of 2, 4, and 6 is 5. Is the claim correct? Show the calculation and give the correct mean.', expected: { calculation: [4] } }
]

test('contracts independently compute every supported operation', () => {
  for (const item of cases) {
    const contract = verifiedCalculationFromQuestion(item.prompt, (item.sources ?? []).map((source) => source.excerpt))
    assert.ok(contract, item.id)
    assert.equal(verifiedCalculationOutputMatches(contract, contract.facts), true, item.id)
    assert.equal(verifiedCalculationAnswerMatches(contract, contract.facts), true, item.id)
    for (const expected of item.expected.calculation) {
      assert.ok(contract.expectedOutput.some((value) => Math.abs(value - expected) < 0.005), item.id)
    }
  }
})

test('a calculation never mixes operands from different retrieved excerpts', () => {
  const question = 'Using the invoice record, list each line total, subtotal, tax, and final total.'
  const first = 'The invoice has 2 books at $5.00 each. Sales tax is 7%.'
  const second = 'The invoice has 3 books at $9.00 each. Sales tax is 7%.'
  assert.equal(verifiedCalculationFromQuestion(question, [first, second]), undefined)
  assert.equal(verifiedCalculationFromQuestion(question, [first]).expectedOutput.at(-1), 10.7)
})

test('explicit invoice and weighted-average inputs in the question use the verified tutor path', async () => {
  const prompts = [
    ['An invoice has 3 notebooks at $19.95 each, 2 filters at $20.50 each, and 5 vials at $3.10 each. List each line total, then calculate the subtotal, 7% tax, and final total. Round money to cents.', 'invoice-total', 124.49],
    ['Section A has 30 students averaging 80, section B has 45 averaging 70, and section C has 25 averaging 90. List each section’s weighted contribution and calculate the overall weighted average.', 'weighted-mean', 78]
  ]
  for (const [question, operation, finalValue] of prompts) {
    const contract = verifiedCalculationFromQuestion(question)
    assert.equal(contract?.operation, operation)
    assert.equal(contract.conclusionValue, finalValue)
    const responses = [
      JSON.stringify({ tool: 'javascript_interpret', code: `console.log("${contract.expectedOutput.join(', ')}")` }),
      verifiedCalculationTeachingFacts(contract)
    ]
    let trace
    const seen = []
    const answer = await runJavascriptInterpret(
      [{ role: 'user', content: question }], question,
      async messages => { seen.push(structuredClone(messages)); return responses.shift() },
      executeIsolatedJavascript, value => { trace = value }
    )
    assert.equal(answer, verifiedCalculationTeachingFacts(contract))
    assert.equal(trace.inferenceCount, 2)
    assert.match(seen[1].at(-1).content, /Authoritative verified inputs, method/)
    assert.equal(trace.verifiedCalculationOperation, operation)
    assert.equal(trace.verifiedCalculationFallback, undefined)
  }
})

test('wrong executable arithmetic is repaired or replaced by verified facts', async () => {
  const item = cases.find((entry) => entry.id === 'weighted-course-score')
  const source = item.sources[0].excerpt
  let trace
  const wrong = JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("Weighted contributions: 24, 31.5, 22.5; average: 78")' })
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: item.prompt }], item.prompt,
    async () => wrong,
    executeIsolatedJavascript,
    (value) => { trace = value },
    [source]
  )
  assert.equal(trace.retryCount, 2)
  assert.equal(trace.verifiedCalculationFallback, true)
  assert.match(answer, /2400 weighted points/)
  assert.match(answer, /3150 weighted points/)
  assert.match(answer, /2250 weighted points/)
  assert.doesNotMatch(answer, /31\.5/)
  assert.match(answer, /\*\*Method:\*\*/)
  assert.match(answer, /divide by the total number of students/)
})

test('a listed value cannot masquerade as the requested average', async () => {
  const item = cases.find((entry) => entry.id === 'temperature-conversion')
  const contract = verifiedCalculationFromQuestion(item.prompt)
  const wrong = '32°F = 0°C; 68°F = 20°C; 95°F = 35°C. Average Celsius temperature: 35°C.'
  assert.equal(verifiedCalculationAnswerMatches(contract, wrong), false)
  assert.equal(verifiedCalculationAnswerMatches(contract, contract.facts), true)

  let trace
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("0, 20, 35"); console.log("Average: 18.33")' }),
    wrong
  ]
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: item.prompt }], item.prompt,
    async () => responses.shift(), executeIsolatedJavascript,
    value => { trace = value }
  )
  assert.equal(trace.verifiedCalculationFallback, true)
  assert.match(answer, /Average Celsius temperature: 18\.33°C/)
})

test('verified fallback renders near-integer time and repeating speed cleanly', () => {
  const question = "Using the trip log, state each segment's distance and time, calculate total distance and total time, then calculate the trip's average speed in miles per hour."
  const nearInteger = verifiedCalculationFromQuestion(question, [
    'Segment 1 covered 40 miles in 0.8 hours. Segment 2 covered 80 miles in 1.6 hours. Segment 3 covered 30 miles in 0.6 hours.'
  ])
  assert.match(nearInteger.facts, /Total time: 3 hours\. Average speed: 150 ÷ 3 = 50 miles per hour/)
  assert.doesNotMatch(nearInteger.facts, /999999|0000000000/)

  const repeating = verifiedCalculationFromQuestion(question, [
    'Segment 1 covered 100 miles in 2 hours. Segment 2 covered 60 miles in 1 hour.'
  ])
  assert.match(repeating.facts, /approximately 53\.33 miles per hour/)
})

test('one final inference keeps a differently structured verified teaching answer', async () => {
  const item = cases.find((entry) => entry.id === 'invoice-total')
  const source = item.sources[0].excerpt
  const contract = verifiedCalculationFromQuestion(item.prompt, [source])
  assert.match(verifiedCalculationTeachingFacts(contract), /Multiply each quantity by its unit price/)
  const explanation = String.raw`### Work through the invoice

Multiply quantity by price: notebooks \(3\times\$19.95=\$59.85\), filters \(2\times\$20.50=\$41.00\), and vials \(5\times\$3.10=\$15.50\).

The subtotal is \$116.35. Applying 7% tax gives \$8.14. Add the rounded tax to the subtotal.

**Amount due: \$124.49.**`
  assert.equal(verifiedCalculationAnswerMatches(contract, explanation), true)
  assert.equal(verifiedCalculationAnswerMatches(contract, explanation.replace('Amount due: \\$124.49', 'Amount due: \\$124.50')), false)

  const calls = []
  let trace
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("59.85, 41.00, 15.50, 116.35, 8.14, 124.49")' }),
    explanation
  ]
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: item.prompt }], item.prompt,
    async (messages, _format, maxTokens) => { calls.push({ messages: structuredClone(messages), maxTokens }); return responses.shift() },
    executeIsolatedJavascript, value => { trace = value }, [source]
  )
  assert.equal(answer, explanation)
  assert.equal(calls.length, 2)
  assert.equal(calls[1].maxTokens, 768)
  assert.match(calls[1].messages.at(-1).content, /Authoritative verified inputs, method, intermediate results/)
  assert.equal(trace.verifiedCalculationFallback, undefined)
})

test('a changed claim verdict is rejected while equivalent tutoring wording passes', () => {
  const item = cases.find((entry) => entry.id === 'incorrect-average-claim')
  const contract = verifiedCalculationFromQuestion(item.prompt)
  assert.equal(verifiedCalculationAnswerMatches(contract, contract.facts.replace('is incorrect', 'is correct')), false)
  assert.equal(verifiedCalculationAnswerMatches(contract, contract.facts.replace('is incorrect', 'is wrong')), true)
})
