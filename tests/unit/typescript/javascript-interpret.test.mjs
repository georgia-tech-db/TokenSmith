import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { executeIsolatedJavascript, javascriptLimits } = requireTranspiledTs('src/main/engine/javascript-executor.ts')
const { formatComputedResult, missingComputedList, missingRequestedList, normalizeMathCommands, outputPrecisionIssue, runJavascriptInterpret, unsupportedFinalNumbers } = requireTranspiledTs('src/main/engine/javascript-interpret.ts')

test('double-escaped TeX commands are repaired only inside math delimiters', () => {
  const output = String.raw`The average is $$\\frac{30}{8}=\\boxed{3.75}$$; literal \\frac stays in prose.`
  assert.equal(normalizeMathCommands(output), String.raw`The average is $$\frac{30}{8}=\boxed{3.75}$$; literal \\frac stays in prose.`)
})

test('QuickJS executes general JavaScript with no host APIs and clean state', async () => {
  const calculation = await executeIsolatedJavascript('[8, 12, 12, 20, 3].reduce((sum, x) => sum + x, 0)')
  assert.equal(calculation.ok, true)
  assert.equal(calculation.result, '55')

  const globals = await executeIsolatedJavascript(
    '[typeof process, typeof require, typeof fetch, typeof XMLHttpRequest, typeof WebSocket].join(",")'
  )
  assert.equal(globals.result, 'undefined,undefined,undefined,undefined,undefined')
  await executeIsolatedJavascript('globalThis.secret = 42; secret')
  const clean = await executeIsolatedJavascript('typeof secret')
  assert.equal(clean.result, 'undefined')

  const logged = await executeIsolatedJavascript('const gaps = [3, 3, 7]; console.log("Gaps:", gaps); console.log("Average:", 13 / 3)')
  assert.equal(logged.ok, true)
  assert.match(logged.result, /Gaps: \[3,3,7\]/)
  assert.match(logged.result, /Average: 4\.333/)

  const years = await executeIsolatedJavascript(`
    const years = [1986, 1989, 1992, 1999, 2003, 2006, 2008, 2011, 2016];
    const gaps = years.slice(1).map((year, index) => year - years[index]);
    console.log('Gaps:', gaps);
    console.log('Average:', gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length);
  `)
  assert.equal(years.ok, true)
  assert.match(years.result, /Gaps: \[3,3,7,4,3,2,3,5\]/)
  assert.match(years.result, /Average: 3\.75/)
})

test('QuickJS limits code, output and execution time', async () => {
  const largeCode = await executeIsolatedJavascript(' '.repeat(javascriptLimits.codeBytes + 1))
  assert.equal(largeCode.ok, false)
  assert.match(largeCode.error, /Code exceeds/)

  const largeOutput = await executeIsolatedJavascript('"x".repeat(9000)')
  assert.equal(largeOutput.ok, false)
  assert.match(largeOutput.error, /Output exceeds/)

  const largeLog = await executeIsolatedJavascript('console.log("x".repeat(9000))')
  assert.equal(largeLog.ok, false)
  assert.match(largeLog.error, /Output exceeds/)

  const loop = await executeIsolatedJavascript('while (true) {}')
  assert.equal(loop.ok, false)
  assert.match(loop.error, /timed out/)

  const missingResult = await executeIsolatedJavascript('const answer = 42')
  assert.equal(missingResult.ok, false)
  assert.match(missingResult.error, /defined result/)

  const nonFiniteOutput = await executeIsolatedJavascript('const values = [10, -10]; console.log(values); console.log((values[0] + values[1] + values[2]) / 3)')
  assert.equal(nonFiniteOutput.ok, false)
  assert.match(nonFiniteOutput.error, /NaN or Infinity/)
  assert.equal(nonFiniteOutput.partialOutput, '[10,-10]\nnull')
})

test('execution error is returned to same model, then success leads to final answer', async () => {
  const seen = []
  let trace
  const answers = [
    JSON.stringify({ tool: 'javascript_interpret', code: 'missingValue + 1' }),
    JSON.stringify({ tool: 'javascript_interpret', code: '40 + 2' }),
    '42'
  ]
  const text = await runJavascriptInterpret(
    [{ role: 'user', content: 'What is 40 + 2?' }],
    'What is 40 + 2?',
    async (messages, format, maxTokens) => { seen.push({ messages: structuredClone(messages), format, maxTokens }); return answers.shift() },
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.equal(text, formatComputedResult('42', '40 + 2'))
  assert.match(seen[1].messages.at(-2).content, /javascript_interpret error:/)
  assert.match(seen[2].messages.at(-1).content, /javascript_interpret result:\n42/)
  assert.equal(seen[2].messages.at(-1).role, 'user')
  assert.equal(seen[2].format, undefined)
  assert.equal(seen[2].maxTokens, 256)
  assert.equal(trace.retryCount, 1)
  assert.equal(trace.inferenceCount, 3)
  assert.equal(trace.attempts.length, 2)
})

test('a direct model answer skips JavaScript execution and quantitative verification', async () => {
  let trace
  let executionCount = 0
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: 'What is 6 times 7?' }], 'What is 6 times 7?',
    async (_messages, format) => {
      assert.deepEqual(format.properties.route.enum, ['answer', 'javascript_interpret'])
      return JSON.stringify({ route: 'answer', content: '6 × 7 = 42.' })
    },
    async () => { executionCount += 1; throw new Error('JavaScript must not run') },
    value => { trace = value }
  )
  assert.equal(answer, '6 × 7 = 42.')
  assert.equal(executionCount, 0)
  assert.equal(trace.directAnswer, true)
  assert.equal(trace.inferenceCount, 1)
  assert.deepEqual(trace.attempts, [])
  assert.equal(trace.verifiedSeriesOperation, undefined)
  assert.equal(trace.verifiedCalculationOperation, undefined)
})

test('a quoted newline syntax error gives a generic separate-log repair hint', async () => {
  const seen = []
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("first\nsecond")' }),
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("first"); console.log("second")' }),
    'first\nsecond'
  ]
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: 'Print first and second.' }], 'Print first and second.',
    async messages => { seen.push(structuredClone(messages)); return responses.shift() },
    executeIsolatedJavascript, () => {}
  )
  assert.match(seen[1].at(-2).content, /separate console\.log calls/)
  assert.match(answer, /first.*second/s)
  assert.equal(seen.length, 3)
})

test('failed execution stops after two retries with no final inference', async () => {
  let calls = 0
  let trace
  await assert.rejects(runJavascriptInterpret(
    [{ role: 'user', content: 'Calculate' }],
    'Calculate',
    async () => { calls += 1; return JSON.stringify({ tool: 'javascript_interpret', code: 'unknownName' }) },
    executeIsolatedJavascript,
    (value) => { trace = value }
  ), /failed after 3 attempts/)
  assert.equal(calls, 3)
  assert.equal(trace.retryCount, 2)
  assert.equal(trace.inferenceCount, 3)
})

test('final answer cannot introduce arithmetic absent from the question and tool result', async () => {
  const question = 'Years: 1986, 1989, 1992, 1999, 2003, 2006, 2008, 2011, 2016. List gaps and average.'
  const result = 'Gaps: 3, 3, 7, 4, 3, 2, 3, 5; Average: 3.75 years'
  assert.deepEqual(unsupportedFinalNumbers('The gaps average 3.75 years.', question, result), [])
  assert.deepEqual(unsupportedFinalNumbers('The sum is 31, so 31/8 = 3.875.', question, result), [31, 8, 3.875])
  assert.deepEqual(unsupportedFinalNumbers('### Step 4: Explain the average\n4. Result: 3.75 years', question, result), [])
  assert.deepEqual(unsupportedFinalNumbers('### Step 4: Explain the average\nThe average is 3.875 years', question, result), [3.875])

  let trace
  const answers = [
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("Gaps: 3, 3, 7, 4, 3, 2, 3, 5; Average: 3.75 years")' }),
    'The sum is 31, so 31/8 = 3.875. The average is 3.75 years.'
  ]
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: `Unrelated source number: 31.\nQuestion: ${question}` }],
    question,
    async () => answers.shift(),
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.equal(answer, formatComputedResult(result, 'console.log("Gaps: 3, 3, 7, 4, 3, 2, 3, 5; Average: 3.75 years")'))
  assert.deepEqual(trace.unsupportedFinalNumbers, [31, 8, 3.875])
  assert.equal(trace.inferenceCount, 2)
  assert.equal(trace.inferenceLatenciesMs.length, 2)
})

test('computed result fallback keeps separate output lines readable', () => {
  assert.equal(formatComputedResult('Gaps: 3, 3, 7\nAverage: 3.75 years'),
    '**Computed result**\n\nGaps: 3, 3, 7  \nAverage: 3.75 years')
})

test('empty final response returns the successful tool result', async () => {
  let trace
  const responses = [JSON.stringify({ tool: 'javascript_interpret', code: '40 + 2' }), '']
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: 'What is 40 + 2?' }],
    'What is 40 + 2?',
    async () => responses.shift(),
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.equal(answer, formatComputedResult('42', '40 + 2'))
  assert.equal(trace.emptyFinalAnswer, true)
  assert.equal(trace.inferenceCount, 2)
})

test('a requested list missing from a scalar result triggers bounded code repair', async () => {
  const question = 'List the consecutive gaps and the average for 1986, 1989, 1992.'
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("Average: 3")' }),
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("Gaps: 3, 3"); console.log("Average: 3")' }),
    'Gaps: 3, 3\nAverage: 3'
  ]
  const seen = []
  let trace
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: question }], question,
    async (messages) => { seen.push(structuredClone(messages)); return responses.shift() },
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.equal(answer, 'Gaps: 3, 3\nAverage: 3')
  assert.equal(trace.attempts[0].incompleteResult, true)
  assert.match(seen[1].at(-1).content, /does not include every requested list item/)
  assert.equal(trace.inferenceCount, 3)
  assert.equal(trace.retryCount, 1)
  assert.equal(missingRequestedList(question, 'Average: 3'), true)
  assert.equal(missingRequestedList(question, 'Gaps: 3, 3\nAverage: 3'), false)
})

test('a short explicit numeric list and unrounded output trigger bounded code repair', async () => {
  const percentQuestion = 'List the three consecutive percentage changes and their mean.'
  assert.equal(missingRequestedList(percentQuestion, '[10,-10]'), true)
  assert.equal(missingRequestedList(percentQuestion, '[10,-10,20]\nMean: 6.67'), false)
  assert.equal(outputPrecisionIssue('Round money to cents.', 'Notebooks: 59.849999999999994'),
    'The question requests values rounded to cents or two decimal places, but the output contains a longer decimal.')
  assert.equal(outputPrecisionIssue('Round money to cents.', 'Notebooks: 59.85'), undefined)
  assert.equal(outputPrecisionIssue('Round money to cents.',
    'Tax: 7.000000000000001% x $116.35 = $8.14'), undefined)
  assert.match(outputPrecisionIssue('Report percentages to two decimal places.',
    'Tax: 7.000000000000001%') ?? '', /longer decimal/)

  const question = 'An invoice has 3 notebooks at $19.95. Calculate the line total and round money to cents.'
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("Notebooks: " + (3 * 19.95))' }),
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("Notebooks: " + (3 * 19.95).toFixed(2))' }),
    'Notebooks: 59.85'
  ]
  const seen = []
  let trace
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: question }], question,
    async (messages) => { seen.push(structuredClone(messages)); return responses.shift() },
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.match(seen[1].at(-1).content, /rounded to cents/)
  assert.equal(trace.retryCount, 1)
  assert.equal(trace.attempts[0].incompleteResult, true)
  assert.match(answer, /Notebooks: 59\.85/)
})

test('a percentage-rate display does not exhaust cents-only retries', async () => {
  const question = 'An invoice has 3 notebooks at $19.95 each, 2 filters at $20.50 each, and 5 vials at $3.10 each. List each line total, then calculate the subtotal, 7% tax, and final total. Round money to cents.'
  const code = 'const items = [{name: "notebooks", quantity: 3, price: 19.95}, {name: "filters", quantity: 2, price: 20.50}, {name: "vials", quantity: 5, price: 3.10}]; let subtotal = 0; for (const item of items) { const total = item.quantity * item.price; subtotal += total; console.log(`${item.name}: $${total.toFixed(2)}`); } const taxRate = 7 / 100; const tax = subtotal * taxRate; console.log(`Subtotal: $${subtotal.toFixed(2)}`); console.log(`Tax: ${taxRate * 100}% = $${tax.toFixed(2)}`); console.log(`Final total: $${(subtotal + tax).toFixed(2)}`);'
  const result = 'notebooks: $59.85\nfilters: $41.00\nvials: $15.50\nSubtotal: $116.35\nTax: 7.000000000000001% = $8.14\nFinal total: $124.49'
  const responses = [JSON.stringify({ tool: 'javascript_interpret', code }), result]
  let trace
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: question }], question,
    async () => responses.shift(), executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.equal(trace.retryCount, 0)
  assert.equal(trace.attempts.length, 1)
  assert.equal(trace.attempts[0].incompleteResult, undefined)
  assert.equal(trace.verifiedCalculationOperation, 'invoice-total')
  assert.equal(trace.verifiedCalculationFallback, true)
  assert.match(answer, /Tax: \$116\.35 × 7% = \$8\.14/)
  assert.doesNotMatch(answer, /7\.000000000000001%/)
})

test('a failed percentage calculation returns the short list to the model for repair', async () => {
  const question = 'Quarterly counts are 1000, 1100, 990, 1188. List the three consecutive percentage changes and their mean to two decimal places.'
  const brokenCode = 'const counts = [1000, 1100, 990, 1188]; const changes = []; for (let i = 1; i < counts.length - 1; i++) changes.push((counts[i] - counts[i-1]) / counts[i-1] * 100); console.log(changes); console.log(((changes[0] + changes[1] + changes[2]) / 3).toFixed(2))'
  const repairedCode = 'const counts = [1000, 1100, 990, 1188]; const changes = []; for (let i = 1; i < counts.length; i++) changes.push((counts[i] - counts[i-1]) / counts[i-1] * 100); console.log("Changes: " + changes.map(value => value.toFixed(2) + "%").join(", ")); console.log("Mean: " + (changes.reduce((sum, value) => sum + value, 0) / changes.length).toFixed(2) + "%")'
  const result = 'Changes: 10.00%, -10.00%, 20.00%\nMean: 6.67%'
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code: brokenCode }),
    JSON.stringify({ tool: 'javascript_interpret', code: repairedCode }),
    result
  ]
  const seen = []
  let trace
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: question }], question,
    async (messages) => { seen.push(structuredClone(messages)); return responses.shift() },
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.match(seen[1].at(-2).content, /Output before failure:\n\[10,-10\]/)
  assert.match(seen[1].at(-2).content, /numeric list has fewer items/)
  assert.equal(trace.attempts[0].execution.ok, false)
  assert.equal(trace.retryCount, 1)
  assert.equal(answer, result)
})

test('final response cannot silently omit a repeated computed list value', async () => {
  const result = 'Gaps: 3, 3, 7, 4, 3, 2, 3, 5\nAverage: 3.75'
  assert.equal(missingComputedList('Gaps: 3, 3, 7, 4, 3, 2, 5. Average: 3.75', result), true)
  assert.equal(missingComputedList('Gaps: 3, 3, 7, 4, 3, 2, 3, 5. Average: 3.75', result), false)
  const formatted = formatComputedResult(result, 'const gaps = [3, 3, 7]; console.log(gaps)')
  assert.match(formatted, /\*\*Calculation \(executed JavaScript\)\*\*\n\n```js\n/)

  let trace
  const code = 'console.log("Gaps: 3, 3, 7, 4, 3, 2, 3, 5"); console.log("Average: 3.75")'
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code }),
    'Gaps: 3, 3, 7, 4, 3, 2, 5. Average: 3.75.'
  ]
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: 'List the gaps and average.' }],
    'List the gaps and average.',
    async () => responses.shift(),
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.equal(answer, formatComputedResult(result, code))
  assert.equal(trace.finalMissingComputedList, true)
})

test('a wrong verbal method cannot replace the executed calculation', async () => {
  const code = 'const gaps = [3, 3, 7, 4, 3, 2, 3, 5]; console.log("Gaps: " + gaps.join(", ")); console.log("Average: " + (gaps.reduce((a, b) => a + b, 0) / gaps.length))'
  const result = 'Gaps: 3, 3, 7, 4, 3, 2, 3, 5\nAverage: 3.75'
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code }),
    'Gaps: 3, 3, 7, 4, 3, 2, 3, 5. Divide by the number of gaps minus one. Average: 3.75.'
  ]
  let trace
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: 'List the gaps and average.' }],
    'List the gaps and average.',
    async () => responses.shift(),
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.equal(answer, formatComputedResult(result, code))
  assert.equal(trace.finalOutputMismatch, true)
  assert.match(trace.rejectedFinalAnswer, /minus one/)
})
