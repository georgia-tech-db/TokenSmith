import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { verifiedSeriesAnswerMatches, verifiedSeriesFacts, verifiedSeriesFromQuestion, verifiedSeriesOutputMatches } =
  requireTranspiledTs('src/main/engine/verified-series.ts')
const { runJavascriptInterpret } = requireTranspiledTs('src/main/engine/javascript-interpret.ts')
const { executeIsolatedJavascript } = requireTranspiledTs('src/main/engine/javascript-executor.ts')

const quarterly = 'Quarterly subscriber counts are 1,000, 1,100, 990, and 1,188. List the three consecutive percentage changes in order and their arithmetic mean, to two decimal places.'

test('an explicit ordered series has independently computed signed changes and formatting', () => {
  const series = verifiedSeriesFromQuestion(quarterly)
  assert.deepEqual(series.inputs, [1000, 1100, 990, 1188])
  assert.deepEqual(series.changes, [10, -10, 20])
  assert.equal(series.mean.toFixed(2), '6.67')
  assert.match(verifiedSeriesFacts(series), /10\.00%.*-10\.00%.*20\.00%/s)
  assert.match(verifiedSeriesFacts(series), /Arithmetic mean: 6\.67%/)
  assert.equal(verifiedSeriesOutputMatches(series, '[10,-10]\nNaN'), false)
  assert.equal(verifiedSeriesOutputMatches(series, '[1000,-1000]\n0.00'), false)
  assert.equal(verifiedSeriesOutputMatches(series, '[10,-10,20]\n6.67'), true)
  assert.equal(verifiedSeriesAnswerMatches(series, '10.00%, -10.00%, 20.00%; mean 6.67%'), true)
  assert.equal(verifiedSeriesAnswerMatches(series, '10%, -10%, 20%; mean 6.67%'), false)
})

test('signed differences retain the requested total and reject absolute-value output', () => {
  const question = 'For monthly values 120, 150, 135, and 180, list the consecutive changes, their total, and their average.'
  const series = verifiedSeriesFromQuestion(question)
  assert.deepEqual(series.changes, [30, -15, 45])
  assert.equal(series.total, 60)
  assert.equal(series.mean, 20)
  assert.equal(verifiedSeriesOutputMatches(series, 'From 120 to 150\nFrom 150 to 135\nFrom 135 to 180\nTotal: 90\nAverage: 30'), false)
  assert.match(verifiedSeriesFacts(series), /Total change: 60[\s\S]*Arithmetic mean: 20/)
  assert.match(verifiedSeriesFacts(series), /45\n\nTotal change: 60/)
  assert.match(verifiedSeriesFacts(series), /150 − 135 = -15|135 − 150 = -15/)
  assert.equal(verifiedSeriesFromQuestion('For values 120, 150, 135, and 180 at 7% tax, list consecutive changes and average.'), undefined)
})

test('a repeating mean is marked approximate in a readable fallback', () => {
  const series = verifiedSeriesFromQuestion('For values 1990, 1996, 2000, and 2009, list the consecutive changes and their average.')
  assert.match(verifiedSeriesFacts(series), /Arithmetic mean: approximately 6\.33/)
})

test('exhausted JavaScript attempts fall back to a verified series result', async () => {
  let trace
  const brokenCode = 'const values = [1000,1100,990,1188]; const changes = []; for (let i=1; i<values.length-1; i++) changes.push((values[i]-values[i-1])/values[i-1]*100); console.log(changes); console.log((changes[0]+changes[1]+changes[2])/3)'
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: quarterly }], quarterly,
    async () => JSON.stringify({ tool: 'javascript_interpret', code: brokenCode }),
    executeIsolatedJavascript,
    (value) => { trace = value }
  )
  assert.equal(trace.retryCount, 2)
  assert.equal(trace.verifiedSeriesFallback, true)
  assert.match(answer, /10\.00%/)
  assert.match(answer, /-10\.00%/)
  assert.match(answer, /20\.00%/)
  assert.match(answer, /6\.67%/)
  assert.match(answer, /Formula:/)
  assert.match(answer, /1100 − 1000/)
  assert.match(answer, /divide by 3/)
})

test('one final inference accepts a teaching explanation with Markdown and KaTeX', async () => {
  const calls = []
  let trace
  const explained = String.raw`Use \(\frac{\text{later}-\text{earlier}}{\text{earlier}}\times100\) for each pair.

- \(1000\to1100:\ (1100-1000)/1000\times100=10.00\%\)
- \(1100\to990:\ (990-1100)/1100\times100=-10.00\%\)
- \(990\to1188:\ (1188-990)/990\times100=20.00\%\)

The three changes add to 20.00 percentage points. Divide by 3 to get the **arithmetic mean: 6.67\%**.`
  const responses = [
    JSON.stringify({ tool: 'javascript_interpret', code: 'console.log("[10,-10,20]"); console.log("6.67")' }),
    explained
  ]
  const answer = await runJavascriptInterpret(
    [{ role: 'user', content: quarterly }], quarterly,
    async (messages, _format, maxTokens) => { calls.push({ messages: structuredClone(messages), maxTokens }); return responses.shift() },
    executeIsolatedJavascript,
    value => { trace = value }
  )
  assert.equal(answer, explained)
  assert.equal(calls.length, 2)
  assert.equal(calls[1].maxTokens, 768)
  assert.match(calls[1].messages.at(-1).content, /Authoritative verified inputs, formula, substitutions/)
  assert.match(calls[1].messages.at(-1).content, /1100 − 1000/)
  assert.equal(trace.verifiedSeriesFallback, undefined)
})
