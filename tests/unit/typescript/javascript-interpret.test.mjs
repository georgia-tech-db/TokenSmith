import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { runJavascriptInterpret } = requireTranspiledTs('src/main/engine/javascript-interpret.ts')
const { executeIsolatedJavascript, javascriptLimits } = requireTranspiledTs('src/main/engine/javascript-executor.ts')
test('QuickJS exposes only bounded console output and no host APIs, with clean state', async () => {
  assert.equal((await executeIsolatedJavascript('[8,12,12,20,3].reduce((sum,x)=>sum+x,0)')).result, '55')
  assert.equal((await executeIsolatedJavascript('[typeof process,typeof require,typeof fetch,typeof XMLHttpRequest,typeof WebSocket].join(",")')).result,
    'undefined,undefined,undefined,undefined,undefined')
  await executeIsolatedJavascript('globalThis.secret=42; secret')
  assert.equal((await executeIsolatedJavascript('typeof secret')).result, 'undefined')
  assert.equal((await executeIsolatedJavascript('console.log("values", [3,7]); console.log(10/2)')).result, 'values [3,7]\n5')
})

test('QuickJS retains code, output, deadline and finite-result limits', async () => {
  for (const [code, error] of [
    [' '.repeat(javascriptLimits.codeBytes + 1), /Code exceeds/],
    ['"x".repeat(9000)', /Output exceeds/], ['console.log("x".repeat(9000))', /Output exceeds/],
    ['while(true){}', /timed out/], ['const result=42', /defined result/], ['console.log(0/0)', /NaN or Infinity/]
  ]) {
    const execution = await executeIsolatedJavascript(code)
    assert.equal(execution.ok, false)
    assert.match(execution.error, error)
  }
})

const question = 'Readings are 12, 18, 15 and 27. List the signed changes, total and mean.'
const code = 'const v=[12,18,15,27],changes=[]; for(let i=1;i<v.length;i++){changes.push(v[i]-v[i-1]);} const total=changes.reduce((a,b)=>a+b,0); console.log({changes,total,mean:total/changes.length});'
const explanation = '**Changes:** 18 − 12 = 6; 15 − 18 = -3; 27 − 15 = 12.\n\nAdd the signed changes for a total of 15. Divide by 3 intervals for a mean of 5.'
const call = (code, fenced = true) => `I will calculate the requested values.\n<javascript_interpret>\n${fenced ? '```javascript\n' : ''}${code}${fenced ? '\n```' : ''}\n</javascript_interpret>`

async function run(responses, execute = executeIsolatedJavascript, initial = [{ role: 'user', content: question }]) {
  const seen = []
  let trace
  const answer = await runJavascriptInterpret(initial,
    async (messages, maxTokens) => {
      seen.push({ messages: structuredClone(messages), maxTokens })
      assert.ok(responses.length, 'Unexpected extra inference')
      return responses.shift()
    }, execute, value => { trace = value })
  return { answer, trace, seen }
}

test('Advanced Math executes ordinary loops and uses two calls with plain teaching narration', async () => {
  const { answer, trace, seen } = await run([call(code), explanation])
  assert.equal(answer, explanation)
  assert.equal(trace.inferenceCount, 2)
  assert.equal(seen[0].maxTokens, 1536)
  assert.equal(trace.retryCount, 0)
  assert.equal(trace.attempts[0].execution.result, '{"changes":[6,-3,12],"total":15,"mean":5}')
  assert.equal(seen[1].messages.at(-2).content, call(code))
  assert.match(seen[1].messages.at(-1).content, /javascript_interpret result:/)
  assert.match(seen[1].messages[0].content, /student-facing Markdown/)
  assert.equal(seen[0].messages[0].content, seen[1].messages[0].content)
  assert.equal(trace.attempts[0].plan, 'I will calculate the requested values.')
  assert.equal(trace.inferenceLatenciesMs.length, 2)
})

test('runtime errors go to the same conversation for bounded repair, then explanation', async () => {
  const { answer, trace, seen } = await run([call('console.log(missingValue)'), call(code), explanation])
  assert.equal(answer, explanation)
  assert.equal(trace.inferenceCount, 3)
  assert.equal(trace.retryCount, 1)
  assert.equal(trace.attempts[0].execution.ok, false)
  assert.match(seen[1].messages.at(-1).content, /javascript_interpret error:.*\n.*missingValue/s)
  assert.equal(seen[1].messages.at(-2).content, call('console.log(missingValue)'))
  assert.match(seen[1].messages.at(-1).content, /ReferenceError at line 1/)
  assert.match(seen[1].messages.at(-1).content, /1: console.log\(missingValue\)/)
  assert.equal(seen[0].messages[0].content, seen[1].messages[0].content)
  assert.equal(seen[0].messages[0].content, seen[2].messages[0].content)
  assert.equal(seen[2].format, undefined)
})

test('three failed executions stop without a final narration or successful partial result', async () => {
  const bad = call('console.log(123); throw new Error("unfinished")')
  const { answer, trace } = await run([bad, bad, bad])
  assert.equal(trace.inferenceCount, 3)
  assert.equal(trace.retryCount, 2)
  assert.equal(trace.attempts.length, 3)
  assert.match(answer, /couldn't complete.*3 JavaScript attempts/)
  assert.doesNotMatch(answer, /123/)
  assert.match(trace.attempts[0].execution.error, /1: console.log\(123\)/)
})

test('malformed or partial tool blocks receive bounded feedback without executing prose', async () => {
  for (const bad of ['', 'Here is the code.', '{"tool":"javascript_interpret"}',
    '<javascript_interpret>\nconsole.log(1)', '<javascript_interpret>\n</javascript_interpret>',
    '<javascript_interpret>\n```javascript\nconsole.log(1)\n</javascript_interpret>']) {
    const { answer, trace } = await run([bad, bad, bad], async () => { throw new Error('Must not execute') })
    assert.equal(trace.inferenceCount, 3)
    assert.ok(trace.attempts.every(attempt => attempt.phase === 'plan'))
    assert.match(answer, /couldn't complete/)
  }
})

test('the first complete tool block executes while speculative prose or further calls are discarded', async () => {
  const first = call('console.log(7*3)')
  for (const tail of ['The result is 999.', call('console.log(999)')]) {
    const { answer, trace, seen } = await run([first + '\n' + tail, 'Multiplying 7 by 3 gives 21.'])
    assert.equal(answer, 'Multiplying 7 by 3 gives 21.')
    assert.equal(trace.inferenceCount, 2)
    assert.equal(trace.retryCount, 0)
    assert.equal(trace.attempts.length, 1)
    assert.equal(trace.attempts[0].code, 'console.log(7*3)')
    assert.equal(trace.attempts[0].execution.result, '21')
    assert.equal(trace.attempts[0].modelResponse, first + '\n' + tail)
    assert.equal(trace.attempts[0].discardedTrailingText, tail)
    assert.equal(seen[1].messages.at(-2).content, first)
    assert.doesNotMatch(seen[1].messages.at(-2).content, /999/)
    assert.match(seen[1].messages.at(-1).content, /result:\n21/)
    assert.match(seen[1].messages[0].content, /restate every requested result/)
  }
})

test('the model may correct successful but incomplete output within the same execution limit', async () => {
  const { answer, trace, seen } = await run([call('console.log(15)'), call(code), explanation])
  assert.equal(answer, explanation)
  assert.equal(trace.inferenceCount, 3)
  assert.equal(trace.retryCount, 1)
  assert.ok(trace.attempts.every(attempt => attempt.execution.ok))
  const exhausted = await run([call(code), call(code), call(code), call(code)])
  assert.equal(exhausted.trace.attempts.length, 3)
  assert.equal(exhausted.trace.inferenceCount, 4)
  assert.match(exhausted.answer, /retry limit/)
})

test('each retry gets a fresh sandbox and no host APIs', async () => {
  const { answer, trace } = await run([
    call('globalThis.leak=42; throw new Error("retry")'),
    call('console.log([typeof leak,typeof process,typeof require,typeof fetch].join(","))'),
    'The sandbox has no previous state or host APIs.'
  ])
  assert.equal(answer, 'The sandbox has no previous state or host APIs.')
  assert.equal(trace.attempts[1].execution.result, 'undefined,undefined,undefined,undefined')
})

test('empty narration falls back to executed output with no extra model call or code dump', async () => {
  const { answer, trace } = await run([call(code), ''])
  assert.equal(trace.inferenceCount, 2)
  assert.equal(trace.narrationFallback, true)
  assert.match(answer, /Executed result/)
  assert.match(answer, /"mean":5/)
  assert.doesNotMatch(answer, /```|for\(|independently verified/)
})

test('packed source messages and custom study instructions remain visible during execution and explanation', async () => {
  const initial = [
    { role: 'system', content: 'Use the supplied textbook.' },
    { role: 'user', content: '### Source unit\nText: Capacity is 12 per batch; there are 5 batches.\n### End source unit\nQuestion: Calculate capacity.' }
  ]
  const { seen } = await run([call('console.log(12*5)'), 'Capacity is 60.'], executeIsolatedJavascript, initial)
  for (const inference of seen) {
    assert.equal(inference.messages[1].content, initial[1].content)
    assert.equal(inference.messages.filter(message => message.role === 'system').length, 1)
    assert.match(inference.messages[0].content, /Use the supplied textbook/)
    assert.doesNotMatch(inference.messages[0].content, /facts\.source_|status.*supported/)
  }
})

test('inference cancellation propagates with its trace and does not publish a fallback', async () => {
  let calls = 0
  let trace
  await assert.rejects(runJavascriptInterpret([{ role: 'user', content: question }],
    async () => {
      if (++calls === 1) return call(code)
      throw new DOMException('Cancelled', 'AbortError')
    }, executeIsolatedJavascript, value => { trace = value }), { name: 'AbortError' })
  assert.equal(calls, 2)
  assert.equal(trace.attempts.length, 1)
  assert.equal(trace.narrationFallback, undefined)
})

test('raw transport preserves code quotes, backslashes and line breaks without JSON decoding', async () => {
  const raw = String.raw`const message="first\nsecond";
const path="C:\\notes";
console.log(message);
console.log(path);
console.log('quoted "word"');`
  for (const fenced of [true, false]) {
    const { trace } = await run([call(raw, fenced), 'The printed strings are shown above.'])
    assert.equal(trace.attempts[0].code, raw)
    assert.equal(trace.attempts[0].execution.result, 'first\nsecond\nC:\\notes\nquoted "word"')
    assert.equal(trace.retryCount, 0)
  }
  const crlf = await run([call(code).replaceAll('\n', '\r\n'), explanation])
  assert.equal(crlf.trace.attempts[0].execution.ok, true)
})

test('an incomplete correction after successful execution is repaired instead of shown as an answer', async () => {
  const { answer, trace } = await run([call('console.log(15)'), '<javascript_interpret>\nconst v=[];', call(code), explanation])
  assert.equal(answer, explanation)
  assert.equal(trace.inferenceCount, 4)
  assert.equal(trace.retryCount, 2)
  assert.equal(trace.attempts[1].phase, 'plan')
})

test('execution diagnostics show actual syntax/runtime locations, remain bounded', async () => {
  const referenceCode = 'const value=1;\nvalue+missing;'
  const detailed = await executeIsolatedJavascript(referenceCode)
  assert.match(detailed.error, /ReferenceError at line 2, column \d+/)
  assert.match(detailed.error, /2: value\+missing;/)
  const syntax = await executeIsolatedJavascript("console.log('first\nsecond');")
  assert.match(syntax.error, /SyntaxError at line 1, column \d+/)
  assert.match(syntax.error, /1: console.log\('first/)
  const huge = await executeIsolatedJavascript('throw new Error("x".repeat(20000));')
  assert.equal(huge.ok, false)
  assert.ok(Buffer.byteLength(huge.error) <= javascriptLimits.outputBytes)
  assert.match(huge.error, /Error at line 1/)
  const output = await executeIsolatedJavascript('console.log("x".repeat(9000))')
  assert.equal(output.ok, false)
  assert.match(output.error, /Output exceeds/)
})
