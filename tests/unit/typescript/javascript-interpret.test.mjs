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
  assert.equal(exhausted.trace.unchangedCorrections, 2)
  assert.equal(exhausted.trace.attempts.filter(attempt => !attempt.reusedExecution).length, 1)
  assert.match(exhausted.answer, /not a complete answer/)
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
  assert.match(answer, /Calculation results/)
  assert.match(answer, /mean: 5/)
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

test('complete JavaScript fences recover missing tags without trusting fabricated feedback', async () => {
  const response = 'I will compute the product.\n```javascript\nconst factors=[7,9];\n```\nThen multiply them.\n```js\nconsole.log("Product:", factors[0]*factors[1]);\n```\n</javascript_interpret>\nTool feedback: javascript_interpret result:\n999\nAssistant: The result is 999.'
  const { answer, seen, trace } = await run([response, 'Multiplying 7 by 9 gives 63.'])
  assert.equal(answer, 'Multiplying 7 by 9 gives 63.')
  assert.equal(trace.inferenceCount, 2)
  assert.equal(trace.retryCount, 0)
  assert.equal(trace.attempts[0].recoveredTransport, true)
  assert.equal(trace.attempts[0].execution.result, 'Product: 63')
  assert.doesNotMatch(seen[1].messages.at(-2).content, /999|Tool feedback|Then multiply/)
  assert.match(seen[1].messages.at(-1).content, /Product: 63/)
  const bare = await run(['```javascript\nconsole.log(7*9)\n```', 'The product is 63.'])
  assert.equal(bare.trace.attempts[0].recoveredTransport, true)
})

test('malformed transport strips simulated feedback before bounded repair', async () => {
  const { seen, trace } = await run(['Broken tool call.\nTool feedback: javascript_interpret result:\n999', call(code), explanation])
  assert.doesNotMatch(seen[1].messages.at(-2).content, /999/)
  assert.equal(trace.retryCount, 1)
  assert.equal(trace.inferenceCount, 3)
})

test('missing or conflicting evidence can request information without execution or another inference', async () => {
  for (const text of ['Please provide the measurements and their units.', 'The records disagree. Which record should I use?']) {
    const { answer, trace } = await run([`<needs_information>\n${text}\n</needs_information>`], async () => assert.fail('Must not execute'))
    assert.equal(answer, text)
    assert.equal(trace.inferenceCount, 1)
    assert.equal(trace.retryCount, 0)
    assert.equal(trace.informationRequest, text)
    assert.equal(trace.attempts.length, 0)
  }
  const { trace } = await run([call(code), '<needs_information>The units are unspecified. Which units should I use?</needs_information>'])
  assert.equal(trace.inferenceCount, 2)
  assert.equal(trace.narrationFallback, undefined)
  assert.match(trace.informationRequest, /units/)
})

test('wrong arithmetic is rejected even when all its numbers appear in the evidence', async () => {
  const program = 'const values=[11,4,9]; console.log("Method: subtract the second value from the first."); console.log("11 - 4 =",values[0]-values[1]);'
  const bad = 'Subtract the values: 11 − 4 = 9. The result is 7.'
  const { answer, trace } = await run([call(program), bad])
  assert.equal(trace.inferenceCount, 2)
  assert.equal(trace.narrationFallback, true)
  assert.equal(trace.rejectedFinalAnswer, bad)
  assert.match(trace.narrationIssues.join('\n'), /arithmetic equation/)
  assert.match(answer, /Method: subtract/)
  assert.match(answer, /11 - 4 = 7/)
  assert.doesNotMatch(answer, / = 9|```/)
})

test('changed or omitted results and unsupported units fall back without another model pass', async () => {
  for (const [program, answer] of [
    ['console.log({total:19,mean:9.5})', 'The total is 19.'],
    ['const inputs=[5,6]; console.log({mean:5})', 'The inputs are 5 and 6. The mean is 6.'],
    ['console.log({changes:[3,5,3]})', 'The changes are 3, 3, 5.'],
    ['console.log("Score:",7)', 'The score is $7.'],
    ['console.log("Change: 12.50%")', 'The change is 12.50.']
  ]) {
    const { trace, answer: fallback } = await run([call(program), answer])
    assert.equal(trace.inferenceCount, 2)
    assert.equal(trace.narrationFallback, true)
    assert.ok(trace.narrationIssues.length)
    assert.match(fallback, /Calculation results/)
  }
})

test('consistent prose, tables, KaTeX and checked intermediate arithmetic do not require a template', async () => {
  const program = 'const a=11,b=4;console.log("Difference:",a-b);'
  for (const answer of [
    'Subtract 4 from 11 to obtain **7**.',
    '| Operation | Result |\n| --- | --- |\n| 11 − 4 | 7 |',
    String.raw`\[11 - 4 = 7\]`
  ]) {
    const { answer: accepted, trace } = await run([call(program), answer])
    assert.equal(accepted, answer)
    assert.equal(trace.narrationFallback, undefined)
    assert.equal(trace.inferenceCount, 2)
  }
  const fraction = String.raw`\[\frac{14}{2}=7\]`
  const checked = await run([call('console.log(14/2)'), fraction])
  assert.equal(checked.answer, fraction)
  assert.equal(checked.trace.narrationFallback, undefined)
  const intermediate = 'Add 7 + 9 = 16, then divide: 16 / 2 = 8.'
  assert.equal((await run([call('console.log((7+9)/2)'), intermediate])).answer, intermediate)
  const rounded = await run([call('console.log("Change: 12.50%")'), 'The percentage change is 12.5%.'])
  assert.equal(rounded.answer, 'The percentage change is 12.50%.')
  assert.equal(rounded.trace.narrationFallback, undefined)
  const percentage = '(11 - 10) / 10 * 100 = 10.00%. The percentage change is 10.00%.'
  assert.equal((await run([call('console.log(((11-10)/10*100).toFixed(2)+"%")'), percentage])).answer, percentage)
  const fractionOfWhole = String.raw`\(0.125 = 12.50\%\)`
  const escapedPercent = await run([call('console.log("12.50%")'), fractionOfWhole])
  assert.equal(escapedPercent.answer,fractionOfWhole)
  assert.equal(escapedPercent.trace.narrationFallback,undefined)
})

test('narration validation is isolated, bounded, and cancellation still propagates', async () => {
  const seen = []
  const { trace } = await run([call('console.log(9*7)'), '9 × 7 = 63.'], async program => {
    seen.push(program)
    return executeIsolatedJavascript(program)
  })
  assert.equal(seen.length, 2)
  assert.equal(trace.attempts.length, 1)
  assert.equal(trace.inferenceCount, 2)
  assert.ok(trace.validationLatencyMs >= 0)
  let calls=0
  await assert.rejects(run([call('console.log(9*7)'), '9 × 7 = 63.'], async program => {
    if (++calls === 2) throw new DOMException('Cancelled', 'AbortError')
    return executeIsolatedJavascript(program)
  }), {name:'AbortError'})
})

test('valid substitution operands in a checked equality chain survive narration validation',async()=>{
  const program='const value=74;console.log("Converted:",(5/9*(value-32)).toFixed(2));'
  const explanation=String.raw`Use \(C=\frac{5}{9}(F-32)\). For 74°F: \[C=\frac{5}{9}\times(74-32)=\frac{5}{9}\times42=23.33°C\] The converted value is 23.33°C.`
  const {answer,trace}=await run([call(program),explanation],executeIsolatedJavascript,[{role:'user',content:'Convert 74 degrees Fahrenheit and show the substitution.'}])
  assert.equal(answer,explanation)
  assert.equal(trace.narrationFallback,undefined)
  assert.equal(trace.inferenceCount,2)
  const bad=explanation.replace('times42','times45')
  const rejected=await run([call(program),bad])
  assert.equal(rejected.trace.narrationFallback,true)
  assert.match(rejected.trace.narrationIssues.join('\n'),/arithmetic equation/)
})

test('existing narration call can decline an inappropriate population after successful arithmetic',async()=>{
  const question='Two revisions of the same measurement disagree. Neither revision is authoritative. Which summary can be calculated?'
  const {answer,trace,seen}=await run([
    call('console.log("Pooled result:",(7+15+7+39)/4)'),
    '<needs_information>Which revision should I use? These are alternative versions of the same measurement.</needs_information>'
  ],executeIsolatedJavascript,[{role:'user',content:question}])
  assert.match(answer,/Which revision/)
  assert.doesNotMatch(answer,/17/)
  assert.equal(trace.inferenceCount,2)
  assert.equal(trace.retryCount,0)
  assert.equal(trace.attempts.length,1)
  assert.match(seen[1].messages.at(-1).content,/incompatible versions/)
  assert.match(seen[1].messages[0].content,/independent observations/)
})

test('a heading followed by numbered teaching steps is not a scalar result assertion',async()=>{
  const explanation='To calculate the average:\n1. Sum the inputs: 7 + 15 + 7 + 39 = 68.\n2. Divide by the count: 68 / 4 = 17.\nThe average is 17.'
  const result=await run([call('console.log("Average:",(7+15+7+39)/4)'),explanation])
  assert.equal(result.answer,explanation)
  assert.equal(result.trace.narrationFallback,undefined)
  assert.equal(result.trace.inferenceCount,2)
})

test('population correction reuses the existing tool loop and retains repeated observations',async()=>{
  const result=await run([
    call('console.log("Average:",(7+15+39)/3)'),
    call('const observations=[7,15,7,39];const total=observations.reduce((a,b)=>a+b,0);console.log("Observations:",observations);console.log("Total:",total);console.log("Average:",total/observations.length)'),
    'The observations are 7, 15, 7, 39. Adding them gives 68; dividing by 4 gives an average of 17.'
  ])
  assert.equal(result.trace.retryCount,1)
  assert.equal(result.trace.inferenceCount,3)
  assert.equal(result.trace.narrationFallback,undefined)
  assert.match(result.seen[1].messages.at(-1).content,/including repeated values/)
  assert.match(result.answer,/average of 17/)
})

test('unchanged successful corrections reuse output and explicitly request progress', async () => {
  const programs = []
  const { trace, seen, answer } = await run([call(code), call(code), explanation], async program => {
    programs.push(program)
    return executeIsolatedJavascript(program)
  })
  assert.equal(programs.filter(program => program === code).length, 1)
  assert.equal(trace.inferenceCount, 3)
  assert.equal(trace.unchangedCorrections, 1)
  assert.match(seen[2].messages.at(-1).content, /unchanged.*not been executed again/s)
  assert.equal(answer, explanation)
  assert.doesNotMatch(seen[1].messages.at(-1).content, /Current student question/)
})

test('equivalent coefficients and prose-wrapped equations retain valid teaching narration', async () => {
  const program = 'const x=77;console.log("Substitution 1: C =",(5/9),"* (77 - 32)");console.log("Converted:",5/9*(x-32))'
  const teaching = String.raw`Use \(C=\frac{5}{9}(F-32)\). Substituting 77 gives \(\frac{5}{9}\times(77-32)=25\). The result is 25°C.`
  const converted = await run([call(program), teaching], executeIsolatedJavascript,
    [{role:'user', content:'Convert 77°F to Celsius and show the substitution.'}])
  assert.equal(converted.answer, teaching)
  assert.equal(converted.trace.narrationFallback, undefined)
  const wrapped = String.raw`The two scenario means are 15 (\((10+20)/2=30/2=15\)) and 25 (\((10+40)/2=50/2=25\)). Neither is authoritative.`
  const scenarios = await run([call('console.log("Scenario A: 15");console.log("Scenario B: 25")'), wrapped], executeIsolatedJavascript,
    [{role:'user',content:'Two alternative scenarios contain 10,20 or 10,40. Show both means without choosing an authoritative scenario.'}])
  assert.equal(scenarios.answer, wrapped)
  assert.equal(scenarios.trace.narrationFallback, undefined)
  const wrong = await run([call('console.log("Scenario A: 15")'), wrapped.replace('30/2=15','30/2=25')])
  assert.equal(wrong.trace.narrationFallback, true)
})

test('requested decimal precision formats displayed percentages without changing formula multipliers', async () => {
  const teaching = '(14 - 10) / 10 × 100% = 40%. The change is 40%.'
  const {answer, trace} = await run([call('console.log("Change: 40%")'), teaching], executeIsolatedJavascript,
    [{role:'user',content:'Change from 10 to 14: report the percentage to two decimal places.'}])
  assert.equal(answer, '(14 - 10) / 10 × 100% = 40.00%. The change is 40.00%.')
  assert.equal(trace.narrationFallback, undefined)
})

const teachingProgram = `const inputs=[{score:9,weight:2},{score:4,weight:5}];
const contributions=inputs.map(input=>input.score*input.weight);
const total=contributions.reduce((sum,value)=>sum+value,0);
const weight=inputs.reduce((sum,input)=>sum+input.weight,0);
const steps=inputs.map((input,i)=>'Group '+(i+1)+': '+input.score+' × '+input.weight+' = '+contributions[i]);
steps.push(contributions.join(' + ')+' = '+total);
steps.push(inputs.map(input=>input.weight).join(' + ')+' = '+weight);
steps.push(total+' ÷ '+weight+' = '+(total/weight).toFixed(2));
console.log({method:'Multiply each score by its weight, add the contributions, then divide by total weight.',inputs,steps,results:{contributions,average:(total/weight).toFixed(2)+' points'}});`
const teachingQuestion = [{role:'user',content:'Scores of 9 and 4 points have weights 2 and 5. List the contributions and weighted average, showing the steps.'}]

test('execution supplies teaching data to the same final call and retains varied valid narration', async () => {
  const explanation = 'Weight each score: Group 1: 9 × 2 = 18; Group 2: 4 × 5 = 20. Add the contributions: 18 + 20 = 38. Total weight is 2 + 5 = 7. Divide 38 ÷ 7 = 5.43 points. The contributions are 18, 20 and the average is 5.43 points.'
  const {answer,trace,seen}=await run([call(teachingProgram),explanation],executeIsolatedJavascript,teachingQuestion)
  assert.equal(answer,explanation)
  assert.equal(trace.inferenceCount,2)
  assert.equal(trace.retryCount,0)
  assert.equal(trace.teachingOutput,true)
  assert.equal(trace.teachingIssues,undefined)
  assert.equal(trace.narrationFallback,undefined)
  assert.match(seen[1].messages.at(-1).content,/\*\*Method:\*\*/)
  assert.match(seen[1].messages.at(-1).content,/Group 1: 9 × 2 = 18/)
  assert.match(seen[1].messages.at(-1).content,/explain their purpose.*without doing new arithmetic/)
})

test('bad or empty narration falls back to executed method and substitutions without another inference',async()=>{
  for(const narration of ['', 'The contributions are 18 and 20, but the average is 9 points.']) {
    const {answer,trace}=await run([call(teachingProgram),narration],executeIsolatedJavascript,teachingQuestion)
    assert.equal(trace.inferenceCount,2)
    assert.equal(trace.narrationFallback,true)
    assert.match(answer,/\*\*Method:\*\* Multiply each score/)
    assert.match(answer,/\*\*Worked steps:\*\*/)
    assert.match(answer,/Group 1: 9 × 2 = 18/)
    assert.match(answer,/Group 2: 4 × 5 = 20/)
    assert.match(answer,/38 ÷ 7 = 5.43/)
    assert.match(answer,/average: 5.43 points/)
    assert.doesNotMatch(answer,/console.log|```js|average is 9/)
  }
})

test('inconsistent printed teaching steps are withheld from feedback and fallback',async()=>{
  const program=teachingProgram.replace('console.log({method:', 'steps.length=0;steps.push("99 × 2 = 18");console.log({method:')
  const {answer,trace,seen}=await run([call(program),''],executeIsolatedJavascript,teachingQuestion)
  assert.equal(trace.inferenceCount,2)
  assert.ok(trace.teachingIssues.some(issue=>issue.includes('arithmetic equation')))
  assert.doesNotMatch(seen[1].messages.at(-1).content,/Group 1: 99 × 2 = 18/)
  assert.doesNotMatch(answer,/99|Worked steps/)
  assert.match(answer,/average: 5.43 points/)
})

test('one wrong printed equation does not discard the other correct instructional steps',async()=>{
  const program=teachingProgram.replace('console.log({method:', 'steps.push("38 ÷ 2 = 5.43");console.log({method:')
  const {answer,trace,seen}=await run([call(program),''],executeIsolatedJavascript,teachingQuestion)
  assert.equal(trace.inferenceCount,2)
  assert.equal(trace.omittedTeachingSteps,1)
  assert.ok(trace.teachingIssues.some(issue=>issue.includes('arithmetic equation')))
  assert.match(seen[1].messages.at(-1).content,/38 ÷ 7 = 5.43/)
  assert.doesNotMatch(seen[1].messages.at(-1).content,/38 ÷ 2 = 5.43/)
  assert.match(answer,/\*\*Method:\*\*/)
  assert.match(answer,/Group 1: 9 × 2 = 18/)
  assert.match(answer,/38 ÷ 7 = 5.43/)
  assert.doesNotMatch(answer,/38 ÷ 2 = 5.43/)
})

test('numbered output labels do not become operands, and wrong label-to-result substitutions still fail',async()=>{
  const program='console.log("Transition 1 to 2: 6");console.log("Transition 2 to 3: -3")'
  const correct='Transition 1 to 2 = 18 - 12 = 6\nTransition 2 to 3 = 15 - 18 = -3'
  const {answer,trace}=await run([call(program),correct])
  assert.equal(answer,correct)
  assert.equal(trace.narrationFallback,undefined)
  const wrong='Transition 1 to 2 = 15 - 18 = -3\nTransition 2 to 3 = 18 - 12 = 6'
  const rejected=await run([call(program),wrong])
  assert.equal(rejected.trace.narrationFallback,true)
  assert.match(rejected.trace.narrationIssues.join(' '),/named result/)
  const actualWrongEquation=await run([call('console.log(9)'), 'Therefore 9 = 5 + 3 = 8.'])
  assert.equal(actualWrongEquation.trace.narrationFallback,true)
  assert.match(actualWrongEquation.trace.narrationIssues.join(' '),/arithmetic equation/)
})

test('fallback displays numeric string arrays without JSON quotes or loss of precision',async()=>{
  const {answer}=await run([call('console.log("Values:",["3.00","-2.00","5.00"])'),''])
  assert.match(answer,/Values: 3.00, -2.00, 5.00/)
  assert.doesNotMatch(answer,/\[|"/)
})

test('a generic result field and explanatory prose after equations do not discard correct narration',async()=>{
  const program='const x=8,y=3;console.log({method:"Multiply the supplied values.",inputs:[x,y],steps:["8 × 3 = 24"],result:{product:x*y}})'
  const teaching='8 × 3 = 24 (This is the product of the supplied values.) The product is 24.'
  const {answer,trace}=await run([call(program),teaching],executeIsolatedJavascript,[{role:'user',content:'Multiply 8 and 3, explaining the substitution.'}])
  assert.equal(answer,teaching)
  assert.equal(trace.teachingOutput,true)
  assert.equal(trace.narrationFallback,undefined)
})

test('fallback retains independently checked substitutions when another narration equation is wrong',async()=>{
  const program='const x=8,y=3;console.log({method:"Multiply the values, then halve the product.",inputs:[x,y],steps:[],results:{product:x*y,half:x*y/2}})'
  const narration='8 × 3 = 24. Then halve: 24 / 2 = 9. The product is 24 and half is 9.'
  const {answer,trace}=await run([call(program),narration],executeIsolatedJavascript,[{role:'user',content:'Multiply 8 by 3 then halve the product. Explain the steps.'}])
  assert.equal(trace.inferenceCount,2)
  assert.equal(trace.narrationFallback,true)
  assert.match(answer,/8 \* 3 = 24/)
  assert.doesNotMatch(answer,/24 \/ 2 = 9/)
  assert.match(answer,/half: 12/)
  // True arithmetic on invented inputs is not instructional evidence.
  const arbitrary=await run([call(program),'99 + 1 = 100. The half is 9.'],executeIsolatedJavascript,[{role:'user',content:'Multiply 8 by 3 then halve the product.'}])
  assert.doesNotMatch(arbitrary.answer,/99 \+ 1 = 100/)
})

test('unresolved printed operands are withheld even when no numeric equality can be parsed',async()=>{
  const program='console.log({method:"Subtract the supplied values.",inputs:[8,3],steps:["8 - undefined = 5"],results:{difference:5}})'
  const {answer,trace,seen}=await run([call(program),''],executeIsolatedJavascript,[{role:'user',content:'Subtract 3 from 8.'}])
  assert.equal(trace.omittedTeachingSteps,1)
  assert.match(trace.teachingIssues.join(' '),/unresolved operand/)
  assert.doesNotMatch(answer,/undefined/)
  assert.doesNotMatch(seen[1].messages.at(-1).content,/8 - undefined/)
  assert.match(answer,/\*\*Method:\*\* Subtract/)
  assert.match(answer,/difference: 5/)
})
