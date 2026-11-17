import {test} from 'node:test'
import assert from 'node:assert/strict'
import {evaluateAnswer} from './local-eval-judge.mjs'
const input={question:'How many cells does the sensor use?',idealAnswer:'The sensor uses two cells.',keyEvidenceQuote:'The sensor uses two cells. The case is blue.',generatedAnswer:'It uses two cells.'}
const plan={requirements:[{requirement:'Number of cells',referenceQuote:'two cells'}]}
const covered={id:'r1',status:'covered',answerIds:['A1'],reason:'The requested number is given.'}
async function run(data,outputs){outputs=outputs.flatMap(o => Object.hasOwn(o,'extraErrors') ? [{checks:o.checks},{extraErrors:o.extraErrors}] : [o]);let i=0;const result=await evaluateAnswer(data,async()=>{assert.ok(i<outputs.length);return outputs[i++]});assert.equal(i,outputs.length);return result}
test('equivalent answers pass without optional evidence details',async()=>{
 const r=await run(input,[plan,{checks:[covered],extraErrors:[]}]);assert.equal(r.verdict,'correct');assert.equal(r.score,1)
})
test('a false missing-fact allegation is independently corrected',async()=>{
 const r=await run(input,[plan,{checks:[{id:'r1',status:'missing',answerIds:[],reason:'Number omitted.'}],extraErrors:[]},{checks:[covered],extraReviews:[]}]);assert.equal(r.verdict,'correct');assert.equal(r.trace.length,4)
})
test('an actual opposite assertion fails after grounded verification',async()=>{
 const wrong={...input,generatedAnswer:'It uses six cells.'};const check={id:'r1',status:'contradicted',answerIds:['A1'],reason:'Six contradicts the required two.'}
 const r=await run(wrong,[plan,{checks:[check],extraErrors:[]},{checks:[check],extraReviews:[]}]);assert.equal(r.verdict,'incorrect');assert.equal(r.score,0)
})
test('a missing required answer is distinct from a contradiction',async()=>{
 const check={id:'r1',status:'missing',answerIds:[],reason:'The required number is absent.'}
 const r=await run({...input,generatedAnswer:'The case is blue.'},[plan,{checks:[check],extraErrors:[]},{checks:[check],extraReviews:[]}]);assert.equal(r.verdict,'incorrect');assert.equal(r.contradictions.length,0);assert.equal(r.missing_required.length,1)
})
test('an invented evidence ID cannot establish a failure',async()=>{
 const r=await run(input,[plan,{checks:[covered],extraErrors:[{answerIds:['missing'],sourceIds:['R1'],reason:'Wrong count.'}]},{checks:[],extraReviews:[{id:'e1',decision:'error',answerIds:['missing'],sourceIds:['R1'],reason:'Wrong count.'}]}]);assert.equal(r.verdict,'manual_review')
})
test('a supported extra factual error is not excused by a correct core answer',async()=>{
 const extra={answerIds:['A1'],sourceIds:['E1'],reason:'The color claim contradicts the evidence.'}
 const r=await run({...input,generatedAnswer:'It uses two cells. The case is red.'},[plan,{checks:[covered],extraErrors:[extra]},{checks:[],extraReviews:[{id:'e1',decision:'error',...extra}]}]);assert.equal(r.verdict,'incorrect');assert.equal(r.score,1)
})
test('malformed or unsupported requirements cannot yield a pass',async()=>{
 assert.equal((await run(input,[{requirements:[{requirement:'Invented',referenceQuote:'nine cells'}]}])).verdict,'manual_review')
 assert.equal((await run(input,[{requirements:[]}])).verdict,'manual_review')
})

test('a secondary reference detail may be explicitly ruled unnecessary',async()=>{
 const r=await run(input,[plan,{checks:[{id:'r1',status:'missing',answerIds:[],reason:'Missing.'}],extraErrors:[]},{checks:[{id:'r1',status:'not_required',answerIds:[],reason:'Not asked.'}],extraReviews:[]}]);assert.equal(r.verdict,'manual_review') // No essential requirements remain: never silently pass.
})

test('excluding optional background preserves coverage of the actual question',async()=>{
 const optional={id:'r2',status:'not_required',answerIds:[],reason:'The question asks for cell count, not color.'}
 const r=await run({...input,idealAnswer:'The sensor uses two cells. The case is blue.'},[
  {requirements:[...plan.requirements,{requirement:'Case color',referenceQuote:'The case is blue.'}]},
  {checks:[covered,optional],extraErrors:[]},
  {checks:[optional],extraReviews:[]}
 ]);assert.equal(r.verdict,'correct');assert.equal(r.score,1)
})

test('authored evidence-backed requirements avoid guessing optional reference scope',async()=>{
 const r=await run({...input,requiredFacts:plan.requirements},[{checks:[covered],extraErrors:[]}]);assert.equal(r.verdict,'correct');assert.equal(r.trace[0].stage,'authored_requirements')
 const bad=await run({...input,requiredFacts:[{requirement:'Wrong count',referenceQuote:'nine cells'}]},[]);assert.equal(bad.verdict,'manual_review')
})

test('the judge cannot silently waive an authored requirement',async()=>{
 const waived={id:'r1',status:'not_required',answerIds:[],reason:'Optional.'}
 const r=await run({...input,requiredFacts:plan.requirements},[{checks:[waived],extraErrors:[]},{checks:[waived],extraReviews:[]}]);assert.equal(r.verdict,'manual_review')
})
