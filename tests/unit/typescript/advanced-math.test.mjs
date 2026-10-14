import assert from 'node:assert/strict'
import test from 'node:test'
import Module from 'node:module'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { advancedMathBucket, advancedMathModelName, advancedMathModelLabel,
  advancedMathModelInstalled, reconcileAdvancedMathEnabled, normalizeAdvancedMathEnabled } = requireTranspiledTs('src/shared/advanced-math.ts')
const { createAdvancedMathService } = requireTranspiledTs('src/main/engine/advanced-math-service.ts')
const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
const { parseQuestionRewrite, questionResolutionSchema, questionRewriteMessages } = requireTranspiledTs('src/main/engine/question-rewrite.ts')
const defaultModel = { id:'ollama:default',name:'Default chat',engine:'ollama',ollamaModelName:'default',contextLength:8192,status:'ready',addedAt:'' }
const defaults = {contextLength:8192,maxLength:1024,systemMessage:'Default persona',temperature:0.7,topP:0.4,topK:40,minP:0,repeatPenalty:1.18}
const request = { prompt:'Compute 9 times 4, then subtract 2.',messages:[],materials:[],model:defaultModel,
  modelSettings:defaults,settings:{maxSources:4,modelSettingsById:{},application:{advancedMathEnabled:true}},
  applicationSettings:{advancedMathEnabled:true,suggestionMode:'off',followUpSuggestionCount:0},advancedMathEnabled:true }
const installed = {running:true,baseUrl:'http://127.0.0.1:11434',models:[{name:advancedMathModelName,details:{contextLength:8192}}]}
const standalone = {mode:'standalone',query:request.prompt,clarification:'',reasoning:false}
function service(overrides={}) {
  let resolves=0, inventories=0
  const router=createAdvancedMathService({resolve:async input=>{resolves++;return {...standalone,query:input.prompt,answerRoute:'chat',routeReason:'Conceptual question.'}},
    status:async()=>{inventories++;return installed},...overrides})
  return {router,counts:()=>({resolves,inventories})}
}

test('three intent buckets distinguish computation from definitions, lookups and single operations',()=>{
  for(const question of ['hi','What is normalization?','Why are indexes useful?','How does a range scan work?','What is 9 + 4?','Explain how to calculate a weighted average.',
    'According to the passage, which editions followed 2010, and how many are there?','What is the percentage-change formula?']) {
    assert.equal(advancedMathBucket(question),'chat',question)
  }
  for(const question of ['Compute 9 times 4, then subtract 2.','Calculate the weighted average of groups of 12 and 18.',
    'For values 4, 8 and 16, list consecutive differences and their mean.','What is 9 + 4 * 2?']) {
    assert.equal(advancedMathBucket(question),'math',question)
  }
  for(const question of ['Compare the outcomes.','What about their average?','Find the eigenvalues of this matrix.',
    'Assess plans A at 9 credits and B at 4 credits when each has a fee of 2 credits.']) assert.equal(advancedMathBucket(question),'ambiguous',question)
})

test('teaching instructions do not override applied computation intent',async()=>{
  for(const question of [
    'For counts 50, 65, 52, list consecutive percentage changes and their mean. Explain the formula and substitutions.',
    'For values 7, 13 and 10, list the changes, total and average. Explain each step.',
    'Explain each substitution when calculating the average of 17, 21 and 25.',
    'Explain the method, then calculate the weighted average using the table.'
  ]) {
    assert.equal(advancedMathBucket(question),'math',question)
    const s=service()
    const result=await s.router.prepare({...request,prompt:question})
    assert.equal(result.advancedMathPlan.route,'math')
    assert.deepEqual(s.counts(),{resolves:0,inventories:1})
  }
  for(const question of ['Explain how to calculate a percentage change.',
    'Describe why a weighted mean differs from an ordinary mean.',
    'What is the mean formula? Explain its meaning.']) assert.equal(advancedMathBucket(question),'chat',question)
})

test('new setting defaults off, persists true, and does not migrate legacy JavaScript opt-in',()=>{
  for(const value of [undefined,null,'true',1,{},false]) assert.equal(normalizeAdvancedMathEnabled(value),false)
  assert.equal(normalizeAdvancedMathEnabled(JSON.parse(JSON.stringify({advancedMathEnabled:true})).advancedMathEnabled),true)
  assert.equal(normalizeAdvancedMathEnabled({javascriptInterpretEnabled:true}.advancedMathEnabled),false)
})

test('off and obvious ordinary requests keep the default model without classifier or inventory calls',async()=>{
  const s=service()
  const off={...request,advancedMathEnabled:false}
  assert.equal(await s.router.prepare(off),off)
  const ordinary=await s.router.prepare({...request,prompt:'What is normalization?'})
  assert.equal(ordinary.model,defaultModel)
  assert.equal(ordinary.advancedMathPlan.route,'chat')
  assert.deepEqual(s.counts(),{resolves:0,inventories:0})
})

test('confirmed missing installation clears the saved opt-in; installing never enables it',()=>{
  const missing={...installed,models:[]}
  assert.equal(advancedMathModelInstalled(missing),false)
  assert.equal(reconcileAdvancedMathEnabled(true,missing),false)
  assert.equal(reconcileAdvancedMathEnabled(false,installed),false)
  assert.equal(reconcileAdvancedMathEnabled(true,installed),true)
  // A transient connection failure does not establish a removal. The UI still
  // presents an unchecked, disabled control until the server is available.
  assert.equal(advancedMathModelInstalled(null),false)
  assert.equal(reconcileAdvancedMathEnabled(true,null),true)
  for(const name of ['qwen2.5-coder:14b','qwen2.5-coder:32b','qwen3-coder:30b','qwen2.5-coder:latest','qwen2.5-coder:7b-instruct']) {
    const alternative={...installed,models:[{name}]}
    assert.equal(advancedMathModelInstalled(alternative),false)
    assert.equal(reconcileAdvancedMathEnabled(true,alternative),false)
  }
})

test('Advanced Math only uses Qwen2.5-Coder 7B, ignoring previous selections and supplied model descriptors',async()=>{
  for(const name of ['qwen2.5-coder:14b','qwen2.5-coder:32b','qwen3-coder:30b']) {
    const settings={...request.settings,application:{...request.settings.application,advancedMathModelName:name},
      modelSettingsById:{[`ollama:${name}`]:{contextLength:4096}}}
    const s=service({status:async()=>({...installed,models:[...installed.models,{name}]})})
    const prepared=await s.router.prepare({...request,settings,
      advancedMathPlan:{prompt:request.prompt,route:'math',reason:'Calculation.',model:{...defaultModel,ollamaModelName:name}}})
    assert.equal(prepared.model.ollamaModelName,advancedMathModelName)
    assert.equal(prepared.model.name,advancedMathModelLabel)
    assert.equal(prepared.modelSettings.contextLength,8192)
    assert.equal(prepared.advancedMathPlan.route,'math')
    assert.equal(prepared.model.ollamaBaseUrl,installed.baseUrl)
    const missing=service({status:async()=>({...installed,models:[{name}]})})
    await assert.rejects(missing.router.prepare({...request,settings}),/requires qwen2.5-coder:7b/)
  }
})

test('clear math uses a dedicated local profile, preserving source/context fields and default selection',async()=>{
  const s=service()
  const original={...request,model:{...defaultModel,engine:'remote',apiKey:'secret',baseUrl:'https://cloud.invalid'},retrievedSources:[{title:'Measurements',excerpt:'Data.'}],
    selectedPassage:{text:'Quoted task',role:'user',messageId:'p'},referenceExchange:{question:'Previous?',answer:'Previous explanation.'},conversationContextMode:'contextual'}
  // A selected passage correctly needs reference resolution, so use an existing
  // plan here to isolate per-request model switching from context inference.
  original.advancedMathPlan={prompt:original.prompt,route:'math',reason:'Resolved calculation task.'}
  const prepared=await s.router.prepare(original)
  assert.equal(prepared.model.ollamaModelName,advancedMathModelName)
  assert.equal(prepared.model.apiKey,undefined)
  assert.equal(prepared.model.baseUrl,undefined)
  assert.equal(prepared.modelSettings.systemMessage,'')
  assert.equal(prepared.advancedMathPlan.route,'math')
  assert.equal(prepared.selectedPassage,original.selectedPassage)
  assert.equal(prepared.referenceExchange,original.referenceExchange)
  assert.equal(prepared.retrievedSources,original.retrievedSources)
  assert.equal(original.model.apiKey,'secret')
  assert.equal(original.model.ollamaModelName,'default')
  await s.router.prepare(original)
  assert.deepEqual(s.counts(),{resolves:0,inventories:1})
})

test('practice tasks bypass Advanced Math while ordinary auto-reasoning retains its shared rewrite decision',async()=>{
  let decisions=0
  const s=service({resolve:async input=>{
    decisions++
    assert.equal(input.classifyAdvancedMath,false)
    return {...standalone,query:input.prompt,reasoning:true}
  }})
  const practice={...request,practiceTask:'feedback'}
  assert.equal(await s.router.prepare(practice),practice)
  assert.equal(decisions,0)
  assert.equal(s.counts().inventories,0)
  const ordinary=await s.router.prepare({...request,prompt:'What is normalization?',modelSettings:{...defaults,reasoningMode:'auto'}})
  assert.equal(ordinary.model,defaultModel)
  assert.equal(ordinary.advancedMathPlan.route,'chat')
  assert.equal(ordinary.reasoning,true)
  assert.equal(decisions,1)
  const math=await s.router.prepare({...request,modelSettings:{...defaults,reasoningMode:'auto'}})
  assert.equal(math.advancedMathPlan.route,'math')
  assert.equal(math.modelSettings.reasoningMode,'off')
  assert.equal(math.modelSettings.contextLengthMode,'manual')
  assert.equal(decisions,1)
})

test('local references to supplied numeric data do not trigger a standalone context rewrite',async()=>{
  const s=service()
  const prepared=await s.router.prepare({...request,prompt:'For these values 2, 8 and 14, calculate the average.',
    messages:[{role:'user',text:'Previous topic.'},{role:'assistant',text:'Previous explanation.'}]})
  assert.equal(prepared.advancedMathPlan.route,'math')
  assert.equal(s.counts().resolves,0)
})

test('ambiguous classification and reference resolution share one call, and the plan is reused before answering',async()=>{
  let calls=0
  const s=service({resolve:async input=>{
    calls++
    assert.equal(input.classifyAdvancedMath,true)
    assert.equal(input.messages.length,2)
    return {mode:'contextual',query:'Compare derived amounts for the two measured populations.',clarification:'',answerRoute:'math',routeReason:'Requires deriving several amounts.'}
  }})
  const contextual={...request,prompt:'Compare these outcomes.',messages:[{role:'user',text:'Two measured populations.'},{role:'assistant',text:'Their observations are in the passage.'}]}
  const resolution=await s.router.resolve(contextual)
  assert.equal(resolution.advancedMathPlan.route,'math')
  await s.router.prepare({...contextual,advancedMathPlan:resolution.advancedMathPlan,referenceExchange:{question:'Two measured populations.',answer:'Their observations are in the passage.'}})
  assert.equal(calls,1)
})

test('ambiguous ordinary classification uses one decision and no Qwen lookup',async()=>{
  const s=service()
  const prepared=await s.router.prepare({...request,prompt:'Compare the outcomes.'})
  assert.equal(prepared.model,defaultModel)
  assert.equal(prepared.advancedMathPlan.route,'chat')
  assert.deepEqual(s.counts(),{resolves:1,inventories:0})
})

test('clarification, invalid classifier output, and missing Qwen never fall back to speculative computation',async()=>{
  const clarification=service({resolve:async()=>({mode:'clarify',query:'',clarification:'Which measured population do you mean?'})})
  const result=await clarification.router.prepare({...request,prompt:'Compare the outcomes.'})
  assert.equal(result.mode,'clarify')
  assert.equal(clarification.counts().inventories,0)
  const invalid=service({resolve:async()=>standalone})
  await assert.rejects(invalid.router.prepare({...request,prompt:'Compare the outcomes.'}),/determine/)
  const missing=service({status:async()=>({...installed,models:[{name:'qwen2.5-coder:14b'}]})})
  await assert.rejects(missing.router.prepare(request),/requires qwen2.5-coder:7b/)
  const offline=service({status:async()=>({...installed,running:false})})
  await assert.rejects(offline.router.prepare(request),/Ollama running/)
})

test('cancellation after inventory lookup stops before an answer model runs',async()=>{
  const controller=new AbortController()
  const s=service({status:async()=>{controller.abort();return installed}})
  await assert.rejects(s.router.prepare(request,controller.signal),{name:'AbortError'})
})

test('RAG selects the math retrieval budget before searching, preserving the original task and sources',async()=>{
  const s=service()
  let searches=0
  const sources=[{title:'Population table',excerpt:'Measured values.'}]
  const prepared=await prepareRewrittenStudyChat(request,{resolve:input=>s.router.resolve(input),search:async(query,plan)=>{
    searches++
    assert.equal(query,request.prompt)
    assert.equal(plan.model.ollamaModelName,advancedMathModelName)
    assert.equal(plan.modelSettings.contextLength,8192)
    return sources
  }})
  assert.equal(searches,1)
  assert.equal(prepared.request.prompt,request.prompt)
  assert.equal(prepared.request.model,defaultModel)
  assert.equal(prepared.request.retrievedSources,sources)
  assert.equal(prepared.request.advancedMathPlan.route,'math')
  assert.equal(s.counts().resolves,0)
})

test('combined routing schema is opt-in and does not alter ordinary rewrite validation',()=>{
  const classified={mode:'standalone',query:'Paraphrase',clarification:'',reasoning:false,answer_route:'math',route_reason:'Multiple derived results.'}
  const result=parseQuestionRewrite(JSON.stringify(classified),'Original?',false,true)
  assert.equal(result.query,'Original?')
  assert.equal(result.answerRoute,'math')
  assert.throws(()=>parseQuestionRewrite(JSON.stringify(classified),'Original?'))
  assert.throws(()=>parseQuestionRewrite(JSON.stringify({...classified,answer_route:'unknown'}),'Original?',false,true))
  assert.ok(questionResolutionSchema(true).required.includes('answer_route'))
  assert.ok(!questionResolutionSchema(false).required.includes('answer_route'))
  assert.match(questionRewriteMessages({...request,classifyAdvancedMath:true})[0].content,/Decide task intent only/)
})

// Capture Electron's tiny app surface while loading the engine; all HTTP calls
// below are mocked. No Ollama model, cloud service or benchmark is invoked.
const originalLoad=Module._load
let engine
try {
  Module._load=function(name,...args){return name==='electron'?{app:{getPath:()=>'/private/tmp/tokensmith-advanced-math-unit',on:()=>{}}}:originalLoad.call(this,name,...args)}
  engine=requireTranspiledTs('src/main/engine/engine-service.ts')
} finally { Module._load=originalLoad }

test('end-to-end cloud default dispatches math to local Qwen for two calls, with RAG citations and actual-model publication',async()=>{
  const originalFetch=globalThis.fetch
  let generationCalls=0
  const published=[]
  const controller=new AbortController()
  try {
    globalThis.fetch=async(url,options)=>{
      assert.ok(url.startsWith('http://127.0.0.1:11434/'))
      if(url.endsWith('/tags'))return {ok:true,json:async()=>({models:[{name:advancedMathModelName,details:{context_length:8192}}]})}
      assert.ok(options.signal instanceof AbortSignal)
      assert.equal(options.headers.Authorization,undefined)
      const body=JSON.parse(options.body)
      assert.equal(body.model,advancedMathModelName)
      assert.equal(body.format,undefined)
      assert.equal(body.options.temperature,0)
      assert.equal(body.think,false)
      generationCalls++
      const text=body.messages.map(m=>m.content).join('\n')
      assert.match(text,/Nine records per batch; four batches/)
      assert.doesNotMatch(text,/Hidden contradictory value/)
      return {ok:true,json:async()=>({done_reason:'stop',message:{content:generationCalls===1
        ? '<javascript_interpret>\nconsole.log("Gross count:",9*4); console.log("Net count:",9*4-2);\n</javascript_interpret>'
        : 'Multiply 9 records by 4 batches to get 36, then subtract 2: the net count is **34**.'},load_duration:1e6,prompt_eval_duration:2e6,eval_duration:3e6})}
    }
    const result=await engine.sendChatMessage({...request,model:{...defaultModel,engine:'remote',apiKey:'secret',baseUrl:'https://cloud.invalid',remoteModelName:'cloud'},
      retrievedSources:[{title:'Batch measurements',excerpt:'Hidden contradictory value: 99.',context:'Nine records per batch; four batches. Subtract two discarded records.'}]},
      {signal:controller.signal,onAnswer:(answer,followUps)=>published.push({answer,followUps})})
    assert.equal(generationCalls,2)
    assert.equal(result.modelName,'Qwen2.5-Coder 7B')
    assert.equal(result.advancedMathUsed,true)
    assert.equal(result.sources.length,1)
    assert.match(result.text,/34/)
    assert.deepEqual(published,[{answer:result,followUps:false}])
  } finally {globalThis.fetch=originalFetch}
})

test('end-to-end ordinary questions keep the default model and their normal context preflight when Advanced Math is on or off',async()=>{
  const originalFetch=globalThis.fetch
  try {
    for(const enabled of [true,false]){
      let calls=0, probes=0
      globalThis.fetch=async(url,options)=>{
        if(url.endsWith('/show'))return {ok:true,json:async()=>({model_info:{'test.context_length':8192},capabilities:[]})}
        const body=JSON.parse(options.body)
        if(body.options.num_predict===1){probes++;return {ok:true,json:async()=>({prompt_eval_count:100,message:{content:'token'}})}}
        calls++
        assert.equal(body.model,'default')
        assert.equal(body.format,undefined)
        return {ok:true,json:async()=>({message:{content:'Normalization organizes related data.'}})}
      }
      const result=await engine.sendChatMessage({...request,prompt:'What is normalization?',advancedMathEnabled:enabled,retrievedSources:[]})
      assert.equal(calls,1)
      assert.equal(probes,1)
      assert.equal(result.modelName,defaultModel.name)
      assert.equal(result.advancedMathUsed,undefined)
    }
  } finally {globalThis.fetch=originalFetch}
})

test('an ambiguous standalone request adds only one default-model classification before two Qwen calls',async()=>{
  const originalFetch=globalThis.fetch
  const models=[]
  const prompt='Assess plans A at 9 credits and B at 4 credits when each has a fee of 2 credits.'
  try {
    globalThis.fetch=async(_url,options)=>{
      const body=JSON.parse(options.body)
      models.push(body.model)
      if(body.model==='default'){
        assert.ok(body.format.required.includes('answer_route'))
        return {ok:true,json:async()=>({message:{content:JSON.stringify({mode:'standalone',query:prompt,clarification:'',reasoning:false,answer_route:'math',route_reason:'Derive the two costs before comparing them.'})}})}
      }
      assert.equal(body.format,undefined)
      return {ok:true,json:async()=>({message:{content:models.length===2
        ? '<javascript_interpret>\nconsole.log("A:",9+2); console.log("B:",4+2);\n</javascript_interpret>'
        : 'Add the fee: A costs 11 credits and B costs 6 credits.'}})}
    }
    const result=await engine.sendChatMessage({...request,prompt,retrievedSources:[]})
    assert.deepEqual(models,['default',advancedMathModelName,advancedMathModelName])
    assert.equal(result.advancedMathUsed,true)
  } finally {globalThis.fetch=originalFetch}
})

test('contextual RAG rewriting shares classification and does not repeat it at answer dispatch',async()=>{
  const originalFetch=globalThis.fetch
  const models=[]
  const prompt='What about their average?'
  const contextual={...request,prompt,messages:[{role:'user',text:'What were the observations?'},{role:'assistant',text:'The two observations are 9 and 4.'}]}
  try {
    globalThis.fetch=async(_url,options)=>{
      const body=JSON.parse(options.body)
      models.push(body.model)
      if(body.model==='default'){
        assert.equal(JSON.parse(body.messages[1].content).previous_exchange.answer,'The two observations are 9 and 4.')
        return {ok:true,json:async()=>({message:{content:JSON.stringify({mode:'contextual',query:'Average of the two observations in the measurement table.',clarification:'',reasoning:false,answer_route:'math',route_reason:'Compute a mean from the resolved population.'})}})}
      }
      assert.match(body.messages.map(m=>m.content).join('\n'),/Measurement table values: 9 and 4/)
      return {ok:true,json:async()=>({message:{content:models.length===2
        ? '<javascript_interpret>\nconsole.log("Mean:",(9+4)/2);\n</javascript_interpret>'
        : 'Add 9 and 4, then divide by 2 observations. The mean is 6.5.'}})}
    }
    let searches=0
    const prepared=await prepareRewrittenStudyChat(contextual,{resolve:engine.resolveChatQuestion,search:async(query,plan)=>{
      searches++
      assert.equal(plan.route,'math')
      assert.match(query,/measurement table/)
      return [{title:'Measurements',excerpt:'Measurement table values: 9 and 4.'}]
    }})
    const result=await engine.sendChatMessage(prepared.request)
    assert.deepEqual(models,['default',advancedMathModelName,advancedMathModelName])
    assert.equal(searches,1)
    assert.equal(result.sources.length,1)
    assert.equal(prepared.request.prompt,prompt)
    assert.equal(prepared.request.referenceExchange.answer,'The two observations are 9 and 4.')
  } finally {globalThis.fetch=originalFetch}
})

test('Advanced Math publishes a missing-evidence clarification with citations in one Qwen call',async()=>{
  const originalFetch=globalThis.fetch
  let calls=0
  try {
    globalThis.fetch=async(_url,options)=>{
      const body=JSON.parse(options.body)
      assert.equal(body.model,advancedMathModelName)
      assert.match(body.messages.map(m=>m.content).join('\n'),/Measurements have not been recorded/)
      calls++
      return {ok:true,json:async()=>({message:{content:'<needs_information>Please provide the measurements and their units.</needs_information>'}})}
    }
    const result=await engine.sendChatMessage({...request,prompt:'Calculate the mean of the measurements.',
      retrievedSources:[{title:'Measurement record',excerpt:'Measurements have not been recorded.'}]})
    assert.equal(calls,1)
    assert.equal(result.text,'Please provide the measurements and their units.')
    assert.equal(result.sources.length,1)
    assert.equal(result.sources[0].title,'Measurement record')
    assert.equal(result.advancedMathUsed,true)
  } finally {globalThis.fetch=originalFetch}
})
