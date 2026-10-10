import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { runRemoteStudyEngine } = requireTranspiledTs('src/main/engine/remote-chat-service.ts')
const sources = Array.from({length:8}, (_,i) => ({ title:'Syllabus', locator:`Page ${i+1}`,
  context: `Course passage ${i+1}. ` + 'Course material explains the topic. '.repeat(48) + (i===7 ? ' Midterm I is September 23 and covers Part I.' : ''), excerpt:'Course material.' }))
const local = name => ({ prompt:'When are the exams?', messages:[], materials:[], settings:{}, retrievedSources:sources,
  model:{engine:'ollama',ollamaModelName:name,contextLength:32768},
  modelSettings:{contextLengthMode:'auto',contextLength:2048,maxLength:4096,reasoningMode:'off'},
  applicationSettings:{suggestionMode:'off'} })
const response = data => ({ok:true,json:async()=>data})

test('Auto discovers a larger real limit and keeps the eighth exam passage in a 16K request', async () => {
  const saved = globalThis.fetch
  let chatCalls=0
  globalThis.fetch = async (url, init) => {
    if (url.endsWith('/api/show')) return response({model_info:{'gemma4.context_length':262144}})
    assert.ok(url.endsWith('/api/chat'))
    const body=JSON.parse(init.body)
    chatCalls++
    assert.equal(body.options.num_ctx,16384)
    assert.equal(body.think,false)
    assert.match(body.messages.at(-1).content,/Midterm I is September 23/)
    if (body.options.num_predict===1) return response({prompt_eval_count:5000,done_reason:'length'})
    assert.equal(body.options.num_predict,4096)
    return response({done_reason:'stop',message:{content:'Midterm I is September 23, covering Part I.'}})
  }
  try {
    const result=await runOllamaStudyEngine(local('auto-context-regression'))
    assert.equal(result.sources.length,8)
    assert.equal(chatCalls,2)
  } finally {globalThis.fetch=saved}
})

test('Auto grows the local window before dropping evidence if measured prompt size exceeds the estimate', async () => {
  const saved=globalThis.fetch, contexts=[]
  globalThis.fetch = async (url, init) => {
    if (url.endsWith('/api/show')) return response({model_info:{'test.context_length':65536}})
    const body=JSON.parse(init.body)
    contexts.push(body.options.num_ctx)
    assert.match(body.messages.at(-1).content,/Midterm I is September 23/)
    return body.options.num_predict===1 ? response({prompt_eval_count:14000}) : response({done_reason:'stop',message:{content:'The answer.'}})
  }
  try {
    const result=await runOllamaStudyEngine(local('auto-context-grow'))
    assert.deepEqual(contexts,[16384,32768,32768])
    assert.equal(result.sources.length,8)
  } finally {globalThis.fetch=saved}
})

test('Custom local context stays fixed and does not perform auto metadata discovery', async () => {
  const saved=globalThis.fetch
  let show=0
  globalThis.fetch=async(url,init)=>{
    if(url.endsWith('/api/show')){show++;return response({capabilities:[]})}
    const body=JSON.parse(init.body)
    assert.equal(body.options.num_ctx,8192)
    return body.options.num_predict===1 ? response({prompt_eval_count:1500}) : response({done_reason:'stop',message:{content:'The answer.'}})
  }
  try {
    const request=local('custom-context')
    request.modelSettings={...request.modelSettings,contextLengthMode:'manual',contextLength:8192}
    await runOllamaStudyEngine(request)
    assert.equal(show,1) // Existing reasoning-capability check only.
  } finally {globalThis.fetch=saved}
})

test('cloud Auto uses discovered source capacity and caps output without sending an Ollama context option', async () => {
  const saved=globalThis.fetch
  let metadataCalls=0, answerCalls=0
  globalThis.fetch=async(url,init)=>{
    if(url.endsWith('/models/cloud-context')){metadataCalls++;return response({context_window:131072,max_completion_tokens:1024})}
    assert.equal(url,'https://context.test/v1/chat/completions')
    answerCalls++
    const body=JSON.parse(init.body)
    assert.equal(body.max_tokens,1024)
    assert.equal(body.num_ctx,undefined)
    assert.match(body.messages.at(-1).content,/Midterm I is September 23/)
    return response({choices:[{message:{content:'The exam answer.'}}]})
  }
  try {
    const request={...local('unused'),model:{engine:'remote',baseUrl:'https://context.test/v1',remoteModelName:'cloud-context',apiKey:'test'},
      retrievedSources:sources.map(source=>({...source,context:source.context.repeat(6)}))}
    await runRemoteStudyEngine(request)
    await runRemoteStudyEngine(request)
    assert.equal(metadataCalls,1)
    assert.equal(answerCalls,2)
  } finally {globalThis.fetch=saved}
})
