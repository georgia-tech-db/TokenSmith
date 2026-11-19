import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const {createBookPreparation}=requireTranspiledTs('src/main/collections/book-preparation.ts')
const {loadSavedTopics}=requireTranspiledTs('src/shared/saved-topics.ts')
const model={id:'gemma',engine:'ollama',ollamaModelName:'gemma4:e4b',status:'ready'}
function fixture() {
 let saved=null, calls=0, fail=false, gate=null
 const events=[]
 const material={id:'c/1',status:'ready',collectionRevision:'r1'}
 const docs=[{materialId:'c/1',documentId:1}]
 const bridge={studyDocuments:async()=>docs,topicSources:async()=>[{...docs[0],chunkId:'1',documentTitle:'Book',title:'Book',excerpt:'Chapter 1. Storage\nBuffer pools cache database pages.'}],loadTopicInventory:async()=>saved,saveTopicInventory:async x=>{saved=structuredClone(x)},sendChatMessage:async r=>{
  calls++;if(gate)await gate;if(fail)throw new Error('Model stopped');return {sources:r.retrievedSources,text:JSON.stringify({topics:[{title:'Buffer pools',aliases:[],objectives:['Caching','Eviction'],sourceKeys:r.practiceAllowedIds.slice(0,1)}]})}
 }}
 const deps={state:async()=>({models:[model],settings:{application:{},modelDefaults:{},modelSettingsById:{}}}),index:async(id,path,m,o,progress)=>{events.push('index');progress({materialId:id,phase:'complete',percent:100});return material},topics:()=>bridge,progress:p=>events.push(p),cancelIndex:async()=>events.push('cancel')}
 const service=createBookPreparation(deps)
 return {service,events,bridge,material,docs,deps,get saved(){return saved},get calls(){return calls},set fail(v){fail=v},set gate(v){gate=v}}
}
test('import awaits saved topics before reporting ready; Practice only reads and never invokes a model',async()=>{
 const f=fixture();let release;f.gate=new Promise(r=>release=r)
 const job=f.service.run('c/1','book.md')
 await new Promise(r=>setTimeout(r,10))
 assert.ok(f.events.some(e=>e.phase==='topics'));assert.ok(!f.events.some(e=>e.phase==='complete'))
 release();await job
 assert.equal(f.saved.complete,true);assert.equal(f.events.at(-1).phase,'complete')
 const calls=f.calls
 const topics=await loadSavedTopics(f.bridge,f.docs,[f.material]);await loadSavedTopics(f.bridge,f.docs,[f.material])
 assert.equal(topics[0].title,'Buffer pools');assert.equal(f.calls,calls)
})
for (const mode of ['basic', 'ai']) {
 test(`${mode} import uses the saved material ID for topics and the request ID for progress`, async () => {
  const f=fixture()
  const requestId='c/import-pending'
  const options={preparation:{mode,instructions:'',documentInstructions:{}}}
  const index=f.deps.index
  f.deps.index=async(id,path,embedder,received,progress)=>{
   assert.equal(id,requestId);assert.equal(path,'book.pdf');assert.deepEqual(received.preparation,options.preparation);if(mode==='ai')assert.equal(received.preparationModel.id,model.id)
   return index(id,path,embedder,received,progress)
  }
  const material=await f.service.run(requestId,'book.pdf',undefined,options)
  assert.equal(material.id,f.material.id)
  assert.equal(f.saved.complete,true)
  assert.equal(f.saved.collectionId,'c')
  assert.ok(f.saved.topics.length>0)
  assert.ok(f.saved.topics.every(topic=>topic.sources.every(source=>source.materialId===f.material.id)))
  assert.equal(f.events.at(-1).phase,'complete')
  assert.ok(f.events.filter(event=>event.phase).every(event=>event.materialId===requestId))
 })
}
test('topic failure keeps import incomplete and can be retried from preparation',async()=>{
 const f=fixture();f.fail=true;await assert.rejects(f.service.run('c/1','book.md'),/Model stopped/)
 assert.ok(!f.events.some(e=>e.phase==='complete'));assert.equal(f.events.at(-1).phase,'error')
 assert.equal(f.events.at(-1).resumePhase,'topics')
 f.deps.materials=async()=>[f.material]
 f.fail=false;await f.service.run('c/1','',undefined,undefined,true);assert.equal(f.saved.complete,true)
 assert.equal(f.events.filter(event=>event==='index').length,1,'Retry topics without indexing the document again')
})
test('cancellation during topics never marks a book ready',async()=>{
 const f=fixture();let release;f.gate=new Promise(r=>release=r)
 const job=f.service.run('c/1','book.md');await new Promise(r=>setTimeout(r,10))
 await f.service.cancel('c/1');release();await assert.rejects(job,/abort/i)
 assert.ok(!f.events.some(e=>e.phase==='complete'));assert.equal(f.saved,null)
})
test('missing local model stops import before indexing',async()=>{
 const f=fixture();f.deps.state=async()=>({models:[]})
 await assert.rejects(f.service.run('c/1','book.md'),/preparation model/);assert.ok(!f.events.includes('index'))
})
test('Practice rejects missing, incomplete, old and stale inventories without writes or model calls',async()=>{
 const docs=[{materialId:'c/1',documentId:1}],materials=[{id:'c/1',collectionRevision:'r1'}]
 for(const saved of [null,{version:2,complete:true,indexRevision:'r1'},{version:3,complete:false,indexRevision:'r1'},{version:3,complete:true,indexRevision:'old'}])
  await assert.rejects(loadSavedTopics({loadTopicInventory:async()=>saved},docs,materials),/Extract topics/)
})
test('Practice component has no generation, model, write or resume dependency',()=>{
 const code=readFileSync('src/renderer/src/PracticeTopicPicker.tsx','utf8')
 assert.doesNotMatch(code,/generateTopicInventory|sendChatMessage|saveTopicInventory|Resume topics|modelSettings/)
 assert.match(code,/loadSavedTopics/)
})


test('renderer exposes only saved topic reads; model loading has no topic generation interface',()=>{
  const preload=readFileSync('src/preload/index.ts','utf8')
  assert.doesNotMatch(preload,/topicSources:|saveTopicInventory:|collectionTopics:|saveCollectionTopics:/)
  const client=readFileSync('src/renderer/src/practice-client.ts','utf8')
  assert.doesNotMatch(client,/async topics\(/)
})

for (const engine of ['ollama', 'remote']) test(`topics use the same selected ${engine} model as preparation`, async () => {
 const f=fixture()
 const chosen={id:'chosen',engine,ollamaModelName:engine==='ollama'?'another-model':undefined,status:'ready'}
 const options={preparation:{mode:'ai',modelId:chosen.id,instructions:'',documentInstructions:{}},preparationModel:chosen}
 const index=f.deps.index
 f.deps.index=async(id,path,embedder,received,progress)=>{assert.equal(received.preparationModel,chosen);return index(id,path,embedder,received,progress)}
 const send=f.bridge.sendChatMessage
 f.bridge.sendChatMessage=async request=>{assert.equal(request.model,chosen);return send(request)}
 await f.service.run('c/1','book.pdf',undefined,options)
 assert.equal(f.saved.modelId,chosen.id)
})
test('preparation defaults to E4B but keeps an explicit selection and refuses unavailable selections', () => {
 const {preparationGenerator}=requireTranspiledTs('src/shared/preparation.ts')
 const other={id:'other',engine:'ollama',status:'ready'}
 assert.equal(preparationGenerator([other,model],undefined,'other'),model)
 assert.equal(preparationGenerator([other,model],'other'),other)
 assert.equal(preparationGenerator([other,model],'missing'),undefined)
})

test('saved preparation model is also used for indexing and topics without an explicit model object', async () => {
 const f=fixture(), chosen={id:'saved',engine:'ollama',ollamaModelName:'saved-model',status:'ready'}
 f.deps.state=async()=>({models:[model,chosen],settings:{application:{defaultModelId:model.id},modelDefaults:{},modelSettingsById:{}}})
 const index=f.deps.index
 f.deps.index=async(id,path,embedder,options,progress)=>{assert.equal(options.preparationModel.id,chosen.id);return index(id,path,embedder,options,progress)}
 await f.service.run('c/1','book.pdf',undefined,{preparation:{mode:'ai',modelId:chosen.id}})
 assert.equal(f.saved.modelId,chosen.id)
})

test('page preparation progress is weighted overall progress and reserves completion for topics',async()=>{
 const f=fixture()
 f.deps.index=async(id,path,model,options,progress)=>{
  progress({materialId:id,phase:'chunking',percent:25,message:'Reading page 6 of 10'})
  progress({materialId:id,phase:'complete',percent:100})
  return f.material
 }
 await f.service.run('c/1','book.pdf')
 const page=f.events.find(e=>e.phase==='chunking')
 assert.equal(page.percent,18.75)
 assert.equal(Math.round(page.percent),19)
 assert.ok(f.events.some(e=>e.phase==='topics'&&e.percent===75))
 assert.equal(f.events.at(-1).phase,'complete')
 assert.equal(f.events.at(-1).percent,100)
 assert.equal(f.saved.complete,true)
})

test('topic extraction status identifies the document, source range, part, saved topics and model', () => {
 const {topicExtractionStatus}=requireTranspiledTs('src/shared/topic-inventory-generation.ts')
 const batch={area:'Chapter 5 — Buffer Management',sources:[{documentTitle:'Database Book',pageStart:23,pageEnd:24},{pageStart:26,pageEnd:28}]}
 const status=topicExtractionStatus(batch,6,18,58,model)
 for(const text of ['Extracting topics','Database Book','Chapter 5','Pages 23–28','Text batch 6 of 18','58 topics saved','Gemma 4 E4B']) assert.ok(status.includes(text),status)
 const markdown=topicExtractionStatus({...batch,sources:[{documentTitle:'notes.md',lineFrom:40,lineTo:85}]},1,2,0,model)
 const single=topicExtractionStatus(batch,1,1,0,model)
 assert.ok(!single.includes('1 of 1')); assert.ok(!single.includes('batch')); assert.ok(single.includes('Pages 23–28'))
 assert.ok(markdown.includes('Lines 40–85'));assert.ok(!markdown.includes('Pages'))
})

test('export is blocked during indexing but allowed while chapter topics are being extracted', async () => {
 const f=fixture()
 let releaseIndex, releaseTopics, enteredTopics
 const indexedGate=new Promise(resolve=>releaseIndex=resolve)
 f.gate=new Promise(resolve=>releaseTopics=resolve)
 const topicsEntered=new Promise(resolve=>enteredTopics=resolve)
 const index=f.deps.index, send=f.bridge.sendChatMessage
 f.deps.index=async(...args)=>{await indexedGate;return index(...args)}
 f.bridge.sendChatMessage=async request=>{enteredTopics();return send(request)}
 const job=f.service.run('c/1','book.md')
 assert.equal(f.service.isCollectionIndexing('c'),true)
 releaseIndex();await topicsEntered
 assert.equal(f.service.isCollectionBusy('c'),true)
 assert.equal(f.service.isCollectionIndexing('c'),false)
 releaseTopics();await job
 assert.equal(f.service.isCollectionBusy('c'),false)
})

test('Basic with topic extraction off completes without a chat model or any topic calls', async()=>{
 const f=fixture();f.deps.state=async()=>({models:[],settings:{application:{}}})
 f.deps.topics=()=>{throw new Error('Topic extraction must stay off')}
 const result=await f.service.run('c/1','book.md',undefined,{preparation:{mode:'basic',ocr:'none',extractTopics:false}})
 assert.equal(result.status,'ready');assert.equal(f.calls,0);assert.equal(f.saved,null)
 assert.ok(!f.events.some(event=>event.phase==='topics'))
})
test('LLM chunking can finish with topic extraction explicitly off',async()=>{
 const f=fixture()
 await f.service.run('c/1','book.md',undefined,{preparation:{mode:'ai',extractTopics:false}})
 assert.equal(f.calls,0);assert.equal(f.saved,null)
})


test('topics can be extracted later without calling the indexer or requiring original files',async()=>{
 const f=fixture()
 f.deps.materials=async()=>[{...f.material,preparation:{mode:'basic',extractTopics:false}}]
 f.deps.index=async()=>{throw new Error('Must not reindex for topic extraction')}
 const result=await f.service.run('c/1','',undefined,{preparationModel:model},true)
 assert.equal(result.status,'ready');assert.ok(f.calls>0);assert.equal(f.saved.complete,true)
 assert.ok(!f.events.includes('index'))
 const topics=await loadSavedTopics(f.bridge,f.docs,[{...f.material,preparation:{extractTopics:false}}])
 assert.ok(topics.length>0,'Manually extracted topics remain available when automatic extraction is off')
})
