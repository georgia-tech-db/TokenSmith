import test from 'node:test'
import assert from 'node:assert/strict'
import { EventEmitter, once } from 'node:events'
import { readFileSync } from 'node:fs'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const {createBookPreparation}=requireTranspiledTs('src/main/collections/book-preparation.ts')
const {loadSavedTopics}=requireTranspiledTs('src/shared/saved-topics.ts')
const model={id:'gemma',engine:'ollama',ollamaModelName:'gemma4:e4b',status:'ready'}
function fixture() {
 let saved=null, calls=0, fail=false, gate=null
 const events=[]
 const topics=new EventEmitter()
 const material={id:'c/1',status:'ready',collectionRevision:'r1'}
 const docs=[{materialId:'c/1',documentId:1}]
 const bridge={studyDocuments:async()=>docs,topicSources:async()=>[{...docs[0],chunkId:'1',documentTitle:'Book',title:'Book',excerpt:'Chapter 1. Storage\nBuffer pools cache database pages.'}],loadTopicInventory:async()=>saved,saveTopicInventory:async x=>{saved=structuredClone(x)},sendChatMessage:async r=>{
  calls++;if(gate)await gate;if(fail)throw new Error('Model stopped');return {sources:r.retrievedSources,text:JSON.stringify({topics:[{title:'Buffer pools',aliases:[],objectives:['Caching','Eviction'],sourceKeys:r.practiceAllowedIds.slice(0,1)}]})}
 }}
 const deps={materials:async()=>[material],topicProgress:p=>topics.emit(p.status === 'error' ? 'failed' : p.status,p),state:async()=>({models:[model],settings:{application:{},modelDefaults:{},modelSettingsById:{}}}),index:async(id,path,m,o,progress)=>{events.push('index');progress({materialId:id,phase:'complete',percent:100});return material},topics:()=>bridge,progress:p=>events.push(p),cancelIndex:async()=>events.push('cancel')}
 const service=createBookPreparation(deps)
 return {service,events,topics,bridge,material,docs,deps,get saved(){return saved},get calls(){return calls},set fail(v){fail=v},set gate(v){gate=v}}
}
test('explicit topic extraction starts after import reports ready', async () => {
 const f=fixture()
 const entered=Promise.withResolvers(), release=Promise.withResolvers()
 const send=f.bridge.sendChatMessage
 f.bridge.sendChatMessage=async request=>{entered.resolve();await release.promise;return send(request)}
 const finished=once(f.topics,'complete')
 const job=f.service.run('c/1','book.md',undefined,{preparation:{mode:'basic',extractTopics:true}})
 assert.equal(f.service.isCollectionIndexing('c'),true)
 assert.equal((await job).status,'ready')
 await entered.promise
 assert.equal(f.service.isCollectionIndexing('c'),false)
 assert.equal(f.events.at(-1).phase,'complete')
 assert.equal(f.saved,null)
 release.resolve()
 await finished
 assert.equal(f.saved.complete,true)
})
for (const mode of ['basic', 'ai']) {
 test(`${mode} import uses the saved material ID for topics and the request ID for progress`, async () => {
  const f=fixture()
  const requestId='c/import-pending'
  const options={preparation:{mode,extractTopics:true,instructions:'',documentInstructions:{}}}
  const index=f.deps.index
  f.deps.index=async(id,path,embedder,received,progress)=>{
   assert.equal(id,requestId);assert.equal(path,'book.pdf');assert.deepEqual(received.preparation,options.preparation);if(mode==='ai')assert.equal(received.preparationModel.id,model.id)
   return index(id,path,embedder,received,progress)
  }
  const finished=once(f.topics,'complete')
  const material=await f.service.run(requestId,'book.pdf',undefined,options)
  await finished
  assert.equal(material.id,f.material.id)
  assert.equal(f.saved.complete,true)
  assert.equal(f.saved.collectionId,'c')
  assert.ok(f.saved.topics.length>0)
  assert.ok(f.saved.topics.every(topic=>topic.sources.every(source=>source.materialId===f.material.id)))
  assert.equal(f.events.at(-1).phase,'complete')
  assert.ok(f.events.filter(event=>event.phase).every(event=>event.materialId===requestId))
 })
}
test('a topic-only failure leaves the index ready and retry uses the same index', async () => {
 const f=fixture()
 await f.service.run('c/1','book.md')
 f.fail=true
 await assert.rejects(f.service.extractTopics('c'),/Model stopped/)
 assert.equal(f.events.at(-1).phase,'complete')
 f.fail=false
 await f.service.extractTopics('c')
 assert.equal(f.saved.complete,true)
 assert.equal(f.events.filter(event=>event==='index').length,1)
})

test('pausing topic extraction leaves the index ready and allows retry', async () => {
 const f=fixture()
 await f.service.run('c/1','book.md')
 const entered=Promise.withResolvers(), release=Promise.withResolvers()
 const send=f.bridge.sendChatMessage
 f.bridge.sendChatMessage=async request=>{entered.resolve();await release.promise;return send(request)}
 const job=f.service.extractTopics('c')
 await entered.promise
 await assert.rejects(f.service.extractTopics('c'),/already being generated/)
 f.service.cancelTopics('c')
 release.resolve()
 await job
 assert.equal(f.saved,null)
 assert.equal(f.events.at(-1).phase,'complete')
 assert.ok(!f.events.includes('cancel'))
 await f.service.extractTopics('c')
 assert.equal(f.saved.complete,true)
})

test('Basic import still succeeds when an explicitly requested topic model is unavailable', async () => {
 const f=fixture();f.deps.state=async()=>({models:[],settings:{application:{}}})
 const failed=once(f.topics,'failed')
 const material=await f.service.run('c/1','book.md',undefined,{preparation:{mode:'basic',extractTopics:true}})
 assert.equal(material.status,'ready')
 assert.match((await failed)[0].message,/available model/)
 assert.equal(f.calls,0)
})

test('AI chunking still requires an available preparation model', async () => {
 const f=fixture();f.deps.state=async()=>({models:[],settings:{application:{}}})
 await assert.rejects(f.service.run('c/1','book.md',undefined,{preparation:{mode:'ai'}}),/preparation model/)
 assert.ok(!f.events.includes('index'))
})
test('Practice rejects missing, incomplete, old and stale inventories without writes or model calls',async()=>{
 const docs=[{materialId:'c/1',documentId:1}],materials=[{id:'c/1',collectionRevision:'r1'}]
 for(const saved of [null,{version:2,complete:true,indexRevision:'r1'},{version:3,complete:false,indexRevision:'r1'},{version:3,complete:true,indexRevision:'old'}])
  await assert.rejects(loadSavedTopics({loadTopicInventory:async()=>saved},docs,materials),/Generate Practice topics in Libraries/)
})
test('Practice component has no generation, model, write or resume dependency',()=>{
 const code=readFileSync('src/renderer/src/PracticeTopicPicker.tsx','utf8')
 assert.doesNotMatch(code,/generateTopicInventory|extractLibraryTopics|sendChatMessage|saveTopicInventory|Resume topics|modelSettings/)
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
 f.deps.state=async()=>({models:[model,chosen],settings:{application:{},modelDefaults:{},modelSettingsById:{}}})
 const options={preparation:{mode:'ai',extractTopics:true,instructions:'',documentInstructions:{}},preparationModel:chosen}
 const index=f.deps.index
 f.deps.index=async(id,path,embedder,received,progress)=>{assert.equal(received.preparationModel,chosen);return index(id,path,embedder,received,progress)}
 const send=f.bridge.sendChatMessage
 f.bridge.sendChatMessage=async request=>{assert.equal(request.model,chosen);return send(request)}
 const finished=once(f.topics,'complete')
 await f.service.run('c/1','book.pdf',undefined,options)
 await finished
 assert.equal(f.saved.modelId,chosen.id)
})
test('preparation defaults to E4B but keeps an explicit selection and refuses unavailable selections', () => {
 const {preparationGenerator}=requireTranspiledTs('src/shared/preparation.ts')
 const other={id:'other',engine:'ollama',status:'ready'}
 assert.equal(preparationGenerator([other,model],undefined,'other'),model)
 assert.equal(preparationGenerator([other,model],'other'),other)
 assert.equal(preparationGenerator([other,model],'missing'),undefined)
})

test('saved preparation model is used for indexing with topics deferred', async () => {
 const f=fixture(), chosen={id:'saved',engine:'ollama',ollamaModelName:'saved-model',status:'ready'}
 f.deps.state=async()=>({models:[model,chosen],settings:{application:{defaultModelId:model.id},modelDefaults:{},modelSettingsById:{}}})
 const index=f.deps.index
 f.deps.index=async(id,path,embedder,options,progress)=>{assert.equal(options.preparationModel.id,chosen.id);return index(id,path,embedder,options,progress)}
 await f.service.run('c/1','book.pdf',undefined,{preparation:{mode:'ai',modelId:chosen.id}})
 assert.equal(f.calls,0)
 assert.equal(f.saved,null)
})

test('indexing reports its own progress through completion',async()=>{
 const f=fixture()
 f.deps.index=async(id,path,model,options,progress)=>{
  progress({materialId:id,phase:'chunking',percent:25,message:'Reading page 6 of 10'})
  progress({materialId:id,phase:'complete',percent:100})
  return f.material
 }
 await f.service.run('c/1','book.pdf')
 const page=f.events.find(e=>e.phase==='chunking')
 assert.equal(page.percent,25)
 assert.equal(f.events.at(-1).phase,'complete')
 assert.equal(f.events.at(-1).percent,100)
 assert.equal(f.saved,null)
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
 await f.service.extractTopics('c',model.id)
 assert.ok(f.calls>0);assert.equal(f.saved.complete,true)
 assert.ok(!f.events.includes('index'))
 const topics=await loadSavedTopics(f.bridge,f.docs,[{...f.material,preparation:{extractTopics:false}}])
 assert.ok(topics.length>0,'Manually extracted topics remain available when automatic extraction is off')
})

test('selecting an indexed book without topics preserves saved topics from another library', async () => {
 const f=fixture()
 await f.service.extractTopics('c')
 const before=f.calls
 const store={loadTopicInventory:async id=>id==='c'?f.saved:null}
 const docs=[...f.docs,{materialId:'other/1',documentId:1}]
 const materials=[{id:'c/pending'},f.material,{id:'other/1',collectionRevision:'r2'}]
 const topics=await loadSavedTopics(store,docs,materials)
 assert.equal(topics[0].title,'Buffer pools')
 assert.equal(f.calls,before)
})
