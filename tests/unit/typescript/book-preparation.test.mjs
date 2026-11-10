import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const {createBookPreparation}=requireTranspiledTs('src/main/collections/book-preparation.ts')
const {loadSavedTopics}=requireTranspiledTs('src/shared/saved-topics.ts')
const model={id:'gemma',engine:'ollama',ollamaModelName:'gemma4:26b',status:'ready'}
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
test('topic failure keeps import incomplete and can be retried from preparation',async()=>{
 const f=fixture();f.fail=true;await assert.rejects(f.service.run('c/1','book.md'),/Model stopped/)
 assert.ok(!f.events.some(e=>e.phase==='complete'));assert.equal(f.events.at(-1).phase,'error')
 f.fail=false;await f.service.run('c/1','book.md');assert.equal(f.saved.complete,true)
})
test('cancellation during topics never marks a book ready',async()=>{
 const f=fixture();let release;f.gate=new Promise(r=>release=r)
 const job=f.service.run('c/1','book.md');await new Promise(r=>setTimeout(r,10))
 await f.service.cancel('c/1');release();await assert.rejects(job,/abort/i)
 assert.ok(!f.events.some(e=>e.phase==='complete'));assert.equal(f.saved,null)
})
test('missing local model stops import before indexing',async()=>{
 const f=fixture();f.deps.state=async()=>({models:[]})
 await assert.rejects(f.service.run('c/1','book.md'),/gemma4:26b/);assert.ok(!f.events.includes('index'))
})
test('Practice rejects missing, incomplete, old and stale inventories without writes or model calls',async()=>{
 const docs=[{materialId:'c/1',documentId:1}],materials=[{id:'c/1',collectionRevision:'r1'}]
 for(const saved of [null,{version:2,complete:true,indexRevision:'r1'},{version:3,complete:false,indexRevision:'r1'},{version:3,complete:true,indexRevision:'old'}])
  await assert.rejects(loadSavedTopics({loadTopicInventory:async()=>saved},docs,materials),/preparation in Libraries/)
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
