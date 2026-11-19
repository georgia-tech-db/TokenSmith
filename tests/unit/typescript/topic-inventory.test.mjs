import test from 'node:test'
import assert from 'node:assert/strict'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { buildTopicBatches, consolidateTopics, parseInventoryTopics } = requireTranspiledTs('src/shared/topic-inventory.ts')
const { generateTopicInventory } = requireTranspiledTs('src/shared/topic-inventory-generation.ts')
const source = (n, heading, documentId = 1) => ({ materialId:'collection/1', documentId, chunkId:String(n), title:'Book', documentTitle:'Book', sectionHeader:heading, excerpt:'A substantive explanation. '.repeat(30) })
const model = {id:'ollama:gemma4:26b',engine:'ollama',ollamaModelName:'gemma4:26b'}
const key = s => `${s.materialId}:${s.documentId}:${s.sourceUnitId}`
const reply = (sources,title) => ({sources,text:JSON.stringify({topics:[{title,aliases:[],objectives:['Mechanism','Tradeoffs'],sourceKeys:[key(sources[0])]}]})})
test('chapters retain all indexed prose, omit code/exercises, and fingerprint late edits', () => {
  const rows=[source(1,'Chapter 5. Buffer Management'),source(2,'5.1 Architecture'),source(3,'Chapter 6. Replacement Policies'),source(4,'6.1 LRU')]
  rows.push({...source(5,'6.2 Code'),chunkKind:'code'},source(6,'6.3 Check Your Understanding'))
  const batches=buildTopicBatches(rows)
  assert.equal(batches.length,2)
  assert.equal(batches.flatMap(b=>b.sources).length,4)
  assert.equal(batches[0].sources[1].excerpt,rows[1].excerpt.trim())
  const earlier=batches[0].fingerprint
  rows[3].excerpt+=' Updated.'
  assert.equal(buildTopicBatches(rows)[0].fingerprint,earlier)
  assert.notEqual(buildTopicBatches(rows)[1].fingerprint,batches[1].fingerprint)
})
test('long chapters split without losing text or their chapter identity', () => {
  const s=source(1,'Chapter 1. Storage');s.excerpt='full text '.repeat(1000)
  const batches=buildTopicBatches([s],2000)
  assert.ok(batches.length>1)
  assert.equal(new Set(batches.map(b=>b.chapterId)).size,1)
  assert.equal(batches.flatMap(b=>b.sources).map(s=>s.excerpt).join(''),s.excerpt.trim())
})
test('PDF chapter includes inline named problems but stops at exercises', () => {
  const rows=[source(1,''),source(2,''),source(3,''),source(4,'')].map(s=>({...s,path:'03-dynprog.pdf'}))
  rows[0].excerpt='Memoization and tabulation'
  rows[1].excerpt='An example.\n3.8 Subset Sum\nThe problem and its recurrence.'
  rows[2].excerpt='Exercises\nPractice problems'
  rows[3].excerpt='Another exercise'
  const batches=buildTopicBatches(rows)
  assert.equal(batches.length,1)
  assert.equal(batches[0].sources.length,2)
  assert.ok(batches[0].sources[1].excerpt.includes('Subset Sum'))
})
test('model names are preserved; only exact terms inside the same chapter coalesce', () => {
  const [a,b]=buildTopicBatches([source(1,'Chapter 1. Indexes'),source(2,'Chapter 2. Queries')])
  const parse=(batch,title)=>parseInventoryTopics(reply(batch.sources,title).text,batch,'collection')
  const one=parse(a,'BuzzDB Text Serialization Format')[0]
  assert.equal(one.title,'BuzzDB Text Serialization Format')
  const topics=consolidateTopics([...parse(a,'B+ Tree Architecture'),...parse(a,'B+ Tree'),...parse(b,'B+ Tree'),...parse(a,'B+ Tree')])
  assert.equal(topics.length,3)
  assert.notEqual(topics[1].id,topics[2].id)
})
test('unsupported source IDs fail without inventing evidence', () => {
  const batch=buildTopicBatches([source(1,'Chapter 1. Storage')])[0]
  const invalid=JSON.parse(reply(batch.sources,'Buffer Pool').text);invalid.topics[0].sourceKeys=['invented']
  assert.throws(()=>parseInventoryTopics(JSON.stringify(invalid),batch,'collection'),/source references/)
})
function fixture() {
  let saved=null, calls=0, fail=false, clipped=false
  const docs=[1,2,3].map(documentId=>({materialId:'collection/1',documentId}))
  const rows=docs.map((d,i)=>source(i,`Chapter ${i+1}. Chapter ${i+1}`,d.documentId))
  const bridge={studyDocuments:async()=>docs,topicSources:async()=>rows,loadTopicInventory:async()=>saved,
    saveTopicInventory:async v=>{saved=structuredClone(v)},sendChatMessage:async r=>{
      calls++;assert.equal(r.practiceTask,'chapter_topics');assert.equal(r.model.ollamaModelName,'gemma4:26b')
      if(fail)throw new Error('Interrupted')
      const response=reply(r.retrievedSources,`Concept ${r.retrievedSources[0].documentId}`)
      if(clipped)response.sources=[]
      return response
    }}
  const run=(signal=new AbortController().signal)=>generateTopicInventory(bridge,[docs[0]],[{id:'collection/1',collectionRevision:'r1'}],model,{application:{}},{},'req',signal,()=>{})
  return {run,rows,bridge,get saved(){return saved},get calls(){return calls},set saved(v){saved=v},set fail(v){fail=v},set clipped(v){clipped=v}}
}
test('generation covers the collection, scopes displayed results, resumes only changed chapters and has no merge pass',async()=>{
  const f=fixture();const topics=await f.run()
  assert.equal(topics.length,1);assert.equal(f.saved.topics.length,3);assert.equal(f.calls,3)
  await f.run();assert.equal(f.calls,3)
  f.rows[0].excerpt+=' change';await f.run();assert.equal(f.calls,4)
  assert.equal(f.saved.version,3);assert.equal(f.saved.complete,true)
})
test('old inventories are discarded rather than used as fallback',async()=>{
  const f=fixture();f.saved={version:2,modelId:model.id,batches:{},topics:[{title:'Old'}],complete:true}
  await f.run();assert.equal(f.calls,3);assert.ok(f.saved.topics.every(t=>t.title!=='Old'))
})
test('context loss and cancellation never save partial generated results',async()=>{
  const f=fixture();f.clipped=true;await assert.rejects(f.run(),/did not fit/);assert.equal(f.saved,null)
  const c=new AbortController();c.abort();await assert.rejects(f.run(c.signal),/abort/i)
})
test('unchanged later checkpoints survive interruption after an early chapter change',async()=>{
  const f=fixture();await f.run();f.rows[0].excerpt+=' change';f.fail=true
  await assert.rejects(f.run(),/Interrupted/)
  assert.equal(Object.keys(f.saved.batches).length,3)
  f.fail=false;await f.run();assert.equal(f.calls,5)
})
test('embedding-only models cannot generate topics',async()=>{
  await assert.rejects(generateTopicInventory({},[],[],{...model,role:'embedder'},{},{},'r',new AbortController().signal,()=>{}),/generation model/)
})

test('invalid topic responses retry with correction and preserve completed batches', async () => {
 const docs=[{materialId:'collection/1',documentId:1}]
 const rows=[source(1,'Chapter 1. Storage'),source(2,'Chapter 2. Indexes')]
 let saved=null,calls=0
 const bridge={studyDocuments:async()=>docs,topicSources:async()=>rows,loadTopicInventory:async()=>saved,
  saveTopicInventory:async v=>{saved=structuredClone(v)},sendChatMessage:async r=>{
   calls++
   if(calls===2) return {sources:r.retrievedSources,text:'{"topics":[]}'}
   if(calls===3) {assert.ok(JSON.parse(r.prompt).correction);assert.equal(saved.completed,1)}
   return reply(r.retrievedSources,'Storage')
  }}
 await generateTopicInventory(bridge,docs,[{id:'collection/1',collectionRevision:'r1'}],model,{application:{}},{},'r',new AbortController().signal,()=>{})
 assert.equal(calls,3);assert.equal(saved.complete,true);assert.equal(saved.completed,2)
})
test('repeated invalid responses stop after three attempts without overwriting saved topics', async () => {
 const docs=[{materialId:'collection/1',documentId:1}]
 let saved=null,calls=0
 const bridge={studyDocuments:async()=>docs,topicSources:async()=>[source(1,'Chapter 1. Storage'),source(2,'Chapter 2. Indexes')],loadTopicInventory:async()=>saved,
  saveTopicInventory:async v=>{saved=structuredClone(v)},sendChatMessage:async r=>{
   calls++;return calls===1?reply(r.retrievedSources,'Storage'):{sources:r.retrievedSources,text:'{"topics":[]}'}
  }}
 await assert.rejects(generateTopicInventory(bridge,docs,[{id:'collection/1',collectionRevision:'r1'}],model,{application:{}},{},'r',new AbortController().signal,()=>{}),/batch 2 of 2 after 3 attempts.*1 topics are saved/)
 assert.equal(calls,4);assert.equal(saved.completed,1);assert.equal(saved.topics.length,1)
})

test('topic validation identifies exact malformed fields and accepts boundary lengths', () => {
 const batch=buildTopicBatches([source(1,'Chapter 1. Storage')])[0]
 const valid=()=>JSON.parse(reply(batch.sources,'Storage').text)
 const cases=[
  [v=>v.topics[0].title='x'.repeat(101), /topics\[0\]\.title: maximum 100 characters; received 101/],
  [v=>v.topics[0].title='Why?', /topics\[0\]\.title/],
  [v=>v.topics[0].title='  ', /topics\[0\]\.title/],
  [v=>v.topics[0].aliases=['x'.repeat(101)], /aliases\[0\]: maximum 100/],
  [v=>v.topics[0].objectives[1]='x'.repeat(501), /objectives\[1\]: maximum 500/],
  [v=>v.topics[0].objectives[0]='  ', /objectives\[0\]: text must not be blank/],
  [v=>v.topics[0].sourceKeys=['invented'], /sourceKeys\[0\]: source references/],
  [v=>v.topics[0]=null, /topics\[0\]: expected an object/],
  [v=>delete v.topics[0].aliases, /aliases: expected an array/],
 ]
 for(const [mutate,expected] of cases){const v=valid();mutate(v);assert.throws(()=>parseInventoryTopics(JSON.stringify(v),batch,'collection'),expected)}
 const boundary=valid();boundary.topics[0].title='x'.repeat(100);boundary.topics[0].aliases=['😀'.repeat(100)];boundary.topics[0].objectives[0]='x'.repeat(500)
 assert.equal(parseInventoryTopics(JSON.stringify(boundary),batch,'collection').length,1)
 assert.throws(()=>parseInventoryTopics('null',batch,'collection'),/expected an object/)
 assert.throws(()=>parseInventoryTopics('{',batch,'collection'),/expected valid JSON/)
})
test('model schema includes the same title, alias, objective and evidence restrictions', () => {
 const {topicResponseSchema}=requireTranspiledTs('src/shared/topic-response.ts')
 const fields=topicResponseSchema(['source-1']).properties.topics.items.properties
 assert.equal(fields.title.maxLength,100);assert.equal(fields.aliases.items.maxLength,100)
 assert.equal(fields.objectives.items.maxLength,500)
 assert.equal(new RegExp(fields.title.pattern).test('Why?'),false)
 assert.equal(new RegExp(fields.objectives.items.pattern).test('  '),false)
 assert.deepEqual(fields.sourceKeys.items.enum,['source-1'])
})
