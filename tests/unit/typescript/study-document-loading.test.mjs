import test from 'node:test'
import assert from 'node:assert/strict'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { loadStudyDocuments } = requireTranspiledTs('src/shared/study-document-loading.ts')
const ipc = message => Error(`Error invoking remote method 'library:study-documents': Error: ${message}`)
for (const message of ['The library changed during this request. Please try again.', 'The library is updating. Please try again shortly.', 'The library was closed. Open it from Libraries to continue.']) {
 test(`catalog recovers automatically: ${message}`, async () => {
  let calls=0;const delays=[];const fresh=[{materialId:'new/1',documentId:2}]
  const result=await loadStudyDocuments(async()=>{if(++calls<3)throw ipc(message);return fresh},new AbortController().signal,async ms=>{delays.push(ms)})
  assert.deepEqual(result,fresh);assert.equal(calls,3);assert.deepEqual(delays,[150,300])
 })
}
test('persistent churn is bounded and never leaks the IPC wrapper',async()=>{
 let calls=0
 await assert.rejects(loadStudyDocuments(async()=>{calls++;throw ipc('The library is updating.')},new AbortController().signal,async()=>{}),{message:'Documents are still updating. Please try again in a moment.'})
 assert.equal(calls,5)
})
test('genuine failures are not retried and retain useful error text',async()=>{
 let calls=0
 await assert.rejects(loadStudyDocuments(async()=>{calls++;throw ipc('Cannot read this library. Check file permissions.')},new AbortController().signal),{message:'Cannot read this library. Check file permissions.'})
 assert.equal(calls,1)
})
test('changing libraries cancels retries and discards late results',async()=>{
 const c=new AbortController();let calls=0
 await assert.rejects(loadStudyDocuments(async()=>{calls++;throw ipc('The library is updating.')},c.signal,async()=>c.abort()),{name:'AbortError'})
 assert.equal(calls,1)
 const d=new AbortController()
 await assert.rejects(loadStudyDocuments(async()=>{d.abort();return [{materialId:'closed/1'}]},d.signal),{name:'AbortError'})
})
