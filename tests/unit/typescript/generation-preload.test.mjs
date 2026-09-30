import assert from 'node:assert/strict'
import test from 'node:test'
import vm from 'node:vm'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

function bridgeFixture() {
  const listeners = new Map(), pending = new Map()
  let bridge
  const ipcRenderer = {
    on: (name, fn) => { const set=listeners.get(name)||new Set();set.add(fn);listeners.set(name,set) },
    off: (name, fn) => listeners.get(name)?.delete(fn),
    invoke: (channel, request) => channel === 'engine:chat'
      ? new Promise((resolve,reject)=>pending.set(request.requestId,{resolve,reject})) : Promise.resolve()
  }
  const code=ts.transpileModule(readFileSync('src/preload/index.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText
  vm.runInNewContext(code,{exports:{},process:{platform:'test'},require:()=>({ipcRenderer,contextBridge:{exposeInMainWorld:(_name,value)=>{bridge=value}}})})
  return {bridge,pending,listeners,emit:(id,answer)=>{for(const listener of listeners.get('engine:answer-ready')||[])listener({},id,answer,true)}}
}

test('early-answer IPC is scoped to its request and listeners are removed on success or failure', async () => {
  const fixture=bridgeFixture(),seen=[]
  const first=fixture.bridge.sendChatMessage({requestId:'first'},answer=>seen.push(['first',answer.text]))
  const second=fixture.bridge.sendChatMessage({requestId:'second'},answer=>seen.push(['second',answer.text]))
  fixture.emit('first',{text:'Ready early'})
  assert.deepEqual(seen,[['first','Ready early']])
  fixture.pending.get('first').resolve({text:'Ready early',followUpSuggestions:['Next?']})
  await first
  assert.equal(fixture.listeners.get('engine:answer-ready').size,1)
  fixture.emit('first',{text:'Stale'})
  assert.equal(seen.length,1)
  const rejected=assert.rejects(second,/stopped/)
  fixture.pending.get('second').reject(new Error('stopped'))
  await rejected
  assert.equal(fixture.listeners.get('engine:answer-ready').size,0)
})

test('legacy callers without early-answer callbacks still use the final result', async () => {
  const fixture=bridgeFixture()
  const pending=fixture.bridge.sendChatMessage({requestId:'legacy'})
  assert.equal(fixture.listeners.has('engine:answer-ready'),false)
  fixture.pending.get('legacy').resolve({text:'Done'})
  assert.equal((await pending).text,'Done')
})
