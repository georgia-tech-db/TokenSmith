import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { runRemoteStudyEngine } = requireTranspiledTs('src/main/engine/remote-chat-service.ts')
const sources = [{ title: 'Book', locator: 'Page 1', excerpt: 'Atomicity is all or nothing.' }]
const request = engine => ({ prompt:'Explain atomicity', messages:[], materials:[], settings:{}, retrievedSources:sources,
  model: engine === 'ollama' ? { name:'Test', engine, ollamaModelName:'test:12b', contextLength:8192 }
    : { name:'Test', engine, remoteModelName:'test', baseUrl:'https://example.test/v1', apiKey:'test' },
  modelSettings:{contextLength:8192,maxLength:512}, applicationSettings:{suggestionMode:'on',followUpSuggestionCount:2} })
const deferred = () => { let resolve; const promise = new Promise(r => resolve=r); return {promise,resolve} }
for (const [engine, run] of [['ollama', runOllamaStudyEngine], ['remote', runRemoteStudyEngine]]) {
  test(`${engine}: delivers the answer before suggestions finish, without adding inference calls`, async () => {
    const saved = globalThis.fetch
    const followUps = deferred(), started = deferred()
    const payload = text => engine === 'ollama' ? {message:{content:text}} : {choices:[{message:{content:text}}]}
    let calls = 0, observed, done = false
    globalThis.fetch = async () => {
      calls++
      if (calls === 1) return {ok:true,json:async()=>payload('An atomic transaction happens entirely or not at all.')}
      started.resolve()
      return followUps.promise
    }
    try {
      const pending = run(request(engine), {onAnswer:(answer, hasFollowUps)=>{observed={answer,hasFollowUps}}}).then(result=>{done=true;return result})
      await started.promise
      assert.equal(done, false)
      assert.match(observed.answer.text, /atomic transaction/)
      assert.deepEqual(observed.answer.sources, sources)
      assert.equal(observed.hasFollowUps, true)
      followUps.resolve({ok:true,json:async()=>payload(JSON.stringify({questions:['What happens during a rollback?','How does durability differ?']}))})
      const result = await pending
      assert.equal(result.text, observed.answer.text)
      assert.equal(calls, 2)
    } finally {globalThis.fetch=saved}
  })
  test(`${engine}: stopping during a response body aborts inference and keeps the already delivered answer`, async () => {
    const saved = globalThis.fetch
    const controller = new AbortController(), started = deferred()
    let calls = 0, observed
    globalThis.fetch = async (_url, options) => {
      calls++
      if (calls === 1) return {ok:true,json:async()=> engine === 'ollama' ? {message:{content:'The answer is ready.'}} : {choices:[{message:{content:'The answer is ready.'}}]}}
      return {ok:true,json:()=>new Promise((_resolve,reject)=>{
        options.signal.addEventListener('abort', ()=>reject(options.signal.reason), {once:true})
        started.resolve()
      })}
    }
    try {
      const pending = run(request(engine), {signal:controller.signal,onAnswer:answer=>{observed=answer}})
      const assertion = assert.rejects(pending, /abort/i)
      await started.promise
      controller.abort()
      await assertion
      assert.equal(observed.text, 'The answer is ready.')
      assert.equal(calls, 2)
    } finally {globalThis.fetch=saved}
  })
}
