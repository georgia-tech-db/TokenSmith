import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { ModelContextMetadataService, parseOllamaContextMetadata, parseRemoteContextMetadata } = requireTranspiledTs('src/main/engine/model-context-metadata.ts')
const response = data => new Response(JSON.stringify(data), {status: 200})

test('Ollama uses the text architecture limit, not an audio limit or Modelfile default', () => {
  assert.deepEqual(parseOllamaContextMetadata({ parameters: 'num_ctx 2048', model_info: {
    'gemma4.audio.context_length': 1500, 'general.architecture': 'gemma4', 'gemma4.context_length': 262144
  } }), { contextLength: 262144 })
  assert.deepEqual(parseOllamaContextMetadata({ parameters: 'num_ctx 2048' }), {})
})

test('provider metadata keeps total, input, and output limits distinct', () => {
  assert.deepEqual(parseRemoteContextMetadata({context_window:131072,max_completion_tokens:32768}), {contextLength:131072,maxOutputTokens:32768})
  assert.deepEqual(parseRemoteContextMetadata({max_context_length:131072}), {contextLength:131072})
  assert.deepEqual(parseRemoteContextMetadata({inputTokenLimit:1048576,outputTokenLimit:65536}), {inputTokenLimit:1048576,maxOutputTokens:65536})
  assert.deepEqual(parseRemoteContextMetadata({id:'gpt-test',object:'model',created:123}), {})
})

test('metadata lookup is cached and concurrent requests share one non-inference call', async () => {
  let calls = 0
  const service = new ModelContextMetadataService(async (url, init) => {
    calls++
    assert.equal(String(url), 'http://127.0.0.1:11434/api/show')
    assert.equal(init.redirect, 'error')
    assert.deepEqual(JSON.parse(init.body), {model:'gemma4:26b'})
    return response({model_info:{'gemma4.context_length':262144}})
  })
  const model = {engine:'ollama',ollamaModelName:'gemma4:26b'}
  const values = await Promise.all([service.get(model), service.get(model)])
  assert.deepEqual(values, [{contextLength:262144},{contextLength:262144}])
  await service.get(model)
  assert.equal(calls, 1)
})

test('Gemini metadata uses the same Google service and a header instead of a key in the URL', async () => {
  const service = new ModelContextMetadataService(async (url, init) => {
    assert.equal(url, 'https://generativelanguage.googleapis.com/v1beta/models/gemini-test')
    assert.equal(init.headers['x-goog-api-key'], 'secret')
    assert.equal(init.body, undefined)
    return response({inputTokenLimit:1048576,outputTokenLimit:65536})
  })
  assert.deepEqual(await service.get({engine:'remote',baseUrl:'https://generativelanguage.googleapis.com/v1beta/openai/',remoteModelName:'models/gemini-test',apiKey:'secret'}), {inputTokenLimit:1048576,maxOutputTokens:65536})
})

test('custom providers can supply limits in their model list when the detail endpoint is unavailable', async () => {
  const urls=[]
  const service = new ModelContextMetadataService(async (url, init) => {
    urls.push(url)
    assert.equal(init.headers.Authorization, 'Bearer secret')
    return url.endsWith('/models') ? response({data:[{id:'other',context_length:4096},{id:'org/model',context_length:65536}]}) : new Response('',{status:404})
  })
  const metadata = await service.get({engine:'remote',baseUrl:'https://custom.test/v1',remoteModelName:'org/model',apiKey:'secret'})
  assert.deepEqual(metadata,{contextLength:65536})
  assert.deepEqual(urls,['https://custom.test/v1/models/org%2Fmodel','https://custom.test/v1/models'])
})

test('OpenAI missing limits and failed discovery remain unknown, without guessed limits or leaked errors', async () => {
  let calls=0
  const service = new ModelContextMetadataService(async () => {calls++;return response({id:'gpt-test',object:'model'})})
  assert.deepEqual(await service.get({engine:'remote',baseUrl:'https://api.openai.com/v1',remoteModelName:'gpt-test',apiKey:'secret'}),{})
  assert.equal(calls,1)
  const offline = new ModelContextMetadataService(async () => {throw new Error('secret')})
  assert.deepEqual(await offline.get({engine:'ollama',ollamaModelName:'test'}),{})
})
