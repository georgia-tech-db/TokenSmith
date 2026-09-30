import assert from 'node:assert/strict'
import test from 'node:test'
import vm from 'node:vm'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { normalizeEmbeddingGpuEnabled } = requireTranspiledTs('src/shared/embedding-settings.ts')
const { createEmbeddingDeviceManager } = requireTranspiledTs('src/main/engine/embedding-device.ts')
const { removeRetiredModelState } = requireTranspiledTs('src/shared/retired-model-state.ts')
const model = { engine: 'ollama', role: 'embedder', ollamaModelName: 'nomic-embed-text' }

test('preload forwards the embedding device preference for indexing and search', async () => {
  let bridge
  const calls = []
  const ipcRenderer = { invoke: async (...args) => { calls.push(args) } }
  const code = ts.transpileModule(readFileSync('src/preload/index.ts', 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText
  vm.runInNewContext(code, { exports: {}, process: { platform: 'test' }, require: () => ({
    ipcRenderer, contextBridge: { exposeInMainWorld: (_name, value) => { bridge = value } }
  }) })
  const options = { embeddingGpuEnabled: false }
  await bridge.indexMaterial('book', '/course', model, options)
  await bridge.searchLibrary('Why?', [], 4, [model], 'hybrid', options)
  assert.deepEqual(calls, [
    ['library:index-material', 'book', '/course', model, options],
    ['library:search', 'Why?', [], 4, [model], 'hybrid', options]
  ])
})

test('embedding GPU is on for fresh/old settings; only an explicit false disables it', () => {
  for (const value of [undefined, null, true, '', 'false', 0]) assert.equal(normalizeEmbeddingGpuEnabled(value), true)
  assert.equal(normalizeEmbeddingGpuEnabled(false), false)
  assert.equal(normalizeEmbeddingGpuEnabled(JSON.parse('{"embeddingGpuEnabled":false}').embeddingGpuEnabled), false)
})

test('GPU to CPU to GPU unloads only the embedding runner once when returning to auto', async () => {
  const calls = []
  const prepare = createEmbeddingDeviceManager(async (url, options) => {
    calls.push({ url, body: options?.body && JSON.parse(options.body) })
    return Response.json({ models: [{ name: 'nomic-embed-text:latest', size_vram: 200 }] })
  })
  await prepare([model])
  await prepare([model], true)
  await prepare([model], false)
  await prepare([model], false)
  await Promise.all([prepare([model], true), prepare([model], true)])
  assert.deepEqual(calls, [
    { url: 'http://127.0.0.1:11434/api/ps', body: undefined },
    { url: 'http://127.0.0.1:11434/api/embed', body: { model: 'nomic-embed-text', input: [], keep_alive: 0 } }
  ])
})

test('a CPU runner left over from an earlier app session is reset once, without deleting models', async () => {
  const calls = []
  const prepare = createEmbeddingDeviceManager(async (url, options) => {
    calls.push({ url, method: options?.method })
    return Response.json({ models: [{ model: 'nomic-embed-text:latest', size_vram: 0 }] })
  })
  await prepare([model], true)
  await prepare([model], true)
  assert.equal(calls.length, 2)
  assert.equal(calls[1].method, 'POST')
  assert.ok(calls[1].url.endsWith('/api/embed'))
})

test('cloud embeddings are untouched and failed device switches can be retried', async () => {
  let calls = 0
  const prepare = createEmbeddingDeviceManager(async () => {
    calls++
    return new Response('', { status: calls === 1 ? 500 : 200 })
  })
  await prepare([{ engine: 'remote', remoteModelName: 'embed' }], false)
  await prepare([model], false)
  assert.equal(calls, 0)
  await assert.rejects(prepare([model], true), /Could not switch.*500/)
  await prepare([model], true)
  assert.equal(calls, 2)
})

test('retired GGUF records are discarded without removing Ollama GGUF models or unrelated chats', () => {
  const state = {
    models: [{ id: 'old', engine: 'python' }, { id: 'ollama', engine: 'ollama', name: 'my-model.gguf' }],
    materials: [{ id: 'old-book', embeddingModel: 'llama-cpp:123' }, { id: 'book', embeddingModel: 'ollama:nomic-embed-text' }],
    conversations: [
      { id: 'old-chat', messages: [{ role: 'assistant', sources: [{ materialId: 'old-book' }] }] },
      { id: 'detached-old-chat', messages: [{ sources: [{ chunkEmbeddingModel: 'llama-cpp:456' }] }] },
      { id: 'chat', messages: [{ text: 'What is GGUF?', sources: [{ materialId: 'book' }] }] }
    ],
    activeConversationId: 'old-chat', selectedModelId: 'old', selectedEmbeddingModelId: 'ollama',
    settings: { application: { defaultModelId: 'old' }, modelSettingsById: { old: {}, ollama: { temperature: 0.5 } } }
  }
  const result = removeRetiredModelState(state)
  assert.deepEqual(result.models.map(item => item.id), ['ollama'])
  assert.deepEqual(result.materials.map(item => item.id), ['book'])
  assert.deepEqual(result.conversations.map(item => item.id), ['chat'])
  assert.equal(result.activeConversationId, 'chat')
  assert.equal(result.selectedModelId, '')
  assert.deepEqual(result.settings.modelSettingsById, { ollama: { temperature: 0.5 } })
  assert.deepEqual(removeRetiredModelState(result), result)
  assert.equal(state.models.length, 2)
})
