import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  modelWithRememberedRemoteApiKey,
  rememberRemoteModelApiKey,
  sanitizeAppStateSecrets,
  sanitizeRemoteModelSecrets
} = requireTranspiledTs('src/main/engine/remote-model-secrets.ts')

const remoteModel = {
  id: 'remote-gemini-test',
  name: 'Gemini gemini-2.0-flash',
  engine: 'remote',
  source: 'remote',
  status: 'ready',
  providerId: 'gemini',
  providerName: 'Gemini',
  baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
  apiKey: 'gemini-secret-key',
  remoteModelName: 'gemini-2.0-flash',
  addedAt: new Date(0).toISOString()
}

test('sanitizeRemoteModelSecrets removes remote provider API keys before storage', () => {
  const sanitized = sanitizeRemoteModelSecrets(remoteModel)

  assert.equal(sanitized.apiKey, undefined)
  assert.equal(sanitized.id, remoteModel.id)
  assert.equal(sanitized.remoteModelName, remoteModel.remoteModelName)
})

test('remembered remote API keys can be restored for the current app session', () => {
  rememberRemoteModelApiKey(remoteModel)

  const restored = modelWithRememberedRemoteApiKey({
    ...sanitizeRemoteModelSecrets(remoteModel),
    status: 'needsRuntime'
  })

  assert.equal(restored.apiKey, remoteModel.apiKey)
  assert.equal(restored.status, 'ready')
})

test('sanitizeAppStateSecrets redacts remote keys without touching local models', () => {
  const localModel = {
    id: 'local-model',
    name: 'Llama 3 8B Instruct',
    engine: 'python',
    status: 'ready',
    path: '/models/llama.gguf',
    addedAt: new Date(0).toISOString()
  }
  const sanitized = sanitizeAppStateSecrets({
    version: 1,
    appVersion: '0.1.0',
    activeScreen: 'chat',
    activeConversationId: 'conversation-1',
    conversations: [],
    materials: [],
    models: [remoteModel, localModel],
    selectedModelId: remoteModel.id,
    selectedEmbeddingModelId: '',
    settings: {},
    updatedAt: new Date(0).toISOString()
  })

  assert.equal(sanitized.models[0].apiKey, undefined)
  assert.equal(sanitized.models[1].path, localModel.path)
})
