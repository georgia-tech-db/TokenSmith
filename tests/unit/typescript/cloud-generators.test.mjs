import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { CloudGeneratorService, cloudResult, cloudChatModelIds, cloudModelIds } = requireTranspiledTs('src/main/engine/cloud-generator-service.ts')
const { attachCloudProviderConnection, disconnectCloudConnectionModels, mergeCloudConnectionModel } = requireTranspiledTs('src/shared/cloud-generators.ts')
const { remoteChatParameters } = requireTranspiledTs('src/main/engine/remote-chat-parameters.ts')
const { setCloudCredentialResolver, modelWithRememberedRemoteApiKey, sanitizeAppStateSecrets } = requireTranspiledTs('src/main/engine/remote-model-secrets.ts')

const key = randomBytes(32)
const encryption = {
  isEncryptionAvailable: () => true,
  getSelectedStorageBackend: () => 'keychain',
  encryptString(value) {
    const iv = randomBytes(12); const cipher = createCipheriv('aes-256-gcm', key, iv)
    const ciphertext = Buffer.concat([cipher.update(value, 'utf8'), cipher.final()])
    return Buffer.concat([iv, cipher.getAuthTag(), ciphertext])
  },
  decryptString(value) {
    const decipher = createDecipheriv('aes-256-gcm', key, value.subarray(0, 12))
    decipher.setAuthTag(value.subarray(12, 28))
    return Buffer.concat([decipher.update(value.subarray(28)), decipher.final()]).toString('utf8')
  }
}
const input = { providerId: 'custom', baseUrl: 'http://127.0.0.1:9999/v1', apiKey: 'test-only-secret', modelName: 'chat-test', remember: true }
function success(url) {
  return Promise.resolve(new Response(JSON.stringify(url.endsWith('/models') ? { data: [{ id: 'chat-test' }] } : { choices: [{ message: { content: 'OK' } }] }), { status: 200 }))
}
async function fixture(t, request = success, encrypt = encryption) {
  const folder = await mkdtemp(join(tmpdir(), 'tokensmith-cloud-test-'))
  t.after(() => rm(folder, { recursive: true, force: true }))
  const file = join(folder, 'connections.json')
  const service = new CloudGeneratorService(file, encrypt, request)
  await service.initialize()
  return { service, file }
}

test('connection check uses only a generic prompt; keys stay in main and encrypted across restart', async t => {
  const calls = []
  const { service, file } = await fixture(t, (url, init) => { calls.push({ url, init }); return success(url) })
  const model = await service.connect(input)
  assert.equal(model.role, 'generator')
  assert.equal(model.apiKey, undefined)
  assert.equal(service.credentialFor(model), input.apiKey)
  assert.deepEqual(JSON.parse(calls[0].init.body).messages, [{ role: 'user', content: 'Reply with the word OK.' }])
  assert.equal(JSON.parse(calls[0].init.body).max_tokens, 512)
  assert.equal(calls[0].init.redirect, 'error')
  assert.ok(calls[0].init.signal)
  assert.equal(JSON.stringify(service.status()).includes(input.apiKey), false)
  assert.equal((await readFile(file, 'utf8')).includes(input.apiKey), false)
  if (process.platform !== 'win32') assert.equal((await stat(file)).mode & 0o777, 0o600)
  const restored = new CloudGeneratorService(file, encryption, success)
  await restored.initialize()
  assert.equal(restored.credentialFor(model), input.apiKey)
  assert.equal(restored.describeModel({ ...model, status: 'needsRuntime' }).status, 'ready')
})

test('forgetting a connection deletes its credential and saved record', async t => {
  const { service, file } = await fixture(t)
  const model = await service.connect(input)
  const forgotten = await service.forget(model.connectionId)
  assert.deepEqual(forgotten.connectionIds, [model.connectionId])
  assert.equal(service.credentialFor(model), undefined)
  assert.deepEqual(service.status().connections, [])
  const restored = new CloudGeneratorService(file, encryption, success)
  await restored.initialize()
  assert.deepEqual(restored.status().connections, [])
})

test('forgetting a fixed provider removes every legacy credential for that provider across restart', async t => {
  const { service, file } = await fixture(t)
  const first = await service.connect({ ...input, providerId: 'groq', modelName: 'llama-3.3-70b-versatile' })
  // Simulate a pre-migration duplicate provider record from the older per-model flow.
  service.records.set('legacy-groq', { id: 'legacy-groq', providerId: 'groq', baseUrl: first.baseUrl, remembered: true, encryptedKey: encryption.encryptString('old-test-secret').toString('base64') })
  service.keys.set('legacy-groq', 'old-test-secret')
  // Exercise the edge case where the selected connection is session-only but a duplicate is persisted.
  service.records.set(first.connectionId, { ...service.records.get(first.connectionId), remembered: false, encryptedKey: undefined })
  const forgotten = await service.forget(first.connectionId)
  assert.deepEqual(new Set(forgotten.connectionIds), new Set([first.connectionId, 'legacy-groq']))
  assert.equal(service.status().connections.filter(connection => connection.providerId === 'groq').length, 0)
  const restored = new CloudGeneratorService(file, encryption, success)
  await restored.initialize()
  assert.equal(restored.status().connections.filter(connection => connection.providerId === 'groq').length, 0)
})

test('provider credentials are shared by chat and embedding models and reconnect repairs every referenced model', async t => {
  const { service } = await fixture(t, url => url.endsWith('/embeddings')
    ? Promise.resolve(new Response(JSON.stringify({ data: [{ embedding: [0.1] }] }), { status: 200 }))
    : success(url))
  const a = await service.connect(input)
  const b = await service.connect({ ...input, apiKey: undefined, connectionId: a.connectionId, modelName: 'embed-test', role: 'embedder' })
  assert.equal(a.connectionId, b.connectionId)
  assert.deepEqual(await service.discover({ providerId: 'custom', connectionId: a.connectionId }), ['chat-test'])
  const repaired = await service.connect({ ...input, apiKey: 'replacement-test-secret', connectionId: a.connectionId, modelId: a.id })
  assert.equal(service.credentialFor(b), 'replacement-test-secret')
  const localEmbedder = { id: 'nomic', engine: 'ollama', role: 'embedder', status: 'ready', ollamaModelName: 'nomic-embed-text' }
  const models = mergeCloudConnectionModel([localEmbedder, { ...a, status: 'needsRuntime' }, { ...b, status: 'needsRuntime' }], repaired)
  assert.equal(models.length, 3)
  assert.equal(models[0].id, a.id)
  assert.equal(models.find(m => m.id === b.id).status, 'ready')
  assert.equal(models.find(m => m.id === 'nomic'), localEmbedder)
  assert.deepEqual(mergeCloudConnectionModel([localEmbedder], { ...repaired, id: 'nomic' }), [localEmbedder])
})

test('a known provider uses one connection even when another model is added with a replacement key', async t => {
  const { service } = await fixture(t)
  const first = await service.connect({ ...input, providerId: 'groq', modelName: 'llama-3.3-70b-versatile' })
  const second = await service.connect({ ...input, providerId: 'groq', apiKey: 'replacement-test-secret', modelName: 'openai/gpt-oss-120b' })
  assert.equal(first.connectionId, second.connectionId)
  assert.equal(service.status().connections.filter(connection => connection.providerId === 'groq').length, 1)
  assert.equal(service.credentialFor(first), 'replacement-test-secret')
})

test('every fixed provider shares one credential across chat and embedding models', async t => {
  const { service } = await fixture(t, url => Promise.resolve(new Response(JSON.stringify(
    url.endsWith('/embeddings') ? { data: [{ embedding: [0.1] }] } : { choices: [{ message: { content: 'OK' } }] }
  ), { status: 200 })))
  for (const providerId of ['groq', 'openai', 'gemini', 'mistral']) {
    const chat = await service.connect({ ...input, providerId, modelName: `${providerId}-chat` })
    const embedder = await service.connect({ ...input, providerId, apiKey: undefined, modelName: `${providerId}-embedding`, role: 'embedder' })
    assert.equal(embedder.connectionId, chat.connectionId, providerId)
    assert.equal(service.credentialFor(embedder), input.apiKey, providerId)
    assert.equal(service.status().connections.filter(connection => connection.providerId === providerId).length, 1, providerId)
  }
})

test('custom providers keep credentials separate by normalized endpoint', async t => {
  const { service } = await fixture(t)
  const first = await service.connect(input)
  const sameEndpoint = await service.connect({ ...input, baseUrl: `${input.baseUrl}/`, apiKey: undefined, modelName: 'chat-two' })
  const otherEndpoint = await service.connect({ ...input, baseUrl: 'http://127.0.0.1:9998/v1', modelName: 'chat-three' })
  assert.equal(first.connectionId, sameEndpoint.connectionId)
  assert.notEqual(first.connectionId, otherEndpoint.connectionId)
  assert.equal(service.status().connections.filter(connection => connection.providerId === 'custom').length, 2)
})

test('connecting a fixed provider repairs older models with stale per-model connections', () => {
  const connected = { id: 'new', engine: 'remote', providerId: 'groq', connectionId: 'groq-shared', baseUrl: 'https://api.groq.com/openai/v1/', status: 'ready' }
  const models = attachCloudProviderConnection([
    { id: 'old-a', engine: 'remote', providerId: 'groq', connectionId: 'old-a', status: 'needsRuntime' },
    { id: 'old-b', engine: 'remote', providerId: 'groq', connectionId: 'old-b', status: 'needsRuntime' },
    { id: 'other-custom', engine: 'remote', providerId: 'custom', connectionId: 'custom-a', baseUrl: 'https://example.test/v1', status: 'needsRuntime' }
  ], connected)
  assert.deepEqual(models.map(model => model.connectionId), ['groq-shared', 'groq-shared', 'custom-a'])
  assert.deepEqual(models.map(model => model.status), ['ready', 'ready', 'needsRuntime'])
})

test('custom connections only repair models at the same endpoint', () => {
  const connected = { id: 'new', engine: 'remote', providerId: 'custom', connectionId: 'custom-shared', baseUrl: 'https://example.test/v1/', status: 'ready' }
  const models = attachCloudProviderConnection([
    { id: 'same', engine: 'remote', providerId: 'custom', connectionId: 'old', baseUrl: 'https://example.test/v1', status: 'needsRuntime' },
    { id: 'different', engine: 'remote', providerId: 'custom', connectionId: 'other', baseUrl: 'https://other.test/v1', status: 'needsRuntime' }
  ], connected)
  assert.deepEqual(models.map(model => model.connectionId), ['custom-shared', 'other'])
})

test('cloud model merging normalizes endpoint URLs and keeps chat and embedding roles separate', () => {
  const chat = { id: 'chat', engine: 'remote', role: 'generator', providerId: 'custom', connectionId: 'shared', baseUrl: 'https://example.test/v1/', remoteModelName: 'multipurpose', status: 'ready' }
  const refreshedChat = { ...chat, id: 'new-chat', baseUrl: 'https://example.test/v1' }
  const embedder = { ...chat, id: 'embedder', role: 'embedder', baseUrl: 'https://example.test/v1' }
  const refreshed = mergeCloudConnectionModel([chat], refreshedChat)
  assert.equal(refreshed.length, 1)
  assert.equal(refreshed[0].id, 'chat')
  const withBothRoles = mergeCloudConnectionModel(refreshed, embedder)
  assert.deepEqual(new Set(withBothRoles.map(model => model.id)), new Set(['chat', 'embedder']))
})

test('all removed legacy connection IDs are marked unavailable in the renderer state', () => {
  const models = disconnectCloudConnectionModels([
    { id: 'a', engine: 'remote', connectionId: 'current', status: 'ready' },
    { id: 'b', engine: 'remote', connectionId: 'legacy', status: 'ready' },
    { id: 'c', engine: 'remote', connectionId: 'other', status: 'ready' }
  ], ['current', 'legacy'])
  assert.deepEqual(models.map(model => model.status), ['needsRuntime', 'needsRuntime', 'ready'])
  assert.deepEqual(models.map(model => model.cloudCredentialStatus), ['reconnect', 'reconnect', undefined])
})

test('embedding validation uses the embeddings endpoint and accepts an embedding response', async t => {
  const calls = []
  const { service } = await fixture(t, (url, init) => {
    calls.push({ url, init })
    return Promise.resolve(new Response(JSON.stringify(url.endsWith('/embeddings') ? { data: [{ embedding: [0.1] }] } : { choices: [{ message: { content: 'OK' } }] }), { status: 200 }))
  })
  const model = await service.connect({ ...input, modelName: 'text-embedding-3-small', role: 'embedder' })
  assert.equal(model.role, 'embedder')
  assert.match(calls[0].url, /\/embeddings$/)
  assert.deepEqual(JSON.parse(calls[0].init.body), { model: 'text-embedding-3-small', input: 'TokenSmith connection check' })
})

test('session-only connections do not write a credential file and visibly require reconnect after restart', async t => {
  const { service, file } = await fixture(t, success, { ...encryption, isEncryptionAvailable: () => false })
  const rejected = await cloudResult(() => service.connect(input))
  assert.equal(rejected.error.code, 'storage')
  const model = await service.connect({ ...input, remember: false })
  await assert.rejects(readFile(file), { code: 'ENOENT' })
  const restarted = new CloudGeneratorService(file, encryption, success)
  await restarted.initialize()
  assert.equal(restarted.describeModel(model).cloudCredentialStatus, 'reconnect')
  assert.equal(restarted.describeModel(model).id, model.id)
})

test('Linux basic_text is not treated as secure storage', async t => {
  const { service } = await fixture(t, success, { ...encryption, getSelectedStorageBackend: () => 'basic_text' })
  assert.equal(service.status().secureStorageAvailable, false)
  assert.equal((await cloudResult(() => service.connect(input))).error.code, 'storage')
})

test('locked encrypted records remain visible and never return a secret', async t => {
  const { service, file } = await fixture(t)
  const model = await service.connect(input)
  const locked = new CloudGeneratorService(file, { ...encryption, decryptString() { throw new Error('locked') } }, success)
  await locked.initialize()
  assert.equal(locked.status().connections[0].connected, false)
  assert.equal(locked.describeModel(model).status, 'needsRuntime')
  assert.equal(locked.credentialFor(model), undefined)
})

test('failed reconnect leaves the previous credential usable', async t => {
  let fail = false
  const { service } = await fixture(t, url => fail ? Promise.resolve(new Response('invalid api key test-only-secret', { status: 401 })) : success(url))
  const model = await service.connect(input)
  fail = true
  const result = await cloudResult(() => service.connect({ ...input, connectionId: model.connectionId, apiKey: 'new-invalid-key' }))
  assert.equal(result.error.code, 'credentials')
  assert.equal(service.credentialFor(model), input.apiKey)
  assert.equal(JSON.stringify(result).includes('test-only-secret'), false)
})

for (const [status, detail, code] of [[401, 'bad key', 'credentials'], [403, 'forbidden', 'access'], [429, 'insufficient_quota', 'quota'], [429, 'rate limit', 'rate_limit'], [404, 'not found', 'model'], [400, 'unsupported parameter', 'model'], [503, 'service unavailable', 'network']]) {
  test(`provider ${status} ${code} has a safe, actionable error without saving`, async t => {
    const { service, file } = await fixture(t, async () => new Response(`${detail} ${input.apiKey}`, { status }))
    const result = await cloudResult(() => service.connect(input))
    assert.equal(result.error.code, code)
    assert.equal(JSON.stringify(result).includes(input.apiKey), false)
    assert.equal(service.status().connections.length, 0)
    await assert.rejects(readFile(file), { code: 'ENOENT' })
  })
}

test('network failures do not expose request credentials', async t => {
  const { service } = await fixture(t, async () => { throw new Error(input.apiKey) })
  const result = await cloudResult(() => service.discover(input))
  assert.equal(result.error.code, 'network')
  assert.equal(JSON.stringify(result).includes(input.apiKey), false)
})

test('cancel during validation never commits a credential, even if a provider ignores abort', async t => {
  let release
  const { service } = await fixture(t, () => new Promise(resolve => { release = resolve }))
  const pending = cloudResult(() => service.connect({ ...input, requestId: 'cancel-me' }))
  service.cancel('cancel-me')
  release(new Response(JSON.stringify({ choices: [{ message: { content: 'OK' } }] })))
  assert.equal((await pending).ok, false)
  assert.equal(service.status().connections.length, 0)
})

test('known providers cannot redirect keys; saved credentials are bound to provider and endpoint', async t => {
  const calls = []
  const { service } = await fixture(t, (url, init) => { calls.push(url); return success(url, init) })
  const model = await service.connect({ ...input, providerId: 'openai', baseUrl: 'https://untrusted.example/v1' })
  assert.equal(calls[0], 'https://api.openai.com/v1/chat/completions')
  assert.equal(service.credentialFor({ ...model, baseUrl: 'https://untrusted.example/v1' }), undefined)
  assert.equal(service.credentialFor({ ...model, role: 'embedder' }), input.apiKey)
  assert.equal((await cloudResult(() => service.connect({ ...input, connectionId: model.connectionId }))).error.code, 'configuration')
  for (const baseUrl of ['http://public.example/v1', 'https://user:pass@example.com/v1', 'https://example.com/v1?key=secret']) {
    assert.equal((await cloudResult(() => service.discover({ ...input, baseUrl }))).error.code, 'configuration')
  }
})

test('discovery excludes non-chat capabilities and keeps manual-entry fallback possible', () => {
  assert.deepEqual(cloudChatModelIds({ data: [{ id: 'models/gemini-2.5-flash' }, { id: 'models/gemini-2.5-flash' }, { id: 'text-embedding-004' }, { id: 'gemini-live-audio' }, { id: 'blocked', capabilities: { completion_chat: false } }] }, 'gemini'), ['gemini-2.5-flash'])
  assert.deepEqual(cloudChatModelIds({ data: [] }, 'custom'), [])
  assert.deepEqual(cloudChatModelIds({ data: ['gemini-3.1-flash-lite', 'gemma-4-31b-it', 'aqa', 'veo-3.1-generate-preview', 'nano-banana-pro-preview', 'lyria-3.5', 'deep-research-pro-preview', 'gemini-3.1-flash-live-preview', 'gemini-robotics-er-2-preview'].map(id => ({ id })) }, 'gemini'), ['gemini-3.1-flash-lite', 'gemma-4-31b-it'])
})

test('discovery returns embedding models separately from chat models', () => {
  const catalog = { data: [{ id: 'chat-test' }, { id: 'text-embedding-3-small', capabilities: { completion_chat: false } }, { id: 'text-embedding-3-large' }] }
  assert.deepEqual(cloudModelIds(catalog, 'openai', 'generator'), ['chat-test'])
  assert.deepEqual(cloudModelIds(catalog, 'openai', 'embedder'), ['text-embedding-3-large', 'text-embedding-3-small'])
})

test('transport errors explain timeout and certificate failures without disclosing the error body', async t => {
  for (const [code, expected] of [['UND_ERR_CONNECT_TIMEOUT', /did not respond in time/], ['UNABLE_TO_VERIFY_LEAF_SIGNATURE', /could not verify/], ['CERT_HAS_EXPIRED', /could not verify/]]) {
    const { service } = await fixture(t, async () => { throw Object.assign(new Error(input.apiKey), { cause: { code } }) })
    const result = await cloudResult(() => service.discover(input))
    assert.equal(result.error.code, 'network')
    assert.match(result.error.message, expected)
    assert.equal(JSON.stringify(result).includes(input.apiKey), false)
  }
})

test('an unreadable catalog and unavailable model offer different recovery from network failure', async t => {
  const { service: invalid } = await fixture(t, async () => new Response('<html>proxy error</html>', { status: 200 }))
  assert.match((await cloudResult(() => invalid.discover(input))).error.message, /unreadable model list/)
  const { service: missing } = await fixture(t, async () => new Response('{}', { status: 404 }))
  assert.match((await cloudResult(() => missing.discover(input))).error.message, /enter a model name manually/)
  assert.match((await cloudResult(() => missing.connect(input))).error.message, /Choose another chat model/)
})

test('runtime credentials hydrate only in main; state serialization is still secret-free', async t => {
  const { service } = await fixture(t)
  const model = await service.connect(input)
  setCloudCredentialResolver(m => service.credentialFor(m))
  t.after(() => setCloudCredentialResolver(() => undefined))
  const hydrated = modelWithRememberedRemoteApiKey(model)
  assert.equal(hydrated.apiKey, input.apiKey)
  assert.equal(model.apiKey, undefined)
  assert.equal(JSON.stringify(sanitizeAppStateSecrets({ models: [hydrated] })).includes(input.apiKey), false)
})

test('validation and chat share provider-compatible output controls', () => {
  assert.deepEqual(remoteChatParameters('https://api.openai.com/v1', 'gpt-5-mini', 512, 0.2, 0.9), { max_completion_tokens: 512 })
  assert.deepEqual(remoteChatParameters('https://api.openai.com/v1', 'gpt-4.1', 512, 0.2, 0.9), { max_completion_tokens: 512, temperature: 0.2, top_p: 0.9 })
  assert.deepEqual(remoteChatParameters('https://generativelanguage.googleapis.com/v1beta/openai', 'gemini-2.5-flash', 512), { max_tokens: 512, temperature: undefined, top_p: undefined })
})
