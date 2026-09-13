// Full production UI with a fake bridge. No network requests or real credentials.
const params = new URLSearchParams(location.search)
const materials = [{ id: 'book', title: 'BuzzDB Book', status: 'ready', isActive: true, chunkCount: 1574, embeddingModelId: 'nomic', addedAt: '' }]
const source = { title: 'BuzzDB Book', documentTitle: 'BuzzDB Book', locator: 'Chapter 9', excerpt: 'A B+ tree keeps its leaves linked for range scans.', context: 'A B+ tree keeps its leaves linked for range scans.', materialId: 'book' }
let state = {
  appVersion: 'Cloud UI test', activeScreen: 'chat', activeConversationId: 'cloud-ui-test',
  conversations: [{ id: 'cloud-ui-test', title: 'B+ tree range scans', period: 'Today', messages: [
    { id: 'q1', role: 'user', text: 'Why are B+ trees useful for range queries?' },
    { id: 'a1', role: 'assistant', text: 'The leaf pages are linked, so the database can scan a range without searching the tree for every key.', sources: [source] }
  ] }],
  models: params.has('empty') ? [{ id: 'nomic', name: 'Nomic', engine: 'ollama', role: 'embedder', status: 'ready', ollamaModelName: 'nomic-embed-text', addedAt: '' }] : [
    { id: 'gemma', name: 'Gemma', engine: 'ollama', role: 'generator', status: 'ready', ollamaModelName: 'gemma4:e4b', addedAt: '' },
    { id: 'nomic', name: 'Nomic', engine: 'ollama', role: 'embedder', status: 'ready', ollamaModelName: 'nomic-embed-text', addedAt: '' },
    { id: 'expired', name: 'Saved Gemini', engine: 'remote', role: 'generator', status: 'needsRuntime', providerId: 'gemini', remoteModelName: 'gemini-2.5-flash', addedAt: '' }
  ],
  selectedModelId: 'gemma', selectedEmbeddingModelId: 'nomic', materials,
  settings: { application: { suggestionMode: 'off' } }
}
const connections = []
const canceled = new Set()
const error = (code, message) => ({ ok: false, error: { code, message } })
const modelNames = { gemini: ['gemini-2.5-flash', 'gemini-2.5-pro'], openai: ['gpt-4.1-mini', 'gpt-4.1'], groq: ['llama-3.3-70b-versatile'], mistral: ['mistral-small-latest'], custom: ['study-chat', 'study-chat-large'] }
async function pause(input) {
  await new Promise(resolve => setTimeout(resolve, input.apiKey === 'slow-key' ? 4000 : 450))
  if (canceled.has(input.requestId)) throw new Error('Canceled')
}
window.tokensmith = {
  getAppVersion: async () => 'Cloud UI test',
  loadAppState: async () => state,
  saveAppState: async next => { state = next; return next },
  listEngines: async () => [], listMaterials: async () => materials,
  getOllamaStatus: async () => ({ running: true, models: [], baseUrl: 'http://127.0.0.1:11434' }),
  onMaterialIndexProgress: () => () => {}, onOllamaPullProgress: () => () => {},
  getCloudConnections: async () => ({ connections, secureStorageAvailable: !params.has('session') }),
  cancelCloudSetup: async id => { canceled.add(id) },
  discoverCloudModels: async input => {
    await pause(input)
    if (input.apiKey === 'bad-key') return error('credentials', 'This API key was not accepted. Check it or create a new key.')
    if (params.get('error') === 'offline') return error('network', 'Could not reach the service. Check your connection and try again.')
    return { ok: true, value: params.has('manual') ? [] : modelNames[input.providerId] }
  },
  connectCloudGenerator: async input => {
    await pause(input)
    if (params.get('error') === 'quota') return error('quota', 'This account has no API usage available. Check billing or quota with your service, then retry.')
    const connectionId = input.connectionId || crypto.randomUUID()
    if (!connections.some(c => c.id === connectionId)) connections.push({ id: connectionId, providerId: input.providerId, baseUrl: input.baseUrl, connected: true, remembered: input.remember })
    return { ok: true, value: { id: input.modelId || `cloud:${connectionId}:${input.modelName}`, name: input.modelName,
      engine: 'remote', source: 'remote', role: 'generator', status: 'ready', providerId: input.providerId,
      providerName: input.providerId, remoteModelName: input.modelName, baseUrl: input.baseUrl,
      connectionId, cloudCredentialStatus: 'connected', addedAt: '' } }
  }
}
await import('../../src/renderer/src/main.tsx')
