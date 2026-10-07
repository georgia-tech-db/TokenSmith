// Disposable UI fixture: no real chats, files, models, or network calls.
const params = new URLSearchParams(location.search)
const sources = [{ title: 'Course notes', locator: 'Internal splits', excerpt: 'Move the separator to the parent. Retain every child exactly once.' }]
const question = 'Split keys [10, 20, 30, 40] with children C0 through C4. Show both nodes.'
const model = { id: 'test', name: 'Reasoning UI test', engine: 'ollama', role: 'both', status: 'ready', ollamaModelName: 'ui-test', addedAt: '', contextLength: 8192 }
let state = {
  appVersion: 'UI test', activeScreen: 'chat', activeConversationId: 'retry-test',
  conversations: [{ id: 'retry-test', title: 'Reasoning retry test', period: 'Today', messages: [
    { id: 'q1', role: 'user', text: question },
    { id: 'a1', role: 'assistant', text: 'Promote 30. Left: keys [10, 20], children C0, C1, C2. Right: key [40], child C4.',
      sources, conversationContextMode: 'standalone', responseDurationMs: 5400,
      latencyTrace: { requestId: 'fixture', totalDurationMs: 5200, spans: [
        { stage: 'Retrieval', startedAt: 1, durationMs: 700, outCount: 1, details: { mode: 'hybrid' }, children: [{ stage: 'Vector search', startedAt: 1, durationMs: 300 }] },
        { stage: 'Generation', startedAt: 2, durationMs: 4500, details: { prompt_tokens: 120 } },
        { stage: 'Suggestions', startedAt: 3, durationMs: 0, status: 'error' }] },
      reasoning: { modelId: model.id, supported: !params.has('unsupported'), used: params.has('used') },
      answerContext: { prompt: question, conversationContextMode: 'standalone', modelSettings: { contextLength: 8192, maxLength: 1536, reasoningMode: 'auto' } },
      followUpSuggestions: ['How does the parent change?'] }
  ] }],
  models: [model], selectedModelId: model.id, selectedEmbeddingModelId: model.id, materials: [],
  settings: { application: { suggestionMode: 'on', showSources: false }, modelSettingsById: { test: { reasoningMode: 'auto', contextLength: 8192, maxLength: 1536 } } }
}
let attempts = 0
window.tokensmith = {
  getAppVersion: async () => 'UI test', loadAppState: async () => state,
  saveAppState: async next => { state = next; return next },
  listEngines: async () => [], listMaterials: async () => [],
  onMaterialIndexProgress: () => () => {}, onOllamaPullProgress: () => () => {},
  cancelChatRequest: async () => true,
  sendChatMessage: async (request, onAnswer) => {
    attempts++
    await new Promise(resolve => setTimeout(resolve, params.has('slow') ? 6000 : 600))
    if (params.has('error') && attempts === 1) throw new Error('Test model disconnected. Your original answer is still available.')
    const reply = {
      text: request.answerToSimplify ? 'The middle key goes up, and each child stays with one of the two nodes.'
        : 'Promote **30**.\n\n- **Left:** keys [10, 20]; children C0, C1, C2.\n- **Right:** key [40]; children **C3, C4**.\n\nAll five children remain, exactly once. The right node has one key and therefore needs two children.',
      sources: request.retrievedSources ?? [],
      reasoning: { modelId: model.id, supported: true, used: request.modelSettings.thinking === true }
    }
    onAnswer?.(reply, true)
    await new Promise(resolve => setTimeout(resolve, 400))
    return { ...reply, followUpSuggestions: ['Why must the right node keep two children?'] }
  }
}
await import('../../src/renderer/src/main.tsx')
