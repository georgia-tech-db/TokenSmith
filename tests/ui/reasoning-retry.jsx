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
      latencyTrace: { requestId: 'fixture', totalDurationMs: 5420, spans: [
        { stage: 'Question understanding', startedAt: 1, durationMs: 180, inCount: 1, details: { mode: 'standalone' } },
        { stage: 'Library search', startedAt: 2, durationMs: 700, outCount: 1, details: { mode: 'hybrid' }, children: [
          { stage: 'Query encoding', startedAt: 2, durationMs: 60 },
          { stage: 'Semantic search', startedAt: 2, durationMs: 320, outCount: 8 },
          { stage: 'Keyword search', startedAt: 2, durationMs: 210, outCount: 6 },
          { stage: 'Passage ranking', startedAt: 2, durationMs: 110, inCount: 14, outCount: 1 }
        ] },
        { stage: 'Context packing', startedAt: 3, durationMs: 40, inCount: 4, outCount: 1, details: { context_tokens: 8192 } },
        { stage: 'Answer generation', startedAt: 4, durationMs: 4500, details: { prompt_tokens: 420, generated_tokens: 95 }, children: [
          { stage: 'Context check', startedAt: 4, durationMs: 280, inCount: 1, outCount: 1, details: { fits: true, prompt_tokens: 420 } },
          { stage: 'Model loading', startedAt: 4, durationMs: 220 },
          { stage: 'Reading context', startedAt: 4, durationMs: 600 },
          { stage: 'Writing answer', startedAt: 4, durationMs: 3400 }
        ] },
        { stage: 'Follow-up suggestions', startedAt: 5, durationMs: 0, status: 'error' }] },
      reasoning: { modelId: model.id, supported: !params.has('unsupported'), used: params.has('used') },
      answerContext: { prompt: question, conversationContextMode: 'standalone', modelSettings: { contextLength: 8192, maxLength: 1536, reasoningMode: 'auto' } },
      followUpSuggestions: ['How does the parent change?'] }
  ] }],
  models: [model], selectedModelId: model.id, selectedEmbeddingModelId: model.id, materials: [],
  settings: { application: { suggestionMode: 'on', showSources: false }, modelSettingsById: { test: { reasoningMode: 'auto', contextLength: 8192, maxLength: 1536 } } }
}
const storageKey = `query-plan-${params.get('run') || 'fixture'}`
if (params.has('plan-test')) {
  const saved = sessionStorage.getItem(storageKey)
  if (saved) state = JSON.parse(saved)
}
window.reasoningTest = { state: () => state }
let attempts = 0
window.tokensmith = {
  getAppVersion: async () => 'UI test', loadAppState: async () => state,
  saveAppState: async next => { state = next; if (params.has('plan-test')) sessionStorage.setItem(storageKey, JSON.stringify(next)); return next },
  listEngines: async () => [], listMaterials: async () => [], listStudyDocuments: async () => [],
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
      latencyTrace: { requestId: request.requestId, totalDurationMs: request.answerToSimplify ? 300 : 900,
        spans: [{ stage: 'Answer generation', startedAt: 1, durationMs: request.answerToSimplify ? 300 : 900 }] },
      reasoning: { modelId: model.id, supported: true, used: request.modelSettings.thinking === true }
    }
    onAnswer?.(reply, true)
    await new Promise(resolve => setTimeout(resolve, 400))
    return { ...reply, latencyTrace: { ...reply.latencyTrace, totalDurationMs: reply.latencyTrace.totalDurationMs + 400, spans: [...reply.latencyTrace.spans, { stage: 'Follow-up suggestions', startedAt: 2, durationMs: 400 }] }, followUpSuggestions: ['Why must the right node keep two children?'] }
  }
}
await import('../../src/renderer/src/main.tsx')
