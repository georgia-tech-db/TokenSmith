// Isolated UI fixture. Model replies are deterministic; no user data or model calls.
const params = new URLSearchParams(location.search)
const storageKey = `practice-ui-test:${params.get('run') || 'default'}`
const model = { id: 'test', name: 'Practice test model', engine: 'ollama', role: 'both', status: 'ready', ollamaModelName: 'ui-test', addedAt: '', contextLength: 8192 }
const materials = [
  { id: 'c1', title: 'Database Systems', status: 'ready', kind: 'folder', isActive: true, path: '/fixture/db', indexedAt: '2026-10-06' },
  { id: 'c2', title: 'Poems', status: 'ready', kind: 'folder', isActive: false, path: '/fixture/poems', indexedAt: '2026-10-06' }
]
const documents = [
  { materialId: 'c1', documentId: 1, title: 'Trees.md', collectionName: 'Database Systems', path: '/fixture/db/Trees.md', chunkCount: 12 },
  { materialId: 'c1', documentId: 2, title: 'Buffer pool.txt', collectionName: 'Database Systems', path: '/fixture/db/Buffer pool.txt', chunkCount: 9 },
  { materialId: 'c2', documentId: 3, title: 'Poems.md', collectionName: 'Poems', path: '/fixture/poems/Poems.md', chunkCount: 6 }
]
let state = JSON.parse(sessionStorage.getItem(storageKey) || 'null') || {
  appVersion: 'UI test', activeScreen: 'chat', activeConversationId: 'chat-test',
  conversations: [{ id: 'chat-test', title: 'New chat', period: 'Today', messages: [] }],
  models: [model], selectedModelId: model.id, selectedEmbeddingModelId: model.id, materials,
  settings: { application: { suggestionMode: params.has('suggestions') ? 'on' : 'off', showSources: false }, modelSettingsById: { test: { reasoningMode: 'off', contextLength: 8192, maxLength: 1536 } } }
}
const calls = []
let questionCalls = 0
const sourceFor = (document, index) => ({
  title: document.title, documentTitle: document.title, materialId: document.materialId, documentId: document.documentId,
  path: document.path, chunkId: `chunk.${index}`, sectionHeader: 'Ancestor paths', locator: 'Ancestor paths',
  excerpt: 'The ancestor path allows a split to propagate to parent nodes.',
  context: 'The ancestor path allows a split to propagate to parent nodes.', lineFrom: 1, lineTo: 4
})
window.practiceTest = { calls, state: () => state }
window.tokensmith = {
  getAppVersion: async () => params.has('new-version') ? 'UI test next' : 'UI test',
  loadAppState: async () => state,
  saveAppState: async next => { state = next; sessionStorage.setItem(storageKey, JSON.stringify(next)); return next },
  listEngines: async () => [], listMaterials: async () => state.materials,
  onMaterialIndexProgress: () => () => {}, onOllamaPullProgress: () => () => {},
  cancelChatRequest: async id => { calls.push({ kind: 'cancel', id }); return true },
  studyDocuments: async () => {
    if (params.has('catalog-error')) throw new Error('The document catalog is temporarily unavailable.')
    return documents
  },
  searchLibrary: async (query, materials, limit, models, mode, options) => {
    calls.push({ kind: 'search', query, mode, options })
    return {
      sources: options.documents.map(ref => sourceFor(documents.find(d => d.materialId === ref.materialId && d.documentId === ref.documentId), 0))
    }
  },
  cancelChatQuestionSuggestions: async id => { calls.push({ kind: 'cancel-suggestions', id }) },
  suggestChatQuestions: async (id, request) => {
    calls.push({ kind: 'suggestions', request })
    return { suggestions: ['Why record ancestors during insertion?'], sources: [] }
  },
  practiceSources: async (selected, used, index) => {
    calls.push({ kind: 'sources', selected, used, index })
    if (params.has('slow-sources')) await new Promise(resolve => setTimeout(resolve, 2000))
    return [sourceFor(selected[index % selected.length], index)]
  },
  resolveChatQuestion: async () => ({ mode: 'contextual', query: 'Explain ancestor paths during insertion.', clarification: '', reasoning: false }),
  sendChatMessage: async request => {
    const kind = request.practiceTask || 'discussion'
    calls.push({ kind, request })
    if (kind === 'question') questionCalls++
    await new Promise(resolve => setTimeout(resolve, params.has('slow-model') ? 2000 : 100))
    if (kind === 'question' && questionCalls === 2 && params.has('error-next')) throw new Error('Test model disconnected. Try preparing the next question again.')
    if (kind === 'feedback' && params.has('error-feedback') && calls.filter(call => call.kind === 'feedback').length === 1) throw new Error('Test grading failed. Your answer is saved.')
    const source = request.retrievedSources?.[0]
    const evidence = source && [{ sourceKey: `${source.materialId}:${source.documentId}:${source.chunkId}`, quote: source.context }]
    const data = kind === 'feedback' ? JSON.parse(request.prompt) : null
    const complete = data?.answer.includes('split')
    const text = kind === 'question' ? JSON.stringify({
      question: `Why does insertion keep track of ancestor nodes${questionCalls > 1 ? ` (question ${questionCalls})` : ''}?`,
      objective: 'Explain how a split propagates through ancestors.', assumptions: [],
      criteria: [{ id: 'c1', description: 'The path enables backtracking.', evidence },
        { id: 'c2', description: 'A split may need to update the parent.', evidence }],
      hint: 'Think about which node must change after a leaf splits.',
      explanation: 'Record each internal node while descending. If a leaf splits, use this path to update its parent and continue upward if necessary.'
    }) : kind === 'feedback' ? JSON.stringify({
      questionAssessment: 'The question is supported and answerable.',
      checks: [{ criterionId: 'c1', status: 'met', feedback: 'You identified that the path lets insertion backtrack.' },
        { criterionId: 'c2', status: complete ? 'met' : 'missing', feedback: complete ? 'You connected backtracking to updating a parent after a split.' : 'You have not explained which event makes backtracking necessary.' }],
      improvement: data.previousAttempt ? 'You now connect the path to propagating a split.' : '',
      nextStep: complete ? '' : 'Explain when insertion needs to return to a parent and what changes there.',
      questionIssue: params.has('question-issue') ? 'This question assumes a condition that it did not state.' : ''
    }) : 'The path tells insertion where to propagate a split without searching from the root again.'
    return { text, sources: request.retrievedSources || [] }
  },
  getMarkdownForSource: async source => ({ title: source.title, path: source.path,
    text: '# Ancestor paths\n\nThe ancestor path allows a split to propagate to parent nodes.\n', chunkText: source.excerpt,
    lineFrom: source.lineFrom, lineTo: source.lineTo })
}
await import('../../src/renderer/src/main.tsx')
