// Isolated UI fixture. Model replies are deterministic; no user data or model calls.
const params = new URLSearchParams(location.search)
const storageKey = `practice-ui-test:${params.get('run') || 'default'}`
const model = { id: 'test', name: 'Practice test model', engine: 'ollama', role: 'both', status: 'ready', ollamaModelName: 'ui-test', addedAt: '', contextLength: 8192 }
if (params.has('model-unavailable')) model.status = 'missing'
const cloud = { id: 'cloud', name: 'Cloud test model', engine: 'remote', role: 'generator', status: 'ready',
  providerId: 'custom', providerName: 'Test provider', remoteModelName: 'cloud-test', cloudCredentialStatus: 'connected', addedAt: '', contextLength: 16384 }
const materials = [
  { id: 'c1', title: 'Database Systems', status: 'ready', kind: 'folder', isActive: true, path: '/fixture/db', indexedAt: '2026-10-06' },
  { id: 'c2', title: 'Poems', status: 'ready', kind: 'folder', isActive: false, path: '/fixture/poems', indexedAt: '2026-10-06' }
]
const documents = [
  { materialId: 'c1', documentId: 1, title: 'Trees.md', collectionName: 'Database Systems', path: '/fixture/db/Trees.md', chunkCount: 12 },
  { materialId: 'c1', documentId: 2, title: 'Buffer pool.txt', collectionName: 'Database Systems', path: '/fixture/db/Buffer pool.txt', chunkCount: 9 },
  { materialId: 'c2', documentId: 3, title: 'Poems.md', collectionName: 'Poems', path: '/fixture/poems/Poems.md', chunkCount: 6 }
]
if (params.has('long-catalog')) {
  for (const collection of materials) {
    for (let index = 1; index <= 24; index++) {
      documents.push({ materialId: collection.id, documentId: documents.length + 1,
        title: `Lecture notes ${index}.md`, collectionName: collection.title,
        path: `${collection.path}/Lecture notes ${index}.md`, chunkCount: 8 })
    }
  }
}
if (params.has('collections')) {
  for (const material of materials) {
    const previous = material.id
    material.collectionId = previous === 'c1' ? 'alpha0000' : 'beta0000'
    material.collectionTitle = previous === 'c1' ? 'Database course' : 'Poetry course'
    material.id = `${material.collectionId}/1`
    for (const document of documents.filter(d => d.materialId === previous)) {
      document.materialId = material.id; document.collectionId = material.collectionId; document.collectionTitle = material.collectionTitle
      if (previous === 'c2') document.documentId = 1
    }
  }
}
let state = JSON.parse(sessionStorage.getItem(storageKey) || 'null') || {
  appVersion: 'UI test', activeScreen: 'chat', activeConversationId: 'chat-test',
  conversations: [{ id: 'chat-test', title: 'New chat', period: 'Today', messages: [] }, ...(params.has('model-selection') ? [
    { id: 'a', title: 'Local conversation', period: 'Today', modelId: model.id, messages: [] },
    { id: 'b', title: 'Cloud conversation', period: 'Today', modelId: cloud.id, messages: [] },
    { id: 'missing', title: 'Removed model conversation', period: 'Today', modelId: 'removed', messages: [] },
    { id: 'disconnected', title: 'Disconnected conversation', period: 'Today', modelId: 'disconnected', messages: [] }
  ] : [])],
  practice: { mode: params.has('practice') ? 'practice' : 'chat', sessions: [] },
  models: params.has('model-selection') ? [model, cloud, { ...cloud, id: 'disconnected', remoteModelName: 'cloud-disconnected', cloudCredentialStatus: 'reconnect', status: 'needsRuntime' },
    { id: 'search', name: 'Search only', role: 'embedder', engine: 'ollama', ollamaModelName: 'search-only', status: 'ready' }] : [model],
  selectedModelId: model.id, selectedEmbeddingModelId: model.id, materials,
  settings: { application: { defaultModelId: model.id, suggestionMode: params.has('suggestions') ? 'on' : 'off', showSources: params.has('show-sources'), readWhileWaiting: params.has('preview-sources') },
    modelSettingsById: { test: { reasoningMode: 'off', contextLength: 8192, maxLength: 1536 }, cloud: { contextLength: 16384, maxLength: 2048 } } }
}
const calls = []
const progressListeners = new Set()
let questionCalls = 0
const sourceFor = (document, index) => ({
  title: document.title, documentTitle: document.title, materialId: document.materialId, documentId: document.documentId,
  path: document.path, chunkId: `chunk.${index}`, sectionHeader: 'Ancestor paths', locator: 'Ancestor paths',
  excerpt: 'The ancestor path allows a split to propagate to parent nodes.',
  context: 'The ancestor path allows a split to propagate to parent nodes.', lineFrom: 1, lineTo: 4
})
window.practiceTest = { calls, state: () => state,
  removeLocalModel: () => progressListeners.forEach(listener => listener({ model: 'ui-test', status: 'removed', percent: 0 })),
  disconnectCloud: () => { state.models = state.models.map(model => model.id === 'cloud' ? { ...model, status: 'needsRuntime', cloudCredentialStatus: 'reconnect' } : model); sessionStorage.setItem(storageKey, JSON.stringify(state)) } }
window.tokensmith = {
  getAppVersion: async () => params.has('new-version') ? 'UI test next' : 'UI test',
  loadAppState: async () => state,
  saveAppState: async next => { state = next; sessionStorage.setItem(storageKey, JSON.stringify(next)); return next },
  getLogFile: async () => {
    calls.push({ kind: 'log' })
    if (params.has('log-error') && calls.filter(call => call.kind === 'log').length === 1) throw new Error('Test log read failed.')
    const text = JSON.stringify({ event: 'practice_question', prompt: 'Fixture practice question', text: 'Fixture model reply',
      modelMessages: [{ role: 'user', content: 'Fixture question prompt' }] })
    return { path: '/fixture/logs/tokensmith.log', text, sizeBytes: text.length, maxBytes: 10000, truncated: false }
  },
  listEngines: async () => [], listMaterials: async () => state.materials,
  onMaterialIndexProgress: () => () => {}, onOllamaPullProgress: listener => { progressListeners.add(listener); return () => progressListeners.delete(listener) },
  cancelChatRequest: async id => { calls.push({ kind: 'cancel', id }); return true },
  studyDocuments: async () => {
    if (params.has('catalog-error')) throw new Error('The document catalog is temporarily unavailable.')
    return documents
  },
  searchLibrary: async (query, materials, limit, models, mode, options) => {
    calls.push({ kind: 'search', query, mode, options })
    if (params.has('slow-search')) await new Promise(resolve => setTimeout(resolve, 1500))
    if (params.has('slow-sources')) await new Promise(resolve => setTimeout(resolve, 2000))
    if (params.has('no-passages')) return { sources: [] }
    return { sources: Array.from({ length: params.has('many-sources') ? 6 : 1 }, (_, index) => {
      const ref = options.documents[index % options.documents.length]
      return sourceFor(documents.find(d => d.materialId === ref.materialId && d.documentId === ref.documentId), index)
    }) }
  },
  starterSources: async (_materials, limit, selected) => {
    calls.push({ kind: 'topic-sources', selected, limit })
    if (params.has('no-passages')) return []
    return selected.map((ref, index) => sourceFor(documents.find(d => d.materialId === ref.materialId && d.documentId === ref.documentId), index))
  },
  cancelChatQuestionSuggestions: async id => { calls.push({ kind: 'cancel-suggestions', id }) },
  suggestChatQuestions: async (id, request) => {
    calls.push({ kind: 'suggestions', request })
    if (params.has('slow-suggestions')) await new Promise(resolve => setTimeout(resolve, 2000))
    if (params.has('error-suggestions') && calls.filter(call => call.kind === 'suggestions').length === 1) throw new Error('Suggestions unavailable')
    return { suggestions: ['Why record ancestors during insertion?', 'How can a split reach the root?', 'What changes in the parent when a leaf splits?'], sources: [] }
  },
  practiceSources: async (selected, used, index) => {
    calls.push({ kind: 'sources', selected, used, index })
    if (params.has('slow-sources')) await new Promise(resolve => setTimeout(resolve, 2000))
    if (params.has('no-passages') || (params.has('exhausted') && index >= 1)) return []
    return Array.from({ length: params.has('many-sources') ? 6 : 1 }, (_, i) => sourceFor(selected[index % selected.length], index * 10 + i))
  },
  resolveChatQuestion: async request => { calls.push({ kind: 'rewrite', request }); return { mode: 'contextual', query: 'Explain ancestor paths during insertion.', clarification: '', reasoning: false } },
  sendChatMessage: async request => {
    const kind = request.practiceTask || 'discussion'
    if (kind === 'question' || kind === 'feedback') throw new Error('Short-answer practice is retired.')
    const preparing = kind === 'true_false_question'
    calls.push({ kind, request })
    if (preparing) questionCalls++
    await new Promise(resolve => setTimeout(resolve, params.has('slow-model') ? 2000 : 100))
    if (preparing && questionCalls === 1 && params.has('error-first')) throw new Error('Test first question failed.')
    if (preparing && questionCalls === 2 && params.has('error-next')) throw new Error('Test model disconnected. Try preparing the next question again.')
    if (kind === 'discussion' && params.has('error-help') && calls.filter(call => call.kind === 'discussion').length === 1) throw new Error('Test explanation failed. Your question is saved.')
    if (kind === 'topics') return { text: JSON.stringify({ topics: [
      { title: 'B+ trees', sourceKeys: request.retrievedSources.slice(0, 1).map(source => `${source.materialId}:${source.documentId}:${source.sourceUnitId || source.chunkId}`) },
      ...(request.retrievedSources.length > 1 ? [{ title: 'Buffer pools', sourceKeys: request.retrievedSources.slice(1, 2).map(source => `${source.materialId}:${source.documentId}:${source.sourceUnitId || source.chunkId}`) }] : [])
    ] }), sources: request.retrievedSources }
    const evidence = request.retrievedSources?.map(source => ({ sourceKey: `${source.materialId}:${source.documentId}:${source.sourceUnitId || source.chunkId}`, quote: source.context || source.excerpt })) || []
    const binary = kind === 'true_false_question'
    const questionNumber = preparing ? JSON.parse(request.prompt).questionNumber : 0
    const correctAnswer = preparing && JSON.parse(request.prompt).targetAnswer
    const text = preparing ? JSON.stringify({
      question: binary ? (params.has('duplicate-question') ? true : correctAnswer)
        ? `The ancestor path lets insertion propagate a split to its parent${questionNumber > 2 && !params.has('duplicate-question') ? ` in insertion example ${questionNumber}` : ''}.`
        : `Insertion never uses the ancestor path to propagate a split to its parent${questionNumber > 2 ? ` in insertion example ${questionNumber}` : ''}.`
        : `Why does insertion keep track of ancestor nodes${questionCalls > 1 ? ` (question ${questionCalls})` : ''}?`,
      ...(preparing && (params.has('open-question-always') || (params.has('open-question-first') && questionCalls === 1)) ? { question: 'If a crash interrupts the table update, what state results?' } : {}),
      ...(binary ? { correctAnswer: params.has('bad-true-false-key') ? 'true' : correctAnswer,
        questionAssessment: 'The passage directly supports or contradicts this statement.',
        questionIssue: params.has('ambiguous-true-false') ? 'The statement is ambiguous.' : '' } : {}),
      objective: 'Explain how a split propagates through ancestors.', assumptions: [],
      criteria: [
        { id: 'c1', description: 'The path enables backtracking.', evidence: evidence.slice(0, 2) },
        { id: 'c2', description: 'A split may need to update the parent.', evidence: evidence.length > 2 ? evidence.slice(2, 4) : evidence },
        ...(evidence.length > 4 ? [{ id: 'c3', description: 'Splits may propagate upward.', evidence: evidence.slice(4, 6) }] : [])
      ],
      explanation: Array(params.has('long-explanation') ? 12 : 1).fill('Record each internal node while descending. If a leaf splits, use this path to update its parent and continue upward if necessary.').join('\n\n')
    }) : 'The path tells insertion where to propagate a split without searching from the root again.'
    return { text, sources: request.retrievedSources || [] }
  },
  getMarkdownForSource: async source => {
    calls.push({ kind: 'read-source', source })
    if (params.has('slow-reader')) await new Promise(resolve => setTimeout(resolve, 1500))
    if (params.has('source-error') && calls.filter(call => call.kind === 'read-source').length === 1) throw new Error('Test source read failed.')
    return { title: source.title, path: source.path,
      text: '# Ancestor paths\n\nThe ancestor path allows a split to propagate to parent nodes.\n', chunkText: source.excerpt,
      lineFrom: source.lineFrom, lineTo: source.lineTo }
  }
}
if (params.has('collections')) {
  const collectionKey = `${storageKey}:collections`
  let collections = JSON.parse(sessionStorage.getItem(collectionKey) || 'null') || [
    { format: 'tokensmith-collection', version: 1, id: 'alpha0000', name: 'Database course', path: '/collections/database', indexRevision: 'revision-a', loaded: true },
    { format: 'tokensmith-collection', version: 1, id: 'beta0000', name: 'Poetry course', path: '/collections/poetry', indexRevision: 'revision-b', loaded: false }
  ]
  const savedTopics = params.has('saved-topics') ? [{ title: 'B+ trees' }] : []
  const loaded = id => collections.some(c => c.id === id && c.loaded)
  const persist = () => { sessionStorage.setItem(collectionKey, JSON.stringify(collections)); return structuredClone(collections) }
  window.collectionTest = { get: () => collections, pickPath: '/collections/poetry', savedTopics }
  Object.assign(window.tokensmith, {
    listCollections: async () => structuredClone(collections),
    listMaterials: async () => materials.filter(m => loaded(m.collectionId)),
    studyDocuments: async () => {
      if (params.has('catalog-error')) throw new Error('The document catalog is temporarily unavailable.')
      return documents.filter(d => loaded(d.collectionId))
    },
    pickMaterialFolder: async () => ({ path: window.collectionTest.pickPath, title: 'Chosen folder', canceled: !window.collectionTest.pickPath }),
    pickLibrary: async () => ({ path: window.collectionTest.pickPath, canceled: !window.collectionTest.pickPath }),
    pickMaterials: async () => ({ canceled: false, materials: [
      { id: 'new-note', title: 'New notes', path: '/fixture/new/notes.md', kind: 'document', status: 'indexing' },
      { id: 'new-handout', title: 'Handout', path: '/fixture/new/handout.pdf', kind: 'document', status: 'indexing' }
    ] }),
    createLibrary: async name => {
      const path = `/app/libraries/${crypto.randomUUID()}`
      calls.push({ kind: 'create-collection', path, name })
      collections.unshift({ format: 'tokensmith-collection', version: 1, id: crypto.randomUUID(), name, path, indexRevision: 'new', loaded: true })
      return persist()
    },
    loadCollection: async path => {
      calls.push({ kind: 'load-collection', path })
      const item = collections.find(c => c.path === path)
      if (!item) throw new Error('Choose a saved TokenSmith library. To add documents, use New library.')
      item.loaded = true; return persist()
    },
    unloadCollection: async id => { calls.push({ kind: 'unload-collection', id }); collections.find(c => c.id === id).loaded = false; return persist() },
    collectionTopics: async () => structuredClone(savedTopics),
    saveCollectionTopics: async (_documents, topics) => { calls.push({ kind: 'save-topics' }); savedTopics.splice(0, savedTopics.length, ...topics) },
    indexMaterial: async (id, path, _model, options) => {
      calls.push({ kind: 'index-document', path, options })
      const collection = collections.find(c => id.startsWith(c.id + '/'))
      const item = { id, path, title: options.title, status: 'ready', kind: 'document', fileCount: 1,
        indexedAt: new Date().toISOString(), collectionId: collection.id, collectionTitle: collection.name }
      materials.push(item)
      documents.push({ materialId: id, documentId: documents.length + 1, title: path.split('/').pop(), path, chunkCount: 1,
        collectionId: collection.id, collectionTitle: collection.name, collectionName: collection.name })
      return item
    },
    preparationReport: async () => ({ documents: [] }),
    locateCollectionFolder: async (id, path) => { calls.push({ kind: 'locate-files', id, path }) }
  })
}
await import('../../src/renderer/src/main.tsx')
