// A disposable full-app fixture. No user files, model calls, or saved chats are touched.
const params = new URLSearchParams(location.search)
const count = params.has('long') ? 80 : 3
const questions = ['Why track the path during B+ tree insertion?', 'What happens when the leaf splits?', 'Can you show the path tracking code?']
const answers = [
  'The insertion records the path so it can return to **parent nodes** when a leaf splits.\n\nA split adds a separator to the parent. If that parent also overflows, the algorithm continues upward using the recorded path.',
  'A full leaf splits into two leaves. The parent receives a new separator key and child pointer.\n\n1. Divide the keys between the leaves.\n2. Connect the new leaf to its neighbors.\n3. Update the parent using the recorded path.',
  'Store each internal node before following its child pointer:\n\n```cpp\nstd::vector<std::shared_ptr<Node>> path;\nwhile (!node->isLeaf) {\n    path.push_back(node);\n    node = node->children[index];\n}\n```\n\nThe path is then available for updating ancestors. A raw node pointer is `Node*`, not a pointer to a shared pointer.'
]
const messages = Array.from({ length: count }, (_, index) => [
  { id: `q${index + 1}`, role: 'user', text: questions[index % 3] + (count > 3 ? ` (turn ${index + 1})` : '') },
  { id: `a${index + 1}`, role: 'assistant', text: answers[index % 3], followUpSuggestions: ['Can you give a small example?'] }
]).flat()
let state = {
  appVersion: 'UI test',
  activeScreen: 'chat', activeConversationId: 'ui-test',
  conversations: [{ id: 'ui-test', title: 'UI interaction test', period: 'Today', messages },
    { id: 'other-test', title: 'Other test chat', period: 'Today', messages: [] }],
  models: [{ id: 'test-model', name: 'UI test stub', engine: 'ollama', role: 'both', status: 'ready', ollamaModelName: 'ui-test', addedAt: '' }],
  selectedModelId: 'test-model', selectedEmbeddingModelId: 'test-model', materials: [],
  settings: { application: { suggestionMode: 'off' } }
}
window.tokensmith = {
  getAppVersion: async () => 'UI test',
  loadAppState: async () => state,
  saveAppState: async (next) => { state = next; return next },
  listEngines: async () => [],
  listMaterials: async () => [],
  onMaterialIndexProgress: () => () => {},
  onOllamaPullProgress: () => () => {},
  sendChatMessage: async (request) => {
    await new Promise((resolve) => setTimeout(resolve, 800))
    return { text: `Test reply. Earlier message IDs: ${request.messages.map((message) => message.id).join(', ') || '(none)'}.\n\nQuestion received:\n\n${request.prompt}`, sources: [], followUpSuggestions: [] }
  }
}
await import('../../src/renderer/src/main.tsx')
