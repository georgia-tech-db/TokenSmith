// Full production App with manually completed requests. No LLM calls or user data.
const storageKey = 'tokensmith-navigation-regression'
const initialState = {
  appVersion: 'UI test', conversations: [], materials: [],
  models: [{id:'test-model',name:'Test model',engine:'ollama',role:'both',status:'ready',ollamaModelName:'ui-test',addedAt:''}],
  selectedModelId:'test-model', selectedEmbeddingModelId:'test-model',
  settings:{application:{suggestionMode:'off',fontSize:'medium'}}
}
let state = JSON.parse(sessionStorage.getItem(storageKey) || 'null') ?? initialState
let callCount = 0
const pending = []
function renderStatus() {
  document.querySelector('#test-status').textContent = `${callCount} calls, ${pending.length} pending`
  document.querySelector('#finish-reply').disabled = pending.length === 0
  document.querySelector('#fail-reply').disabled = pending.length === 0
}
document.querySelector('#finish-reply').onclick = () => {
  const next = pending.shift()
  next?.resolve({text:`Completed answer: ${next.prompt}`,sources:[],followUpSuggestions:[]})
  renderStatus()
}
document.querySelector('#fail-reply').onclick = () => {
  pending.shift()?.reject(new Error('Test generation failure'))
  renderStatus()
}
window.tokensmith = {
  getAppVersion: async () => 'UI test',
  loadAppState: async () => state,
  saveAppState: async next => {state=next;sessionStorage.setItem(storageKey,JSON.stringify(next));return next},
  listEngines: async () => [], listMaterials: async () => [],
  onMaterialIndexProgress: () => () => {}, onOllamaPullProgress: () => () => {},
  getOllamaStatus: async () => ({running:true,models:[]}),
  sendChatMessage: request => new Promise((resolve,reject) => {
    callCount++
    pending.push({resolve,reject,prompt:request.prompt})
    renderStatus()
  })
}
await import('../../src/renderer/src/main.tsx')
