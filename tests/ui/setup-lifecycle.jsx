// Production renderer, deterministic OS/network boundary. Never starts Ollama or downloads models.
const params = new URLSearchParams(location.search)
let state = JSON.parse(sessionStorage.getItem('setup-state') || 'null')
let installed = params.has('installed'), running = params.has('running')
let models = JSON.parse(sessionStorage.getItem('setup-models') || '[]')
const calls = [], listeners = new Set(), pending = new Map()
let startError = '', statusError = '', deleteError = '', catalogErrors = params.has('catalog-race') ? 2 : 0, catalogCalls = 0
const emit = progress => listeners.forEach(fn => fn(progress))
const persist = () => sessionStorage.setItem('setup-models', JSON.stringify(models))
window.setupTest = {
  state: () => state, calls, pending: () => [...pending.keys()], catalogCalls: () => catalogCalls,
  configure: options => {
    if ('installed' in options) installed = options.installed
    if ('running' in options) running = options.running
    if ('startError' in options) startError = options.startError
    if ('statusError' in options) statusError = options.statusError
    if ('deleteError' in options) deleteError = options.deleteError
    if ('catalogErrors' in options) catalogErrors = options.catalogErrors
  },
  progress: (model, percent) => emit({model,status:'downloading',percent,total:100,completed:percent,message:'pulling model files'}),
  finish: (model, {error, verified=true}={}) => {
    const job = pending.get(model); if (!job) throw Error(`No download for ${model}`)
    pending.delete(model)
    if (error) { emit({model,status:'error',percent:0,error,message:error}); job.reject(Error(error)); return }
    if (verified) { models.push({name:`${model}:latest`,size:100,details:{contextLength:8192}}); persist() }
    emit({model,status:'complete',percent:100,message:'success'})
    job.resolve({model,status:'success'})
  }
}
window.tokensmith = {
  getAppVersion: async () => 'Setup test', loadAppState: async () => state,
  saveAppState: async next => { state=next;sessionStorage.setItem('setup-state',JSON.stringify(next));return next },
  listEngines: async () => [], listMaterials: async () => [], listCollections: async () => [],
  studyDocuments: async () => { catalogCalls++;if(catalogErrors-->0)throw Error("Error invoking remote method 'library:study-documents': Error: The library changed during this request. Please try again");return [] },
  getCloudConnections: async () => ({connections:[],secureStorageAvailable:true}),
  getDeviceCapabilities: async () => {
    const gb=1024**3, strong=params.has('strong')
    return {platform:'macos',architecture:'arm64',platformRelease:'24.6.0',cpu:{model:'Test',logicalCores:10,availableParallelism:10},memory:{totalBytes:(strong?64:16)*gb},storage:{availableBytes:100*gb},accelerators:[{name:'Apple GPU',backend:'metal',memoryTopology:'unified',totalMemoryBytes:(strong?64:16)*gb,availableMemoryBytes:(strong?64:16)*gb,runtimeSupport:'supported',source:'test'}],detectionWarnings:[]}
  },
  getOllamaStatus: async () => { if(statusError)throw Error(statusError);return {running,installedApp:installed,models,baseUrl:'http://127.0.0.1:11434'} },
  openOllamaDownloadPage: async () => {calls.push({kind:'install'});return {opened:true}},
  openOllamaApp: async () => {calls.push({kind:'open'});running=true;return {opened:true}},
  startOllamaService: async () => {calls.push({kind:'start'});if(startError)return {opened:false,message:startError};running=true;return {opened:true}},
  pullOllamaModel: model => {calls.push({kind:'pull',model});return new Promise((resolve,reject)=>pending.set(model,{resolve,reject}))},
  cancelOllamaPull: async model => {calls.push({kind:'cancel',model});const job=pending.get(model);pending.delete(model);emit({model,status:'incomplete',percent:40,message:'Download paused'});job?.reject(Error('Download paused'))},
  removeModel: async model => {calls.push({kind:'remove',model:model.ollamaModelName});if(deleteError)throw Error(deleteError);models=models.filter(m=>m.name!==`${model.ollamaModelName}:latest`);persist()},
  deleteOllamaModel: async model => {calls.push({kind:'delete',model});pending.get(model)?.reject(Error('Download cancelled'));pending.delete(model);models=models.filter(m=>m.name!==`${model}:latest`);persist();return {model,status:'deleted'}},
  onOllamaPullProgress: fn => {listeners.add(fn);return ()=>listeners.delete(fn)},
  onMaterialIndexProgress: () => () => {}
}
await import('../../src/renderer/src/main.tsx')
