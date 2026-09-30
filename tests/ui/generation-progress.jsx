// Deterministic full-app fixture, isolated from desktop data and real inference.
import { timingStorageKey, recordTiming } from '../../src/shared/generation-estimate'
const params = new URLSearchParams(location.search)
const storageKey = params.has('mixed') ? 'tokensmith-mixed-source-fixture' : params.has('preview') ? 'tokensmith-source-preview-fixture' : 'tokensmith-progress-fixture'
const model = {id:'progress-test',name:'Gemma 26B test',engine:'ollama',role:'both',status:'ready',ollamaModelName:'gemma4:26b',addedAt:''}
const source = {title:'Course book',locator:'Page 1',...(params.has('preview') ? {path: params.get('preview') === 'pdf' ? '/fixture/book.pdf' : '/fixture/book.md',lineFrom:3,lineTo:3} : {}),excerpt:'Transactions are atomic: either all changes commit or none do.'}
const mixedSources = ['notes.md', 'slides.pdf', 'exercises.md', 'appendix.pdf'].map((name,index)=>({
  path:'/fixture/'+name, title:name, documentTitle:name, materialId:'book', excerpt:`Passage ${index+1}: transactions commit all changes together.`,
  locator:name.endsWith('.pdf')?'Page 1':'Line 3', ...(name.endsWith('.pdf')?{pageStart:1,pageEnd:1}:{lineFrom:3,lineTo:3})
}))
const retrievedSources = params.has('mixed') ? mixedSources : [source,{...source,title:'Second source',excerpt:'Durability preserves committed changes.'}]
function markdownDocument(source) {
  const text=params.has('mixed') ? `# ${source.title}\n\n## Overview\n\n${source.excerpt || 'Transactions are atomic.'}\n\n${('A transaction groups related changes into a single unit.\n\n').repeat(15)}## Details\n\nUse a log to recover committed work.\n\n### Example\n\nConsider a bank transfer.\n\n${('The two account updates must succeed together.\n\n').repeat(15)}## Details\n\nA second section with the same heading.\n\n\`\`\`md\n# Not a heading\n\`\`\`\n\nSummary\n-------\n\nReview the transaction guarantees.`
    : '# '+source.title+'\n\n'+source.excerpt+'\n\n## Further reading\n\nCompare atomicity with durability.'
  return {title:source.title,path:source.path,text,chunkText:source.excerpt,lineFrom:source.lineFrom,lineTo:source.lineTo}
}
const materials = params.has('starter') || params.has('preview') ? [{id:'book',title:'Course book',path:'/fixture',fileCount:params.has('mixed')?4:1,kind:'folder',status:'ready',indexedAt:'today',addedAt:'',isActive:true,embeddingModelId:model.id}] : []
let state = JSON.parse(sessionStorage.getItem(storageKey) || 'null') ?? {
  appVersion:'UI test',activeScreen:'chat',activeConversationId:'test-chat',
  conversations:[{id:'test-chat',title:'Progress test',period:'Today',messages:[]}],
  materials,models:[model],selectedModelId:model.id,selectedEmbeddingModelId:model.id,
  settings:{application:{suggestionMode:params.has('preview') ? 'off' : 'on',followUpSuggestionCount:2,fontSize:'normal'},modelDefaults:{contextLength:8192,maxLength:1536,thinking:false}}
}
if (params.has('fast-estimate')) {
  let samples=[]
  for(let i=0;i<6;i++) samples=recordTiming(samples,{kind:'answer',model,settings:{contextLength:8192,maxLength:1536,thinking:false},inputChars:0,count:2,depth:'standard'},1000)
  localStorage.setItem(timingStorageKey,JSON.stringify(samples))
}
let calls=0, cancellations=0
const sourceLoads=[]
function maybeDelaySource(document) {
  return params.has('slow-source') ? new Promise(resolve=>sourceLoads.push(()=>resolve(document))) : Promise.resolve(document)
}
document.querySelector('#source').onclick=()=>sourceLoads.shift()?.()
function testPdf() {
  const contents='BT /F1 18 Tf 50 730 Td (Transactions are atomic: all changes commit or none do.) Tj ET'
  const objects=[ '<</Type /Catalog /Pages 2 0 R>>', '<</Type /Pages /Kids [3 0 R] /Count 1>>', '<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources <</Font <</F1 4 0 R>>>> /Contents 5 0 R>>', '<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>', `<</Length ${contents.length}>>\nstream\n${contents}\nendstream` ]
  let pdf='%PDF-1.4\n', offsets=[0]
  objects.forEach((object,index)=>{offsets.push(pdf.length);pdf+=`${index+1} 0 obj\n${object}\nendobj\n`})
  const xref=pdf.length
  pdf+=`xref\n0 6\n0000000000 65535 f \n${offsets.slice(1).map(offset=>String(offset).padStart(10,'0')+' 00000 n ').join('\n')}\ntrailer <</Size 6 /Root 1 0 R>>\nstartxref\n${xref}\n%%EOF`
  return 'data:application/pdf;base64,'+btoa(pdf)
}
const jobs=[]
const reply=job=>({engineId:'tokensmith',modelName:model.name,text:`Completed answer: ${job.request.prompt || 'Suggested questions'}. An atomic transaction commits every change together.`,sources:job.request.retrievedSources ?? [],followUpSuggestions:[]})
function metrics(){document.querySelector('#metrics').textContent=`${calls} calls · ${jobs.length} pending · ${cancellations} canceled · ${JSON.parse(localStorage.getItem(timingStorageKey)||'[]').length} samples`}
setInterval(metrics,250)
document.querySelector('#answer').onclick=()=>{const job=jobs[0];const deliver=()=>{if(job && !job.delivered && !job.starter){job.delivered=true;job.onAnswer?.(reply(job),true)}};params.has('answer-delay') ? setTimeout(deliver,2000) : deliver()}
document.querySelector('#finish').onclick=()=>{const job=jobs.shift();if(job)job.resolve(job.starter?{suggestions:['What does atomicity guarantee?','How does rollback work?']}:{...reply(job),followUpSuggestions:['How does rollback work?','What happens after a crash?']});metrics()}
document.querySelector('#fail').onclick=()=>{jobs.shift()?.reject(new Error('Test generation failure'));metrics()}
document.querySelector('#reset').onclick=()=>{sessionStorage.removeItem(storageKey);localStorage.removeItem(timingStorageKey);location.reload()}
function cancel(id){const index=jobs.findIndex(job=>job.id===id);if(index>=0){cancellations++;jobs.splice(index,1)[0].reject(new Error('Request aborted'))}metrics()}
window.tokensmith={
  getAppVersion:async()=> 'UI test',loadAppState:async()=>state,
  saveAppState:async next=>{state=next;sessionStorage.setItem(storageKey,JSON.stringify(next));return next},
  listEngines:async()=>[],listMaterials:async()=>state.materials,
  onMaterialIndexProgress:()=>()=>{},onOllamaPullProgress:()=>()=>{},
  preparationReport:async()=>({documents:mixedSources.map(source=>({path:source.path,title:source.title,status:'ready',chunkCount:1,chunks:[{text:source.excerpt,pageStart:source.pageStart,pageEnd:source.pageEnd,lineFrom:source.lineFrom,lineTo:source.lineTo}]}))}),
  getOllamaStatus:async()=>({running:true,models:[]}),
  resolveChatQuestion:async request=>({mode:'standalone',query:request.prompt,clarification:''}),
  searchLibrary:async()=>retrievedSources, starterSources:async()=>[source],
  getMarkdownForSource:async source=>maybeDelaySource(markdownDocument(source)),
  getPdfForSource:async source=>{if(params.has('fail-source'))throw new Error('Test PDF unavailable');return maybeDelaySource({title:source.title,path:source.path,page:1,dataUrl:testPdf()})},
  sendChatMessage:(request,onAnswer)=>new Promise((resolve,reject)=>{calls++;jobs.push({id:request.requestId,request,onAnswer,resolve,reject});metrics()}),
  cancelChatRequest:async id=>cancel(id),
  suggestChatQuestions:(id,request)=>new Promise((resolve,reject)=>{calls++;jobs.push({id,request,resolve,reject,starter:true});metrics()}),
  cancelChatQuestionSuggestions:async id=>cancel(id)
}
await import('../../src/renderer/src/main.tsx')
