// Production component with an isolated bridge for interaction checks.
import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { CollectionLibrary } from '../../src/renderer/src/CollectionLibrary'
import { GenerationStatus } from '../../src/renderer/src/GenerationStatus'
import { LibraryWorkspace } from '../../src/renderer/src/LibraryWorkspace'
import '../../src/renderer/src/styles.css'
import '../../src/renderer/src/themes.css'

const parameters = new URLSearchParams(location.search)
const models = [
  { id: 'gemma', name: 'Gemma', engine: 'ollama', role: 'generator', status: 'ready' },
  { id: 'gemma-26b', name: 'Gemma 4 26B', engine: 'ollama', role: 'generator', status: 'ready' },
  { id: 'nomic', name: 'Nomic', engine: 'ollama', role: 'embedder', status: 'ready' }
].filter(model => parameters.has('no-generator') ? model.role === 'embedder' : true)
const documents = ['Poetry collection', 'VLDB Looking Glass', 'Annual report', 'Query Optimization', 'Calculus Volume 3'].map((title, i) => ({
  path: `/sample/document-${i}.pdf`, title, status: 'ready', chunkCount: 1,
  chunks: [{ text: '# A source passage\n\nOriginal document text remains intact.\n\nEach boundary is chosen by the model.', sectionHeader: 'A source passage', pageStart: i + 1, pageEnd: i + 1, reason: 'A complete section.', tokensmithChunkKind: 'section' }]
}))
window.preparationTest = { calls: [] }
const preparation = { mode: 'ai', modelId: 'gemma', instructions: '', documentInstructions: {} }
window.tokensmith = {
  exportLibrary: async id => { window.preparationTest.calls.push(['export',id]); return true },
  pickMaterials: async () => ({ canceled: true, materials: [] }),
  pickDocumentFolder: async () => ({path:'/sample/new-document.md',title:'New document',canceled:false}),
  pickMaterialFolder: async () => ({ path: '/sample/documents', title: 'Documents', canceled: false }),
  preparationReport: async (_, selected) => ({ documents: documents.map(d => ({ ...d, chunks: d.path === selected ? d.chunks : [] })) })
}
function Fixture() {
  const Workspace=parameters.has('export')?CollectionLibrary:LibraryWorkspace
  const [materials, setMaterials] = useState([{ id: '1', collectionId: 'test', title: 'Mixed reference library', status: parameters.has('lifecycle') ? 'indexing' : 'ready', indexedAt: '2026-09-13', path: '/sample/documents', isActive: true, fileCount: 5, chunkCount: 5, indexing: {...(parameters.has('ocr-stage')?{stage:'OCR',stagePercent:60,estimatedRemainingMs:20000}:{}),percent:82,phase:parameters.has('parsing')?'parsing':'topics',message:'Extracting topics · Database Book · Chapter 5 — Buffer Management · Pages 23–28 · Part 6 of 18 · 58 topics saved · Gemma 4 E4B'}, preparation: parameters.has('unprepared') ? undefined : preparation }])
  const [notice, setNotice] = useState('')
  function start(id, path, model, options) {
    if (options.preparation.mode === 'basic' && options.preparation.extractTopics === false && options.preparation.ocr !== 'gemma' && (options.preparationModel || options.preparation.modelId)) {
      throw new Error('Basic must not submit a preparation model')
    }
    window.preparationTest.started = { modelId: options.preparation.modelId, mode: options.preparation.mode, extractTopics: options.preparation.extractTopics }
    setNotice(`Started ${options.preparation.mode} preparation. Model: ${options.preparationModel?.name || 'None'}. Instructions: ${options.preparation.instructions || 'None'}`)
    setMaterials(items => items.map(m => m.id === id ? { ...m, status: 'ready', indexedAt: new Date().toISOString(), preparation: options.preparation, fileCount: 5, chunkCount: 5 } : m))
  }
  return <><div role="status">{notice}</div><Workspace collections={[{id:'test',name:'Study library',loaded:true}]} onCollectionsChange={async()=>{}} onUnloadCollection={async()=>{}} onDeleteLibrary={async()=>{}} collection={{id:"test",name:"Test",loaded:true}} createRequest={0} documents={documents.map((d, i) => ({ ...d, documentId: i + 1, materialId: '1', collectionName: 'Mixed reference library' }))} materials={materials} models={models} selectedModelId="gemma" selectedEmbeddingModelId="nomic"
    onAddMaterials={items => setMaterials(old => [...items, ...old])} onStartMaterialIndexing={start}
    onRemoveMaterial={id => setMaterials(items => items.filter(m => m.id !== id))}
    onPauseMaterialIndexing={id => { window.preparationTest.calls.push(['pause',id]); setMaterials(items => items.map(m => m.id === id ? {...m,status:'paused'} : m)) }} onResumeMaterialIndexing={id => { window.preparationTest.calls.push(['resume',id]); setMaterials(items => items.map(m => m.id === id ? {...m,status:'indexing'} : m)) }}
    onToggleMaterialActive={id => setMaterials(items => items.map(m => m.id === id ? { ...m, isActive: !m.isActive } : m))}
    onOpenSource={source => setNotice(`Open ${source.path} at page ${source.pageStart}`)} /><GenerationStatus progress={null} preparation={materials.find(m=>m.status==='indexing')}
    onOpen={()=>{}} onOpenPreparation={()=>setNotice('Open preparation')}
    onPausePreparation={()=>setMaterials(items=>items.map(m=>({...m,status:'paused'})))} /></>
}
document.documentElement.dataset.theme = parameters.get('theme') || 'light'
createRoot(document.getElementById('root')).render(<Fixture />)
