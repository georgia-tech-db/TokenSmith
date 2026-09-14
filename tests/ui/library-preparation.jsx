// Production component with an isolated bridge for interaction checks.
import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { LibraryWorkspace } from '../../src/renderer/src/LibraryWorkspace'
import '../../src/renderer/src/styles.css'
import '../../src/renderer/src/themes.css'

const parameters = new URLSearchParams(location.search)
const models = [
  { id: 'gemma', name: 'Gemma', engine: 'ollama', role: 'generator', status: 'ready' },
  { id: 'nomic', name: 'Nomic', engine: 'ollama', role: 'embedder', status: 'ready' }
].filter(model => parameters.has('no-generator') ? model.role === 'embedder' : true)
const documents = ['Poetry collection', 'VLDB Looking Glass', 'Annual report', 'Query Optimization', 'Calculus Volume 3'].map((title, i) => ({
  path: `/sample/document-${i}.pdf`, title, status: 'ready', chunkCount: 1,
  chunks: [{ text: '# A source passage\n\nOriginal document text remains intact.\n\nEach boundary is chosen by the model.', sectionHeader: 'A source passage', pageStart: i + 1, pageEnd: i + 1, reason: 'A complete section.', tokensmithChunkKind: 'section' }]
}))
const preparation = { mode: 'ai', modelId: 'gemma', instructions: '', documentInstructions: {} }
window.tokensmith = {
  pickMaterialFolder: async () => ({ path: '/sample/documents', title: 'Documents', canceled: false }),
  preparationReport: async (_, selected) => ({ documents: documents.map(d => ({ ...d, chunks: d.path === selected ? d.chunks : [] })) })
}
function Fixture() {
  const [materials, setMaterials] = useState([{ id: '1', title: 'Mixed reference library', status: 'ready', indexedAt: '2026-09-13', path: '/sample/documents', isActive: true, fileCount: 5, chunkCount: 5, preparation }])
  const [notice, setNotice] = useState('')
  function start(id, path, model, options) {
    if (options.preparation.mode === 'basic' && (options.preparationModel || options.preparation.modelId)) {
      throw new Error('Basic must not submit a preparation model')
    }
    setNotice(`Started ${options.preparation.mode} preparation. Model: ${options.preparationModel?.name || 'None'}. Instructions: ${options.preparation.instructions || 'None'}`)
    setMaterials(items => items.map(m => m.id === id ? { ...m, status: 'ready', indexedAt: new Date().toISOString(), preparation: options.preparation, fileCount: 5, chunkCount: 5 } : m))
  }
  return <><div role="status">{notice}</div><LibraryWorkspace createRequest={0} materials={materials} models={models} selectedModelId="gemma" selectedEmbeddingModelId="nomic"
    onAddMaterials={items => setMaterials(old => [...items, ...old])} onStartMaterialIndexing={start}
    onRemoveMaterial={id => setMaterials(items => items.filter(m => m.id !== id))}
    onPauseMaterialIndexing={() => {}} onResumeMaterialIndexing={() => {}}
    onToggleMaterialActive={id => setMaterials(items => items.map(m => m.id === id ? { ...m, isActive: !m.isActive } : m))}
    onOpenSource={source => setNotice(`Open ${source.path} at page ${source.pageStart}`)} /></>
}
document.documentElement.dataset.theme = parameters.get('theme') || 'light'
createRoot(document.getElementById('root')).render(<Fixture />)
