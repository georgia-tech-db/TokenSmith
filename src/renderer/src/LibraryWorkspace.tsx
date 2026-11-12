import type { Collection } from '@shared/collections'
import { useEffect, useState } from 'react'
import { ArrowLeft, FileText, Pause, Plus, RefreshCw, Settings2, Trash2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { ChatSource, CourseMaterial, LocalModel } from '../../shared/app-state'
import { defaultPreparation, type IndexMaterialOptions, type PreparationReport, type PreparationSettings } from '../../shared/preparation'
import { collectionDocumentSources } from './source-navigation'
import './library-workspace.css'
import type { StudyDocument } from '@shared/study-scope'
import { addLibraryDocuments, libraryError } from './library-documents'

interface Props {
  collection?: Collection
  onLibraryRefresh?: () => Promise<void>
  createRequest: number
  materials: CourseMaterial[]
  documents?: StudyDocument[]
  documentsLoading?: boolean
  documentsError?: string
  onReloadDocuments?: () => void
  models: LocalModel[]
  selectedModelId: string
  selectedEmbeddingModelId: string
  onAddMaterials: (items: CourseMaterial[]) => void
  onStartMaterialIndexing: (id: string, path: string, model?: LocalModel, options?: IndexMaterialOptions) => void
  onRemoveMaterial: (id: string) => void
  onResumeMaterialIndexing: (id: string) => void
  onPauseMaterialIndexing: (id: string) => void
  onOpenSource: (source: ChatSource, sources: ChatSource[]) => void
}

const leaf = (path: string) => path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || 'Documents'
const label = (item: CourseMaterial) => item.status === 'ready' ? 'Ready for chat and practice' : item.status === 'paused' ? 'Paused'
  : item.status === 'needsReview' ? 'Needs attention' : item.indexing?.message || 'Preparing'

export function LibraryWorkspace(props: Props) {
  const { materials, models } = props
  const generators = models.filter(m => m.role !== 'embedder' && m.status === 'ready')
  const embedders = models.filter(m => (m.role === 'embedder' || m.role === 'both') && m.status === 'ready')
  const [adding, setAdding] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = materials.find(m => m.id === selectedId)
  const [preparation, setPreparation] = useState<PreparationSettings>(defaultPreparation)
  const [embedderId, setEmbedderId] = useState(props.selectedEmbeddingModelId)
  const [error, setError] = useState('')
  const [report, setReport] = useState<PreparationReport | null>(null)
  const [documentPath, setDocumentPath] = useState('')
  const [chunkPage, setChunkPage] = useState(0)
  const [documentOverride, setDocumentOverride] = useState('')
  const [reportLoading, setReportLoading] = useState(false)
  const modelId = preparation.modelId || props.selectedModelId
  const generator = generators.find(m => m.id === modelId)
  const embedder = embedders.find(m => m.id === embedderId)
  const document = report?.documents.find(d => d.path === documentPath)
  const documents = props.documents ?? []
  const pending = materials.filter(item => !documents.some(d => d.materialId === item.id))

  useEffect(() => {
    if (!selected?.path || !window.tokensmith) return
    let cancelled = false
    setReportLoading(true)
    window.tokensmith!.preparationReport(selected.path, documentPath, selected.id).then(value => {
      if (!cancelled) setReport(value)
    }).catch(err => { if (!cancelled) setError(libraryError(err)) }).finally(() => { if (!cancelled) setReportLoading(false) })
    return () => { cancelled = true }
  }, [selected?.id, selected?.path, selected?.indexedAt, selected?.status, selected?.indexing?.processedFiles, documentPath])

  function inspect(item: CourseMaterial, path = '') {
    setSelectedId(item.id); setError(''); setReport(null); setDocumentPath(path); setChunkPage(0)
    setPreparation(item.preparation || defaultPreparation()); setEmbedderId(item.embeddingModelId || props.selectedEmbeddingModelId)
  }
  async function addDocuments() {
    setAdding(true); setError('')
    try {
      const result = await window.tokensmith!.pickMaterials()
      if (!result.canceled && result.materials.length) addLibraryDocuments(result.materials, props)
    } catch (reason) { setError(libraryError(reason)) }
    finally { setAdding(false) }
  }
  function openDocument(document: StudyDocument) {
    const source: ChatSource = { ...document, excerpt: '', documentTitle: document.title,
      locator: /\.pdf$/i.test(document.path) ? 'Page 1' : 'Start of document',
      ...(/\.pdf$/i.test(document.path) ? { pageStart: 1 } : { lineFrom: 1, lineTo: 1 }) }
    props.onOpenSource(source, collectionDocumentSources(source, documents))
  }
  function chooseDocument(value: string) {
    setDocumentPath(value); setChunkPage(0)
    setDocumentOverride(preparation.documentInstructions[value] || '')
  }
  function start(next: PreparationSettings = preparation) {
    const sourcePath = selected?.path
    if (!sourcePath) { setError('Select a document first.'); return }
    if (!embedder) { setError('Reconnect your search model in Models, or choose an available embedding model in Settings below.'); return }
    if ((next.mode === 'ai' || next.plainLanguageTerms) && !generator) { setError('Choose an available preparation model, or start it in Models.'); return }
    const preparationModel = next.mode === 'ai' || next.plainLanguageTerms ? generator : undefined
    const configuration = { ...next, modelId: preparationModel?.id, modelName: preparationModel?.name }
    if (!selected) return
    props.onStartMaterialIndexing(selected.id, sourcePath, embedder, { title: selected.title, preparation: configuration, preparationModel })
    setError('')
  }
  function settings() {
    return <div className="library-preparation">
      <div className="library-preparation-choice">
        <span id="preparation-method-label">Preparation</span>
        <div className="library-preparation-modes" role="group" aria-labelledby="preparation-method-label">
          {(['basic', 'ai'] as const).map(mode => <button key={mode} type="button" aria-pressed={preparation.mode === mode}
            onClick={() => { setPreparation({ ...preparation, mode }); setError('') }}>{mode === 'basic' ? 'Basic' : 'AI'}</button>)}
        </div>
        {preparation.mode === 'ai' && <small>Experimental</small>}
      </div>
      <details className="library-preparation-settings">
        <summary><Settings2 size={15} /> Settings <span>Customize</span></summary>
        {preparation.mode === 'ai' && <>
          <label>Preparation model<select value={modelId} onChange={e => setPreparation({ ...preparation, modelId: e.target.value })}>
            {!generator && <option value={modelId}>Choose an available model</option>}
            {generators.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select></label>
          <p className="library-hint">{generator?.engine === 'remote' ? 'Document text will be processed by the selected cloud provider.' : 'Document text is processed by your local model.'}</p>
          <label>Additional instructions <span className="library-hint">Optional</span><textarea rows={4} value={preparation.instructions}
            placeholder="Describe what should stay together, where to split, or what context to retain…"
            onChange={e => setPreparation({ ...preparation, instructions: e.target.value })} /></label>
        </>}
        <label className="library-preparation-toggle">
          <input type="checkbox" checked={preparation.plainLanguageTerms === true}
            onChange={e => setPreparation({ ...preparation, plainLanguageTerms: e.target.checked })} />
          Plain-language search terms <span className="library-hint">Experimental</span>
        </label>
        <p className="library-hint">Writes one everyday-language sentence for each passage so questions asked in a student's own words can find it. The sentence is used for search only; passages are never changed. Preparing takes longer and needs a preparation model.</p>
        {preparation.mode === 'basic' && preparation.plainLanguageTerms && <label>Preparation model<select value={modelId} onChange={e => setPreparation({ ...preparation, modelId: e.target.value })}>
          {!generator && <option value={modelId}>Choose an available model</option>}
          {generators.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select></label>}
        <label>Search model<select value={embedder?.id || ''} onChange={e => setEmbedderId(e.target.value)}>
          {!embedder && <option value="">Choose an embedding model</option>}
          {embedders.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select></label>
        {selected && <button className="primary-action" disabled={selected.status === 'indexing'} onClick={() => start()}>Save and prepare</button>}
      </details>
    </div>
  }

  return <div className="view-frame standard-frame library-workspace">
    <header className="screen-header"><div>
      {selected && <button className="library-back" onClick={() => { setSelectedId(null); setError('') }}><ArrowLeft size={16} />All documents</button>}
      <h1>{selected ? (document?.title || (documentPath ? leaf(documentPath) : selected.title)) : 'Documents'}</h1>
      <p>{selected ? label(selected) : props.documentsLoading ? 'Loading documents…' : `${documents.length + pending.length} document${documents.length + pending.length === 1 ? '' : 's'}`}</p>
    </div>{!selected && <button className="primary-action" disabled={adding} onClick={() => void addDocuments()}><Plus size={16} />Add documents</button>}</header>
    {error && <p className="inline-error" role="alert">{error}</p>}
    {props.documentsError && <p className="inline-error" role="alert">{props.documentsError} <button onClick={props.onReloadDocuments}>Retry</button></p>}
    {selected ? <>
      {settings()}
      {props.collection && <button onClick={async () => {
        try {
          const result = await window.tokensmith!.pickMaterialFolder()
          if (result.path) { await window.tokensmith!.locateCollectionFolder(selected.id, result.path); await props.onLibraryRefresh?.(); props.onReloadDocuments?.() }
        } catch (reason) { setError(libraryError(reason)) }
      }}>Locate original documents</button>}
      {selected.indexing && <p role="status">{selected.indexing.message} · {Math.round(selected.indexing.percent)}%</p>}
      {selected.error && <p className="inline-error">{selected.error}</p>}
      <div className="library-document-workspace">
        <aside aria-label="Documents"><h2>Documents</h2>
          {report?.documents.map(d => <button key={d.path} className={d.path === documentPath ? 'selected' : ''} onClick={() => chooseDocument(d.path)}>
            <FileText size={16} /><span>{d.title}<small>{d.status === 'ready' ? `${d.chunkCount} ${d.chunkCount === 1 ? 'chunk' : 'chunks'}` : 'Needs attention'}{d.warning ? ' · Check extraction' : ''}</small></span>
          </button>)}
          {!report?.documents.length && <p className="library-hint">{reportLoading ? 'Loading documents…' : 'Document details appear as preparation finishes. Use Settings to change preparation.'}</p>}
        </aside>
        <section className="library-document-reader" aria-label="Prepared document">
          {!document ? <div className="library-reader-empty"><FileText size={24} /><p>Select a document to inspect its passages or add specific instructions.</p><small>No review is required to use successfully prepared documents.</small></div> : <>
            <header><h2>{document.title}</h2><p>{document.pageCount ? `${document.pageCount} pages · ` : ''}{document.chunkCount} {document.chunkCount === 1 ? 'chunk' : 'chunks'}</p></header>
            {document.error && <p className="inline-error">{document.error}</p>}
            {document.warning && <p className="library-notice">{document.warning}</p>}
            {preparation.mode === 'ai' && <details className="library-document-instructions"><summary>{Object.hasOwn(preparation.documentInstructions, document.path) ? 'Document instructions' : 'Using shared instructions'} · Customize</summary>
              <label>Instructions for this document<textarea rows={3} value={documentOverride} onChange={e => setDocumentOverride(e.target.value)} placeholder="Leave blank to use the shared instructions." /></label>
              <button disabled={selected.status === 'indexing'} onClick={() => {
                const overrides = { ...preparation.documentInstructions }
                if (documentOverride.trim()) overrides[document.path] = documentOverride.trim(); else delete overrides[document.path]
                const next = { ...preparation, documentInstructions: overrides }; setPreparation(next); start(next)
              }}>Save and prepare</button>
            </details>}
            {reportLoading ? <p role="status">Loading passages…</p> : <>
              <div className="library-chunk-navigation"><span>Passages {document.chunks.length ? chunkPage * 10 + 1 : 0}–{Math.min((chunkPage + 1) * 10, document.chunks.length)} of {document.chunks.length}</span>
                <button disabled={chunkPage === 0} onClick={() => setChunkPage(v => v - 1)}>Previous</button><button disabled={(chunkPage + 1) * 10 >= document.chunks.length} onClick={() => setChunkPage(v => v + 1)}>Next</button></div>
              {document.chunks.slice(chunkPage * 10, (chunkPage + 1) * 10).map((chunk, i) => <article className="library-source-chunk" key={chunkPage * 10 + i}>
                <header><strong>{chunk.sectionHeader || `Passage ${chunkPage * 10 + i + 1}`}</strong><button onClick={() => {
                  const source: ChatSource = {
                    title: document.title, path: document.path, materialId: selected.id, documentTitle: document.title,
                    excerpt: chunk.text, locator: chunk.pageStart ? `Page ${chunk.pageStart}` : `Lines ${chunk.lineFrom}–${chunk.lineTo}`,
                    pageStart: chunk.pageStart, pageEnd: chunk.pageEnd, lineFrom: chunk.lineFrom, lineTo: chunk.lineTo
                  }
                  props.onOpenSource(source, collectionDocumentSources(source, documents.length ? documents : report?.documents ?? []))
                }}>{chunk.pageStart ? `Page ${chunk.pageStart}${chunk.pageEnd !== chunk.pageStart ? `–${chunk.pageEnd}` : ''}` : `Lines ${chunk.lineFrom}–${chunk.lineTo}`} ↗</button></header>
                <small>{chunk.tokensmithChunkKind}{(chunk.parts || 0) > 1 ? ` · Part ${chunk.part} of ${chunk.parts}` : ''}</small>
                <div className="library-source-text"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={{ img: () => null, a: ({ children }) => <span>{children}</span> }}>{chunk.text}</ReactMarkdown></div>
                {chunk.reason && <details><summary>Why this boundary?</summary><p>{chunk.reason}</p></details>}
                {chunk.keywords && <details><summary>Plain-language search terms</summary><p>{chunk.keywords}</p></details>}
              </article>)}
            </>}
          </>}
        </section>
      </div>
    </> : <div className="library-collections">
      {!materials.length && !props.documentsLoading && <div className="library-reader-empty"><FileText size={30} /><h2>No documents yet</h2><p>Add documents to start studying.</p></div>}
      {documents.map(document => {
        const item = materials.find(m => m.id === document.materialId)
        if (!item) return null
        return <article key={`${document.materialId}:${document.documentId}`} className="library-collection-row">
          <div><button className="library-title" onClick={() => openDocument(document)}><FileText size={18} /> {document.title}</button><p>{label(item)}</p></div>
          <div className="library-collection-actions"><div>
            <button onClick={() => openDocument(document)}>Read</button>
            <button onClick={() => inspect(item, document.path)}>Details</button>
            {item.status === 'indexing' && <button onClick={() => props.onPauseMaterialIndexing(item.id)}><Pause size={14} />Pause</button>}
            {documents.filter(d => d.materialId === item.id).length === 1 && (item.fileCount ?? 1) <= 1 &&
              <button aria-label={`Remove ${document.title}`} onClick={() => props.onRemoveMaterial(item.id)}><Trash2 size={14} /></button>}
          </div></div>
        </article>
      })}
      {pending.map(item => <article key={item.id} className="library-collection-row">
        <div><span className="library-title">{item.title}</span><p>{label(item)}{item.status === 'indexing' ? ` · ${Math.round(item.indexing?.percent || 0)}%` : ''}</p>
          {item.error && <p className="inline-error">{libraryError(item.error)}</p>}</div>
        <div className="library-collection-actions"><div>
          {item.status === 'indexing' ? <button onClick={() => props.onPauseMaterialIndexing(item.id)}><Pause size={14} />Pause</button> :
            item.status !== 'ready' && <button onClick={() => props.onResumeMaterialIndexing(item.id)}><RefreshCw size={14} />Retry</button>}
          {!props.documentsLoading && !props.documentsError && (item.fileCount ?? 1) <= 1 && <button aria-label={`Remove ${item.title}`} onClick={() => props.onRemoveMaterial(item.id)}><Trash2 size={14} /></button>}
        </div></div>
      </article>)}
    </div>}
  </div>
}
