import { useEffect, useState, type FormEvent } from 'react'
import { ArrowLeft, FileText, FolderOpen, Pause, Plus, RefreshCw, Settings2, Trash2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { ChatSource, CourseMaterial, LocalModel } from '../../shared/app-state'
import { automaticPreparation, type IndexMaterialOptions, type PreparationReport, type PreparationSettings } from '../../shared/preparation'
import './library-workspace.css'

interface Props {
  createRequest: number
  materials: CourseMaterial[]
  models: LocalModel[]
  selectedModelId: string
  selectedEmbeddingModelId: string
  onAddMaterials: (items: CourseMaterial[]) => void
  onStartMaterialIndexing: (id: string, path: string, model?: LocalModel, options?: IndexMaterialOptions) => void
  onRemoveMaterial: (id: string) => void
  onResumeMaterialIndexing: (id: string) => void
  onPauseMaterialIndexing: (id: string) => void
  onToggleMaterialActive: (id: string) => void
  onOpenSource: (source: ChatSource) => void
}

const leaf = (path: string) => path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || 'Documents'
const label = (item: CourseMaterial) => item.status === 'ready' ? 'Ready for chat' : item.status === 'paused' ? 'Paused'
  : item.status === 'needsReview' ? 'Needs attention' : item.indexing?.message || 'Preparing'

export function LibraryWorkspace(props: Props) {
  const { materials, models } = props
  const generators = models.filter(m => m.role !== 'embedder' && m.status === 'ready')
  const embedders = models.filter(m => (m.role === 'embedder' || m.role === 'both') && m.status === 'ready')
  const [creating, setCreating] = useState(false)
  const [path, setPath] = useState('')
  const [name, setName] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = materials.find(m => m.id === selectedId)
  const [preparation, setPreparation] = useState<PreparationSettings>(automaticPreparation)
  const [embedderId, setEmbedderId] = useState(props.selectedEmbeddingModelId)
  const [error, setError] = useState('')
  const [report, setReport] = useState<PreparationReport | null>(null)
  const [documentPath, setDocumentPath] = useState('')
  const [chunkPage, setChunkPage] = useState(0)
  const [documentOverride, setDocumentOverride] = useState('')
  const [reportLoading, setReportLoading] = useState(false)
  const modelId = preparation.modelId || props.selectedModelId
  const generator = generators.find(m => m.id === modelId)
  const embedder = embedders.find(m => m.id === embedderId) || embedders[0]
  const document = report?.documents.find(d => d.path === documentPath)

  useEffect(() => { if (props.createRequest) openCreate() }, [props.createRequest])
  useEffect(() => {
    if (!selected?.path || !window.tokensmith) return
    let cancelled = false
    setReportLoading(true)
    window.tokensmith.preparationReport(selected.path, documentPath).then(value => {
      if (!cancelled) setReport(value)
    }).catch(err => { if (!cancelled) setError(String(err)) }).finally(() => { if (!cancelled) setReportLoading(false) })
    return () => { cancelled = true }
  }, [selected?.path, selected?.indexedAt, selected?.status, selected?.indexing?.processedFiles, documentPath])

  function openCreate() {
    setCreating(true); setSelectedId(null); setPath(''); setName(''); setError('')
    setPreparation(automaticPreparation()); setEmbedderId(props.selectedEmbeddingModelId)
  }
  function inspect(item: CourseMaterial) {
    setSelectedId(item.id); setCreating(false); setError(''); setReport(null); setDocumentPath('')
    setPreparation(item.preparation || automaticPreparation()); setEmbedderId(item.embeddingModelId || props.selectedEmbeddingModelId)
  }
  function chooseDocument(value: string) {
    setDocumentPath(value); setChunkPage(0)
    setDocumentOverride(preparation.documentInstructions[value] || '')
  }
  async function browse() {
    try {
      const result = await window.tokensmith?.pickMaterialFolder()
      if (result?.path) { setPath(result.path); if (!name) setName(result.title || leaf(result.path)) }
    } catch (err) { setError(String(err)) }
  }
  function start(next: PreparationSettings = preparation) {
    const sourcePath = creating ? path.trim().replace(/^['"]|['"]$/g, '') : selected?.path
    if (!sourcePath) { setError('Choose a folder or enter a document path.'); return }
    if (!embedder) { setError('Add an embedding model in Models first.'); return }
    if (next.mode === 'ai' && !generator) { setError('Choose an available preparation model, or start it in Models.'); return }
    const configuration = { ...next, modelId: generator?.id, modelName: generator?.name }
    const id = selected?.id || `material-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    const title = selected?.title || name.trim() || leaf(sourcePath)
    if (creating) props.onAddMaterials([{
      id, title, path: sourcePath, detail: 'Preparing automatically', status: 'indexing', kind: 'folder',
      addedAt: new Date().toISOString(), isActive: false, preparation: configuration
    }])
    props.onStartMaterialIndexing(id, sourcePath, embedder, { title, preparation: configuration, preparationModel: generator })
    setCreating(false); setError('')
  }
  function settings() {
    return <details className="library-preparation-settings">
      <summary><Settings2 size={15} /> Preparation: {preparation.mode === 'ai' ? 'AI automatic' : 'Basic'} <span>Customize</span></summary>
      <p>Each document is prepared automatically. Instructions are optional and apply to every document without an override.</p>
      <label>Preparation method<select value={preparation.mode} onChange={e => setPreparation({ ...preparation, mode: e.target.value as PreparationSettings['mode'] })}>
        <option value="ai">AI automatic</option><option value="basic">Basic splitting (without AI)</option>
      </select></label>
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
      <label>Search model<select value={embedder?.id || ''} onChange={e => setEmbedderId(e.target.value)}>
        {!embedder && <option value="">Choose an embedding model</option>}
        {embedders.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
      </select></label>
      {selected && <button className="primary-action" disabled={selected.status === 'indexing'} onClick={() => start()}>Save and prepare automatically</button>}
    </details>
  }

  return <div className="view-frame standard-frame library-workspace">
    <header className="screen-header"><div>
      {(creating || selected) && <button className="library-back" onClick={() => { setCreating(false); setSelectedId(null); setError('') }}><ArrowLeft size={16} /> Library</button>}
      <h1>{creating ? 'Add documents' : selected ? selected.title : 'Document library'}</h1>
      <p>{creating ? 'Choose a folder of PDFs, Markdown, or text. Preparation runs in the background.' : selected ? label(selected) : 'Add a collection and let AI prepare its documents for chat.'}</p>
    </div>{!creating && !selected && <button className="primary-action" onClick={openCreate}><Plus size={16} /> Add documents</button>}</header>
    {error && <p className="inline-error" role="alert">{error}</p>}
    {creating ? <form className="library-import" onSubmit={(event: FormEvent) => { event.preventDefault(); start() }}>
      <label>Folder or document<div className="library-path-field"><input value={path} onChange={e => setPath(e.target.value)} placeholder="Folder or file path" /><button type="button" onClick={browse}><FolderOpen size={16} /> Browse folder</button></div></label>
      <label>Collection name <span className="library-hint">Optional</span><input value={name} onChange={e => setName(e.target.value)} placeholder={leaf(path)} /></label>
      {settings()}
      <p className="library-hint">{generator ? `Prepared with ${generator.name}.` : 'Choose a preparation model under Customize.'} Successful documents become available to chat automatically.</p>
      <button type="submit" className="primary-action">Add and prepare</button>
    </form> : selected ? <>
      {settings()}
      {selected.indexing && <p role="status">{selected.indexing.message} · {Math.round(selected.indexing.percent)}%</p>}
      {selected.error && <p className="inline-error">{selected.error}</p>}
      <div className="library-document-workspace">
        <aside aria-label="Documents"><h2>Documents</h2>
          {report?.documents.map(d => <button key={d.path} className={d.path === documentPath ? 'selected' : ''} onClick={() => chooseDocument(d.path)}>
            <FileText size={16} /><span>{d.title}<small>{d.status === 'ready' ? `${d.chunkCount} ${d.chunkCount === 1 ? 'chunk' : 'chunks'}` : 'Needs attention'}{d.warning ? ' · Check extraction' : ''}</small></span>
          </button>)}
          {!report?.documents.length && <p className="library-hint">{reportLoading ? 'Loading documents…' : 'Document details appear as preparation finishes. Older collections can be prepared with AI using Customize.'}</p>}
        </aside>
        <section className="library-document-reader" aria-label="Prepared document">
          {!document ? <div className="library-reader-empty"><FileText size={24} /><p>Select a document to inspect its passages or add specific instructions.</p><small>No review is required to use successfully prepared documents.</small></div> : <>
            <header><h2>{document.title}</h2><p>{document.pageCount ? `${document.pageCount} pages · ` : ''}{document.chunkCount} {document.chunkCount === 1 ? 'chunk' : 'chunks'}</p></header>
            {document.error && <p className="inline-error">{document.error}</p>}
            {document.warning && <p className="library-notice">{document.warning}</p>}
            <details className="library-document-instructions"><summary>{Object.hasOwn(preparation.documentInstructions, document.path) ? 'Document instructions' : 'Using collection instructions'} · Customize</summary>
              <label>Instructions for this document<textarea rows={3} value={documentOverride} onChange={e => setDocumentOverride(e.target.value)} placeholder="Leave blank to use the collection’s instructions." /></label>
              <button disabled={selected.status === 'indexing'} onClick={() => {
                const overrides = { ...preparation.documentInstructions }
                if (documentOverride.trim()) overrides[document.path] = documentOverride.trim(); else delete overrides[document.path]
                const next = { ...preparation, documentInstructions: overrides }; setPreparation(next); start(next)
              }}>Save and prepare</button>
            </details>
            {reportLoading ? <p role="status">Loading passages…</p> : <>
              <div className="library-chunk-navigation"><span>Passages {document.chunks.length ? chunkPage * 10 + 1 : 0}–{Math.min((chunkPage + 1) * 10, document.chunks.length)} of {document.chunks.length}</span>
                <button disabled={chunkPage === 0} onClick={() => setChunkPage(v => v - 1)}>Previous</button><button disabled={(chunkPage + 1) * 10 >= document.chunks.length} onClick={() => setChunkPage(v => v + 1)}>Next</button></div>
              {document.chunks.slice(chunkPage * 10, (chunkPage + 1) * 10).map((chunk, i) => <article className="library-source-chunk" key={chunkPage * 10 + i}>
                <header><strong>{chunk.sectionHeader || `Passage ${chunkPage * 10 + i + 1}`}</strong><button onClick={() => props.onOpenSource({
                  title: document.title, path: document.path, materialId: selected.id, documentTitle: document.title,
                  excerpt: chunk.text, locator: chunk.pageStart ? `Page ${chunk.pageStart}` : `Lines ${chunk.lineFrom}–${chunk.lineTo}`,
                  pageStart: chunk.pageStart, pageEnd: chunk.pageEnd, lineFrom: chunk.lineFrom, lineTo: chunk.lineTo
                })}>{chunk.pageStart ? `Page ${chunk.pageStart}${chunk.pageEnd !== chunk.pageStart ? `–${chunk.pageEnd}` : ''}` : `Lines ${chunk.lineFrom}–${chunk.lineTo}`} ↗</button></header>
                <small>{chunk.tokensmithChunkKind}{(chunk.parts || 0) > 1 ? ` · Part ${chunk.part} of ${chunk.parts}` : ''}</small>
                <div className="library-source-text"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={{ img: () => null, a: ({ children }) => <span>{children}</span> }}>{chunk.text}</ReactMarkdown></div>
                {chunk.reason && <details><summary>Why this boundary?</summary><p>{chunk.reason}</p></details>}
              </article>)}
            </>}
          </>}
        </section>
      </div>
    </> : <div className="library-collections">
      {!materials.length && <div className="library-reader-empty"><FolderOpen size={30} /><h2>Your documents, ready for chat</h2><p>Add a folder. AI handles each document’s structure automatically.</p></div>}
      {materials.map(item => <article key={item.id} className="library-collection-row">
        <div><button className="library-title" onClick={() => inspect(item)}>{item.title}</button><p>{item.fileCount || 0} documents · {item.chunkCount || 0} searchable chunks · {item.preparation?.mode === 'ai' ? 'AI automatic' : item.preparation?.mode === 'basic' ? 'Basic preparation' : 'Previous preparation'}</p>
          {item.preparationIssueCount ? <button className="library-issue" onClick={() => inspect(item)}>{item.preparationIssueCount} items need attention</button> : null}
          {item.error && <p className="inline-error">{item.error}</p>}
        </div>
        <div className="library-collection-actions"><span className={`library-state ${item.status}`}>{label(item)}{item.status === 'indexing' ? ` · ${Math.round(item.indexing?.percent || 0)}%` : ''}</span>
          <div><button onClick={() => inspect(item)}>Details</button>
            {item.status === 'indexing' ? <button onClick={() => props.onPauseMaterialIndexing(item.id)}><Pause size={14} /> Pause</button> : <button onClick={() => props.onResumeMaterialIndexing(item.id)}><RefreshCw size={14} /> {item.status === 'ready' ? 'Update' : 'Resume'}</button>}
            <button aria-label={`Remove ${item.title}`} onClick={() => props.onRemoveMaterial(item.id)}><Trash2 size={14} /></button>
          </div><label><input type="checkbox" checked={item.isActive !== false} disabled={item.status !== 'ready' && !item.indexedAt} onChange={() => props.onToggleMaterialActive(item.id)} /> Use in chat</label>
        </div>
      </article>)}
    </div>}
  </div>
}
