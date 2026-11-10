import { useEffect, useState, type ComponentProps } from 'react'
import { ArrowLeft, FileText, FolderOpen, Library, Plus, X } from 'lucide-react'
import type { Collection } from '@shared/collections'
import type { CourseMaterial } from '@shared/app-state'
import { LibraryWorkspace } from './LibraryWorkspace'
import { addLibraryDocuments, libraryError, librarySearchModel } from './library-documents'
import './library-workspace.css'

type Props = ComponentProps<typeof LibraryWorkspace> & {
  collections: Collection[]
  onCollectionsChange: (collections: Collection[]) => Promise<void>
  onUnloadCollection: (id: string) => Promise<void>
}

export function CollectionLibrary({ collections, onCollectionsChange, onUnloadCollection, ...props }: Props) {
  const [selectedId, setSelectedId] = useState<string>()
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [documents, setDocuments] = useState<CourseMaterial[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const selected = collections.find(c => c.id === selectedId && c.loaded)
  useEffect(() => { if (props.createRequest) { setCreating(true); setSelectedId(undefined) } }, [props.createRequest])
  async function perform(work: () => Promise<void>) {
    if (busy) return
    setBusy(true); setError('')
    try { await work() } catch (reason) { setError(libraryError(reason)) }
    finally { setBusy(false) }
  }
  async function open(path?: string) {
    const chosen = path ?? (await window.tokensmith!.pickLibrary()).path
    if (!chosen) return
    const next = await window.tokensmith!.loadCollection(chosen)
    await onCollectionsChange(next)
    setCreating(false)
  }
  const count = (id: string) => {
    const libraryMaterials = props.materials.filter(m => m.collectionId === id)
    return libraryMaterials.reduce((n, m) => n + Math.max(props.documents?.filter(d => d.materialId === m.id).length ?? 0, m.fileCount ?? (m.kind === 'document' ? 1 : 0)), 0)
  }
  return <div className="collection-library">
    <header className="collection-header">
      <div>{(selected || creating) && <button className="library-back" disabled={busy} onClick={() => { setSelectedId(undefined); setCreating(false); setError('') }}><ArrowLeft size={16} />All libraries</button>}
        <h1>{creating ? 'New library' : selected?.name ?? 'Libraries'}</h1></div>
      <div className="collection-header-actions">
        {selected ? <button className="secondary-action" disabled={busy} onClick={() => void perform(async () => { await onUnloadCollection(selected.id); setSelectedId(undefined) })}>Close library</button> : !creating && <>
          <button className="secondary-action" disabled={busy} onClick={() => void perform(() => open())}><FolderOpen size={16} />Open library</button>
          <button className="primary-action" disabled={busy} onClick={() => { setCreating(true); setName(''); setDocuments([]); setError('') }}><Plus size={16} />New library</button>
        </>}
      </div>
    </header>
    {error && <p className="inline-error" role="alert">{error}</p>}
    {creating ? <form className="collection-create" onSubmit={event => { event.preventDefault(); void perform(async () => {
      librarySearchModel(props)
      const next = await window.tokensmith!.createLibrary(name)
      const created = next[0]
      await onCollectionsChange(next)
      setCreating(false); setSelectedId(created.id)
      addLibraryDocuments(documents, { ...props, materials: [], documents: [], collection: created })
      setName(''); setDocuments([])
    }) }}>
      <label>Name<input value={name} maxLength={200} required disabled={busy} autoFocus placeholder="e.g. Database Systems" onChange={event => setName(event.target.value)} /></label>
      <div className="library-file-selection"><h2>Documents</h2><p className="library-hint">PDF, Markdown, or text</p>
        {documents.length > 0 && <ul>{documents.map(document => <li key={document.path}><FileText size={17} /><span>{document.path?.split(/[\\/]/).pop() || document.title}</span>
          <button type="button" className="icon-button subtle" aria-label={`Remove ${document.title}`} disabled={busy} onClick={() => setDocuments(items => items.filter(item => item.path !== document.path))}><X size={16} /></button></li>)}</ul>}
        <button type="button" className="secondary-action" disabled={busy} onClick={() => void perform(async () => {
          const result = await window.tokensmith!.pickMaterials()
          if (!result.canceled) setDocuments(current => [...new Map([...current, ...result.materials].map(d => [d.path, d])).values()])
        })}><Plus size={16} />Add documents</button>
      </div>
      <div className="collection-header-actions"><button className="primary-action" disabled={busy || !name.trim() || !documents.length}>{busy ? 'Creating…' : 'Create library'}</button>
        <button type="button" className="secondary-action" disabled={busy} onClick={() => setCreating(false)}>Cancel</button></div>
    </form> : selected ? <LibraryWorkspace {...props} createRequest={0} onLibraryRefresh={async () => onCollectionsChange(await window.tokensmith!.listCollections())}
      collection={selected} documents={props.documents?.filter(d => d.collectionId === selected.id)} materials={props.materials.filter(m => m.collectionId === selected.id)} /> : <div className="collection-list">
      {!collections.length && <div className="library-reader-empty"><Library size={30} /><h2>No libraries yet</h2><p>Create a library with your documents, or open an existing library.</p></div>}
      {collections.map(collection => <article className="library-collection-row" key={collection.id}>
        <div><button className="library-title" disabled={!collection.loaded} onClick={() => { setSelectedId(collection.id); setError('') }}>{collection.name}</button>
          <p>{collection.loaded ? `${count(collection.id)} document${count(collection.id) === 1 ? '' : 's'}` : 'Closed · Documents are saved'}</p>
          {collection.error && <p className="inline-error">{libraryError(collection.error)}</p>}</div>
        <div className="collection-header-actions">{collection.loaded ? <>
          <button className="secondary-action" disabled={busy} onClick={() => { setSelectedId(collection.id); setError('') }}>View documents</button>
          <button className="secondary-action" disabled={busy} onClick={() => void perform(() => onUnloadCollection(collection.id))}>Close library</button>
        </> : <button className="secondary-action" disabled={busy} onClick={() => void perform(() => open(collection.path))}>Open library</button>}</div>
      </article>)}
    </div>}
  </div>
}
