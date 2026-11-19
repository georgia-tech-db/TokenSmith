import { useDocumentFolderPreparation } from './DocumentFolderPreparation'
import { useEffect, useState, type ComponentProps } from 'react'
import { ArrowLeft, FolderOpen, Library, Plus, FolderClosed, Trash2, FileText, Download, Upload } from 'lucide-react'
import type { Collection } from '@shared/collections'
import { LibraryWorkspace } from './LibraryWorkspace'
import { addLibraryDocuments, libraryError, pickDocumentFolder } from './library-documents'
import './library-workspace.css'

type Props = ComponentProps<typeof LibraryWorkspace> & {
  collections: Collection[]
  onCollectionsChange: (collections: Collection[]) => Promise<void>
  onDeleteLibrary: (id: string) => Promise<void>
  onUnloadCollection: (id: string) => Promise<void>
}

export function CollectionLibrary({ collections, onCollectionsChange, onUnloadCollection, onDeleteLibrary, ...props }: Props) {
  const folderPreparation = useDocumentFolderPreparation()
  const [selectedId, setSelectedId] = useState<string>()
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const selected = collections.find(c => c.id === selectedId && c.loaded)
  useEffect(() => { if (props.createRequest) void perform(createLibrary) }, [props.createRequest])
  async function perform(work: () => Promise<void>) {
    if (busy) return
    setBusy(true); setError(''); setNotice('')
    try { await work() } catch (reason) { setNotice(''); setError(libraryError(reason)) }
    finally { setBusy(false) }
  }
  async function open(path: string) {
    const next = await window.tokensmith!.loadCollection(path)
    await onCollectionsChange(next)
  }
  async function importLibrary() {
    const next = await window.tokensmith!.importLibrary()
    if (!next) return
    await onCollectionsChange(next)
    setSelectedId(next[0].id); setNotice('Library imported.')
  }
  async function createLibrary() {
    const folders = await pickDocumentFolder(() => folderPreparation.pick(true, collections.map(library => library.name)))
    if (!folders.length) return
    const next = await window.tokensmith!.createLibrary(folders[0].title)
    await onCollectionsChange(next)
    setSelectedId(next[0].id)
    addLibraryDocuments(folders, { ...props, materials: [], documents: [], collection: next[0] })
  }
  async function exportLibrary(id: string) {
    setNotice('Exporting library…')
    const exported = await window.tokensmith!.exportLibrary(id)
    setNotice(exported ? 'Library exported.' : '')
  }
  const preparing = (id: string) => props.materials.some(m => m.collectionId === id && m.status === 'indexing' && m.indexing?.phase !== 'topics')
  const count = (id: string) => {
    const libraryMaterials = props.materials.filter(m => m.collectionId === id)
    return libraryMaterials.reduce((n, m) => n + Math.max(props.documents?.filter(d => d.materialId === m.id).length ?? 0, m.fileCount ?? (m.kind === 'document' ? 1 : 0)), 0)
  }
  return <div className="collection-library">
    <header className="collection-header">
      <div>{(selected) && <button className="library-back" disabled={busy} onClick={() => { setSelectedId(undefined); setError('') }}><ArrowLeft size={16} />All libraries</button>}
        <h1>{selected?.name ?? 'Libraries'}</h1></div>
      <div className="collection-header-actions">
        {selected ? <><button className="secondary-action" disabled={busy || preparing(selected.id)} title={preparing(selected.id) ? 'Available after document indexing finishes. Topic extraction can continue during export.' : 'Export library and saved topic progress'} onClick={() => void perform(() => exportLibrary(selected.id))}><Download size={16} aria-hidden="true" />Export Library</button><button className="secondary-action" disabled={busy} onClick={() => void perform(async () => { await onUnloadCollection(selected.id); setSelectedId(undefined) })}><FolderClosed size={16} aria-hidden="true" />Close library</button><button className="secondary-action" disabled={busy} onClick={() => void perform(() => onDeleteLibrary(selected.id))}><Trash2 size={16} aria-hidden="true" />Delete Library</button></> : <>
          <button className="secondary-action" disabled={busy} onClick={() => void perform(importLibrary)}><Upload size={16} aria-hidden="true" />Import Existing Library</button>
          <button className="primary-action" disabled={busy} onClick={() => void perform(createLibrary)}><Plus size={16} />Create New Library</button>
        </>}
      </div>
    </header>
    {folderPreparation.panel}
    {notice && <p role="status">{notice}</p>}
    {error && <p className="inline-error" role="alert">{error}</p>}
    {selected ? <LibraryWorkspace {...props} createRequest={0} onLibraryRefresh={async () => onCollectionsChange(await window.tokensmith!.listCollections())}
      collection={selected} documents={props.documents?.filter(d => d.collectionId === selected.id)} materials={props.materials.filter(m => m.collectionId === selected.id)} /> : <div className="collection-list">
      {!collections.length && <div className="library-reader-empty"><Library size={30} /><h2>No libraries yet</h2><p>Create a library with your documents, or open an existing library.</p></div>}
      {collections.map(collection => <article className="library-collection-row" key={collection.id}>
        <div><button className="library-title" disabled={!collection.loaded} onClick={() => { setSelectedId(collection.id); setError('') }}>{collection.name}</button>
          <p>{collection.loaded ? `${count(collection.id)} document${count(collection.id) === 1 ? '' : 's'}` : 'Closed · Documents are saved'}</p>
          {collection.error && <p className="inline-error">{libraryError(collection.error)}</p>}</div>
        <div className="collection-header-actions">{collection.loaded ? <>
          <button className="secondary-action" disabled={busy} onClick={() => { setSelectedId(collection.id); setError('') }}><FileText size={16} aria-hidden="true" />View documents</button>
          <button className="secondary-action" disabled={busy || preparing(collection.id)} title={preparing(collection.id) ? 'Available after document indexing finishes. Topic extraction can continue during export.' : 'Export library and saved topic progress'} onClick={() => void perform(() => exportLibrary(collection.id))}><Download size={16} aria-hidden="true" />Export Library</button>
          <button className="secondary-action" disabled={busy} onClick={() => void perform(() => onUnloadCollection(collection.id))}><FolderClosed size={16} aria-hidden="true" />Close library</button>
        </> : <button className="secondary-action" disabled={busy} onClick={() => void perform(() => open(collection.path))}><FolderOpen size={16} aria-hidden="true" />Open library</button>}<button className="secondary-action" disabled={busy} onClick={() => void perform(() => onDeleteLibrary(collection.id))}><Trash2 size={16} aria-hidden="true" />Delete Library</button></div>
      </article>)}
    </div>}
  </div>
}
