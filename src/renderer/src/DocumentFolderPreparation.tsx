import { FolderOpen, Plus, X } from 'lucide-react'
import { libraryNameKey, uniqueLibraryName } from '@shared/library-name'
import { useEffect, useId, useRef, useState } from 'react'
import { defaultPreparation, type PreparationSettings } from '@shared/preparation'
import type { PickMaterialFolderResult } from '@shared/engine'
import './library-workspace.css'

export type PreparedFolder = PickMaterialFolderResult & { preparation?: PreparationSettings }

/** Folder selection is native; preparation choices stay together in the app. */
export function useDocumentFolderPreparation() {
  const confirmationHintId = useId()
  const [open, setOpen] = useState(false)
  const [existingNames, setExistingNames] = useState<string[]>([])
  const [choosing, setChoosing] = useState(false)
  const [error, setError] = useState('')
  const editedName = useRef(false)
  const [name, setName] = useState('')
  const [creatingLibrary, setCreatingLibrary] = useState(false)
  const [folder, setFolder] = useState<PickMaterialFolderResult>()
  const [settings, setSettings] = useState(defaultPreparation)
  const [modeChosen, setModeChosen] = useState(false)
  const resolve = useRef<((value: PreparedFolder) => void) | undefined>(undefined)
  useEffect(() => () => { resolve.current?.({ canceled: true }) }, [])
  async function pick(createLibrary = false, names: string[] = []): Promise<PreparedFolder> {
    const picked = createLibrary ? undefined : await window.tokensmith!.pickDocumentFolder()
    if (!createLibrary && (!picked?.path || picked.canceled)) return { canceled: true }
    resolve.current?.({ canceled: true })
    setCreatingLibrary(createLibrary); setExistingNames(names); setError(''); editedName.current = false
    setName(picked?.title || ''); setSettings(defaultPreparation()); setModeChosen(false); setFolder(picked); setOpen(true)
    return new Promise(done => { resolve.current = done })
  }
  async function chooseFolder() {
    setChoosing(true); setError('')
    try {
      const picked = await window.tokensmith!.pickDocumentFolder()
      if (picked.canceled || !picked.path) return
      setFolder(picked)
      if (!editedName.current) setName(creatingLibrary ? uniqueLibraryName(picked.title || 'Library', existingNames) : picked.title || 'Document Folder')
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not select the folder.') }
    finally { setChoosing(false) }
  }
  const duplicate = creatingLibrary && existingNames.some(existing => libraryNameKey(existing) === libraryNameKey(name))
  const canConfirm = Boolean(folder?.path && modeChosen && name.trim() && !duplicate && !choosing)
  const confirmationHint = choosing ? 'Choosing documents…' : !folder?.path ? 'Choose a folder or file to continue.' : !name.trim() ? 'Enter a library name to continue.' : duplicate ? 'Choose a unique library name to continue.' : !modeChosen ? 'Choose a chunking mode to continue.' : ''
  function finish(confirmed: boolean) {
    if (confirmed && !canConfirm) return
    resolve.current?.(confirmed && folder ? { ...folder, title: name.trim(), preparation: settings } : { canceled: true })
    resolve.current = undefined; setFolder(undefined); setOpen(false)
  }
  const panel = open && <section className="folder-preparation" aria-label="Prepare document folder">
    <h2>{creatingLibrary ? 'Create New Library' : 'Add documents'}</h2>
    <div className="folder-preparation-source">
      <button type="button" className={`${folder?.path ? 'secondary-action' : 'primary-action'} folder-preparation-choose`} disabled={choosing} onClick={() => void chooseFolder()}><FolderOpen size={22} aria-hidden="true" />{choosing ? 'Choosing…' : 'Choose folder or file'}</button>
      {folder?.path && <p className="folder-preparation-selected-path">{folder.path}</p>}
    </div>
    {error && <p className="inline-error" role="alert">{error}</p>}
    <label className="folder-preparation-name">Name<input value={name} maxLength={200} onChange={event => { editedName.current = true; setName(event.target.value) }} /></label>
    {duplicate && <p className="inline-error" role="alert">A library with this name already exists. Choose another name.</p>}
    <div className="folder-preparation-row"><strong>OCR</strong><div role="group" aria-label="OCR">
      <button type="button" className="secondary-action" aria-pressed={settings.ocr !== 'none'} onClick={() => setSettings({ ...settings, ocr: 'rapid' })}>Yes</button>
      <button type="button" className="secondary-action" aria-pressed={settings.ocr === 'none'} onClick={() => setSettings({ ...settings, ocr: 'none' })}>No</button>
    </div></div>
    {settings.ocr !== 'none' && <div className="folder-preparation-row"><strong>OCR engine</strong><div role="group" aria-label="OCR engine">
      {(['rapid', 'gemma'] as const).map(ocr => <button key={ocr} type="button" className="secondary-action" aria-pressed={settings.ocr === ocr}
        onClick={() => setSettings({ ...settings, ocr })}>{ocr === 'rapid' ? 'Rapid OCR' : 'Gemma OCR'}</button>)}
    </div></div>}
    <div className="folder-preparation-row"><strong>Chunking Mode</strong><div role="group" aria-label="Chunking Mode">
      {(['basic', 'ai'] as const).map(mode => <button key={mode} type="button" className="secondary-action" aria-pressed={modeChosen && settings.mode === mode}
        onClick={() => { setSettings({ ...settings, mode, extractTopics: mode === 'ai' }); setModeChosen(true) }}>{mode === 'basic' ? 'Basic Chunking' : 'LLM-Based Chunking'}</button>)}
    </div></div>
    <div className="folder-preparation-row"><strong>Topic Extraction</strong><div role="group" aria-label="Topic Extraction">
      {[true, false].map(extractTopics => <button key={String(extractTopics)} type="button" className="secondary-action" disabled={!modeChosen} aria-pressed={modeChosen && (settings.extractTopics !== false) === extractTopics}
        onClick={() => setSettings({ ...settings, extractTopics })}>{extractTopics ? 'Yes' : 'No'}</button>)}
    </div></div>
    <p className="library-hint">Markdown and text files never use OCR.</p>
    {confirmationHint && <p id={confirmationHintId} className="folder-preparation-required" role="status">{confirmationHint}</p>}
    <div className="folder-preparation-actions"><button type="button" className="secondary-action" onClick={() => finish(false)}><X size={16} aria-hidden="true" />Cancel</button>
      <button type="button" className="primary-action" disabled={!canConfirm} aria-describedby={confirmationHint ? confirmationHintId : undefined} onClick={() => finish(true)}><Plus size={16} aria-hidden="true" />{creatingLibrary ? 'Create library' : 'Add documents'}</button></div>
  </section>
  return { pick, panel }
}
