import type { ReactNode } from 'react'
import { FolderOpen, RefreshCw, X } from 'lucide-react'
import { studyDocumentKey, type StudyDocumentRef } from '@shared/study-scope'
import type { StudyCatalog } from './hooks/useStudyDocuments'
import { StudyDocumentPicker } from './StudyDocumentPicker'
import './study-workspace.css'

export function StudyMaterialsPane({ catalog, scope, open, disabled, locked, onChange, onClose, onManage, onNewPractice, children }: {
  catalog: StudyCatalog; scope: StudyDocumentRef[]; open: boolean; disabled?: boolean; locked?: boolean
  onChange: (scope: StudyDocumentRef[]) => void; onClose: () => void; onManage: () => void
  onNewPractice?: () => void; children?: ReactNode
}) {
  const available = new Set(catalog.documents.map(studyDocumentKey))
  const missing = scope.filter(item => !available.has(studyDocumentKey(item))).length
  return <aside className="study-materials" aria-label="Study materials" hidden={!open}>
    <header><div><h2>Materials</h2><span>{scope.length} document{scope.length === 1 ? '' : 's'} selected</span></div>
      <button type="button" className="icon-button subtle" aria-label="Close materials" title="Close materials" onClick={onClose}><X size={18} /></button></header>
    <div className="study-materials-scroll">
      {locked && <div className="study-scope-note"><span>Documents for this session</span><button className="practice-link" type="button" onClick={onNewPractice}>New practice</button></div>}
      {catalog.status === 'loading' && <p role="status">Loading documents...</p>}
      {catalog.status === 'error' && <div role="alert"><p>{catalog.error}</p><button className="practice-link" type="button" onClick={catalog.reload}><RefreshCw size={15} />Retry</button></div>}
      {catalog.status === 'ready' && <>
        {missing > 0 && <div role="status" className="study-scope-note"><span>{missing} selected document{missing === 1 ? ' is' : 's are'} unavailable.</span>
          {!locked && <button className="practice-link" type="button" disabled={disabled} onClick={() => onChange(scope.filter(item => available.has(studyDocumentKey(item))))}>Remove unavailable selections</button>}</div>}
        {!catalog.documents.length ? <p>No indexed documents yet.</p> : <StudyDocumentPicker documents={catalog.documents} scope={scope}
          disabled={Boolean(disabled || locked)} onChange={onChange} />}
      </>}
      {children && <section className="study-supporting-passages"><h3>Supporting passages</h3>{children}</section>}
    </div>
    <footer><button className="practice-link" type="button" onClick={onManage}><FolderOpen size={17} />Manage Library</button></footer>
  </aside>
}
