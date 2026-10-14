import type { ReactNode } from 'react'
import { FolderOpen, RefreshCw, X } from 'lucide-react'
import { studyDocumentKey, type StudyDocumentRef } from '@shared/study-scope'
import type { StudyCatalog } from './hooks/useStudyDocuments'
import { StudyDocumentPicker, type StudyDocumentExpansion } from './StudyDocumentPicker'
import './study-workspace.css'
import './answer-explanation.css'

export function StudyMaterialsPane({ catalog, scope, open, disabled, onChange, onClose, onManage, children, collectionExpansion, onCollectionExpansionChange }: {
  catalog: StudyCatalog; scope: StudyDocumentRef[]; open: boolean; disabled?: boolean
  onChange: (scope: StudyDocumentRef[]) => void; onClose: () => void; onManage: () => void
  children?: ReactNode
} & StudyDocumentExpansion) {
  const available = new Set(catalog.documents.map(studyDocumentKey))
  const missing = scope.filter(item => !available.has(studyDocumentKey(item))).length
  return <aside className="study-materials" aria-label="Library" hidden={!open}>
    <header><div><h2>Library</h2><span>{scope.length} document{scope.length === 1 ? '' : 's'} selected</span></div>
      <button type="button" className="icon-button subtle" aria-label="Close Library" title="Close Library" onClick={onClose}><X size={18} /></button></header>
    <div className={`study-materials-body ${children ? 'has-sources' : ''}`}>
      <div className="study-materials-scroll" role="region" aria-label="Folders and documents" tabIndex={0}>
        {catalog.status === 'loading' && <p role="status">Loading documents...</p>}
        {catalog.status === 'error' && <div role="alert"><p>{catalog.error}</p><button className="practice-link" type="button" onClick={catalog.reload}><RefreshCw size={15} />Retry</button></div>}
        {catalog.status === 'ready' && <>
          {missing > 0 && <div role="status" className="study-scope-note"><span>{missing} selected document{missing === 1 ? ' is' : 's are'} unavailable.</span>
            <button className="practice-link" type="button" disabled={disabled} onClick={() => onChange(scope.filter(item => available.has(studyDocumentKey(item))))}>Remove unavailable selections</button></div>}
          {!catalog.documents.length ? <p>No indexed documents yet.</p> : <StudyDocumentPicker documents={catalog.documents} scope={scope}
            disabled={Boolean(disabled)} onChange={onChange} collectionExpansion={collectionExpansion} onCollectionExpansionChange={onCollectionExpansionChange} />}
        </>}
      </div>
      {children}
    </div>
    <footer><button className="study-action-button" type="button" onClick={onManage}><FolderOpen size={17} aria-hidden="true" />Manage Library</button></footer>
  </aside>
}
