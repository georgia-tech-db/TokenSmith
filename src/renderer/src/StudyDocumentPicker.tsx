import { useEffect, useRef } from 'react'
import { FileText, Folder } from 'lucide-react'
import { studyDocumentKey, type StudyDocument, type StudyDocumentRef } from '@shared/study-scope'

function CollectionCheckbox({ checked, partial, onChange, title, disabled }: { checked: boolean; partial: boolean; onChange: () => void; title: string; disabled: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => { if (input.current) input.current.indeterminate = partial }, [partial])
  return <label><input ref={input} type="checkbox" checked={checked} disabled={disabled} onChange={onChange} /><Folder size={17} aria-hidden="true" /><span>{title}</span></label>
}

export function StudyDocumentPicker({ documents, scope, disabled, onChange }: {
  documents: StudyDocument[]; scope: StudyDocumentRef[]; disabled: boolean; onChange: (scope: StudyDocumentRef[]) => void
}) {
  const selected = scope.map(studyDocumentKey)
  const collections = [...new Set(documents.map(document => document.materialId))]
  const toggle = (files: StudyDocument[], checked: boolean) => {
    const keys = files.map(studyDocumentKey)
    onChange(checked ? [...scope.filter(item => !keys.includes(studyDocumentKey(item))), ...files.map(({ materialId, documentId }) => ({ materialId, documentId }))]
      : scope.filter(item => !keys.includes(studyDocumentKey(item))))
  }
  return <div className="study-document-picker" aria-label="Study documents">
    {collections.map(id => {
      const files = documents.filter(document => document.materialId === id)
      const keys = files.map(studyDocumentKey)
      const count = keys.filter(key => selected.includes(key)).length
      return <div className="study-collection" key={id}>
        <CollectionCheckbox title={files[0].collectionName} checked={count === keys.length}
          partial={count > 0 && count < keys.length} disabled={disabled} onChange={() => toggle(files, count !== keys.length)} />
        <details open><summary>Documents ({files.length})</summary><div className="study-document-list">
          {files.map(document => <label key={studyDocumentKey(document)}>
            <input type="checkbox" disabled={disabled} checked={selected.includes(studyDocumentKey(document))}
              onChange={event => toggle([document], event.target.checked)} />
            <FileText size={16} aria-hidden="true" /><span>{document.title}</span>
          </label>)}
        </div></details>
      </div>
    })}
  </div>
}
