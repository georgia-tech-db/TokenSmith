import { useEffect, useState } from 'react'
import type { CourseMaterial } from '@shared/app-state'
import type { StudyDocument } from '@shared/study-scope'

export interface StudyCatalog {
  documents: StudyDocument[]
  status: 'loading' | 'ready' | 'error'
  error: string
  reload: () => void
}

export function useStudyDocuments(materials: CourseMaterial[], enabled: boolean): StudyCatalog {
  const [documents, setDocuments] = useState<StudyDocument[]>([])
  const [status, setStatus] = useState<StudyCatalog['status']>('loading')
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)
  const signature = materials.map(item => `${item.id}:${item.status}:${item.indexedAt}:${item.chunkCount}`).join('|')
  useEffect(() => {
    if (!enabled) return
    let disposed = false
    setStatus('loading')
    setError('')
    const request = window.tokensmith?.studyDocuments
      ? window.tokensmith.studyDocuments() : Promise.reject(new Error('Update TokenSmith to load study documents.'))
    void request.then(items => {
      if (disposed) return
      setDocuments(items); setStatus('ready')
    }).catch(reason => {
      if (disposed) return
      setError(reason instanceof Error ? reason.message : String(reason)); setStatus('error')
    })
    return () => { disposed = true }
  }, [enabled, signature, revision])
  return { documents, status, error, reload: () => setRevision(value => value + 1) }
}
