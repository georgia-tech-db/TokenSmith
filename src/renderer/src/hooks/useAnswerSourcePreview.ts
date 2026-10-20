import { useCallback, useMemo, useState } from 'react'
import { createAnswerSourcePreview, type SourceReadingContext, type SourceReadingStatus } from '../answer-source-preview'
import type { SourceReaderState } from './useSourceReader'

interface RequestReadingContext extends SourceReadingContext {
  requestId: string | null
}

interface AnswerSourcePreview {
  source: SourceReaderState | null
  reading: SourceReadingContext | null
  begin: (requestId: string) => void
  open: (requestId: string, question: string, load: () => Promise<SourceReaderState>) => Promise<void>
  finish: (requestId: string, status?: 'ready' | 'stopped') => void
  dismiss: () => void
  keepOpen: () => void
  openManual: (question: string | undefined, status: SourceReadingStatus, requestId?: string | null) => void
  transferToManual: () => void
}

// Own the automatic reader and its question/status together. Manual document
// navigation remains in useSourceReader; transferring keeps the reading context.
export function useAnswerSourcePreview(): AnswerSourcePreview {
  const [source, setSource] = useState<SourceReaderState | null>(null)
  const [reading, setReading] = useState<RequestReadingContext | null>(null)
  const lifecycle = useMemo(() => createAnswerSourcePreview<SourceReaderState>(setSource), [])

  const begin = useCallback((requestId: string) => {
    lifecycle.begin(requestId)
    setReading(null)
  }, [lifecycle])

  const open = useCallback((requestId: string, question: string, load: () => Promise<SourceReaderState>) =>
    lifecycle.open(requestId, () => {
      // The lifecycle rejects stale requests before invoking this loader.
      setReading({ requestId, question, keptOpen: false, status: 'preparing' })
      return load()
    }), [lifecycle])

  const finish = useCallback((requestId: string, status: 'ready' | 'stopped' = 'stopped') => {
    lifecycle.finish(requestId)
    setReading(current => current?.requestId === requestId && current.status === 'preparing'
      ? { ...current, status } : current)
  }, [lifecycle])

  const dismiss = useCallback(() => {
    lifecycle.dismiss()
    setReading(null)
  }, [lifecycle])

  const keepOpen = useCallback(() => {
    if (!source || !reading?.requestId || !lifecycle.keepOpen(reading.requestId)) return
    setReading(current => current?.requestId === reading.requestId ? { ...current, keptOpen: true } : current)
  }, [lifecycle, source, reading])

  const openManual = useCallback((question: string | undefined, status: SourceReadingStatus, requestId?: string | null) => {
    lifecycle.dismiss()
    // The active chat supplies the request even after its preview was dismissed.
    setReading(question ? { question, keptOpen: true, status,
      requestId: status === 'preparing' ? requestId ?? null : null } : null)
  }, [lifecycle])

  const transferToManual = useCallback(() => {
    lifecycle.dismiss()
    setReading(current => current ? { ...current, keptOpen: true } : current)
  }, [lifecycle])

  return { source, reading, begin, open, finish, dismiss, keepOpen, openManual, transferToManual }
}
