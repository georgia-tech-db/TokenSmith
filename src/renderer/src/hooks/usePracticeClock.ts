import { useEffect, useRef } from 'react'
import type { PracticeObservation, PracticeQuestion, PracticeSession } from '@shared/practice'

export function usePracticeClock(session: PracticeSession | undefined, question: PracticeQuestion | undefined, enabled: boolean,
  update: (id: string, fn: (s: PracticeSession) => PracticeSession) => void) {
  const segment = useRef<{ start: number; questionId: string } | undefined>(undefined)
  useEffect(() => {
    if (!session || !question || question.attempts.length || !enabled) return
    const add = (kind: PracticeObservation['kind'], visibleMs?: number) => update(session.id, s => ({ ...s,
      observations: [...(s.observations ?? []), { id: crypto.randomUUID(), questionId: question.id, kind, at: new Date().toISOString(), visibleMs }] }))
    const pause = () => {
      if (!segment.current) return
      const duration = Math.max(0, performance.now() - segment.current.start)
      segment.current = undefined
      add('paused', duration)
    }
    const resume = () => {
      if (document.visibilityState !== 'visible' || !document.hasFocus() || segment.current) return
      segment.current = { start: performance.now(), questionId: question.id }; add('presented')
    }
    const visibility = () => document.visibilityState === 'visible' ? resume() : pause()
    resume()
    window.addEventListener('focus', resume); window.addEventListener('blur', pause)
    document.addEventListener('visibilitychange', visibility)
    window.addEventListener('pagehide', pause)
    return () => {
      pause(); window.removeEventListener('focus', resume); window.removeEventListener('blur', pause)
      document.removeEventListener('visibilitychange', visibility); window.removeEventListener('pagehide', pause)
    }
  }, [session?.id, question?.id, question?.attempts.length, enabled])
  return () => {
    const events = session?.observations?.filter(e => e.questionId === question?.id) ?? []
    const first = events.find(e => e.kind === 'presented')
    const live = segment.current?.questionId === question?.id ? Math.max(0, performance.now() - segment.current!.start) : 0
    return { submittedAt: new Date().toISOString(), durationMs: first ? Math.max(0, Date.now() - Date.parse(first.at)) : 0,
      visibleMs: events.reduce((n, e) => n + (e.kind === 'paused' ? e.visibleMs ?? 0 : 0), 0) + live,
      timingComplete: Boolean(first) && events.filter(e => e.kind === 'presented').length === events.filter(e => e.kind === 'paused').length + Number(Boolean(segment.current)) }
  }
}
