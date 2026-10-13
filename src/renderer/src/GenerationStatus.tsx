import { useEffect, useState } from 'react'
import { waitDisplay } from '@shared/generation-estimate'
import type { GenerationProgress } from './hooks/useGenerationProgress'
import './generation-status.css'

export function GenerationStatus({ progress, onOpen }: { progress: GenerationProgress | null; onOpen: (conversationId: string) => void }) {
  const [now, setNow] = useState(performance.now())
  useEffect(() => {
    setNow(performance.now())
    if (!progress || progress.complete) return
    const timer = window.setInterval(() => setNow(performance.now()), 1000)
    return () => window.clearInterval(timer)
  }, [progress?.id, progress?.startedAt, progress?.complete])
  const display = progress ? waitDisplay(progress.estimate, now - progress.startedAt) : null
  return <footer className="generation-status" aria-label="Generation status">
    <span className="sr-only" role="status" aria-live="polite">{progress?.label ?? ''}</span>
    {progress && <>
      <button type="button" className="generation-status-label" onClick={() => onOpen(progress.conversationId)} title="Open conversation">{progress.label}</button>
      <progress aria-label="Estimated time progress" aria-valuetext={progress.complete ? 'Complete' : display?.text}
        max={1} value={progress.complete ? 1 : display?.fraction} />
      <span className="generation-status-time" title={progress.estimate.samples ? 'Based on recent runs with this model on this installation' : 'Initial estimate; learns from completed runs'}>
        {progress.complete ? '' : display?.text}
      </span>
      {!progress.complete && <button type="button" className="generation-status-stop" onClick={progress.onStop}>Stop</button>}
    </>}
  </footer>
}
