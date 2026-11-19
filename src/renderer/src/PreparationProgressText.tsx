import { useEffect, useState } from 'react'
import type { MaterialIndexProgress } from '@shared/app-state'
import { preparationProgress } from '@shared/preparation-progress'

export function PreparationProgressText({ status }: { status?: MaterialIndexProgress }) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    setElapsed(0)
    const started = performance.now()
    const timer = window.setInterval(() => setElapsed(performance.now() - started), 1000)
    return () => window.clearInterval(timer)
  }, [status])
  return <>{preparationProgress(status, elapsed).text}</>
}
