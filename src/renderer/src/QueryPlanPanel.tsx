import { useId, useMemo, useState } from 'react'
import { Activity, ChevronDown } from 'lucide-react'
import { formatLatencyTrace, type LatencyTrace } from '@shared/latency-trace'

function formatSeconds(durationMs: number): string {
  return `${(durationMs / 1000).toFixed(2)} s`
}

export function latencyPlanText(trace: LatencyTrace, wallTimeMs?: number): string | undefined {
  // A formatting problem must never break an otherwise successful answer.
  try {
    return formatLatencyTrace(trace, wallTimeMs)
  } catch {
    return undefined
  }
}

export function QueryPlanPanel({ trace, wallTimeMs }: { trace?: LatencyTrace; wallTimeMs?: number }) {
  const [open, setOpen] = useState(false)
  const panelId = useId()
  const text = useMemo(() => trace ? latencyPlanText(trace, wallTimeMs) : undefined, [trace, wallTimeMs])
  if (!trace || !text) return null

  const hasWallTime = typeof wallTimeMs === 'number' && Number.isFinite(wallTimeMs) && wallTimeMs >= 0
  return (
    <div className="query-plan">
      <button className="source-chip" type="button" aria-expanded={open} aria-controls={panelId}
        onClick={() => setOpen(value => !value)}>
        <Activity size={16} aria-hidden="true" />
        <span>Latency plan</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {open && (
        <div className="query-plan-body" id={panelId} role="region" aria-label="Latency plan">
          <dl className="query-plan-totals">
            <div><dt>Measured stages</dt><dd>{formatSeconds(trace.totalDurationMs)}</dd></div>
            {hasWallTime && <div><dt>Wall time</dt><dd>{formatSeconds(wallTimeMs)}</dd></div>}
          </dl>
          <pre className="query-plan-text">{text}</pre>
        </div>
      )}
    </div>
  )
}
