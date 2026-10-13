import { useId, useMemo, useState } from 'react'
import { Activity, ChevronDown } from 'lucide-react'
import {
  buildLatencyBreakdown,
  formatLatencyTrace,
  type LatencyBreakdown,
  type LatencyStageBreakdownItem,
  type LatencyTrace
} from '@shared/latency-trace'

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

export function latencyPlanBreakdown(trace: LatencyTrace): LatencyBreakdown | undefined {
  try {
    return buildLatencyBreakdown(trace)
  } catch {
    return undefined
  }
}

function stageToneClass(stage: LatencyStageBreakdownItem): string {
  if (stage.status === 'skipped') return 'is-skipped'
  if (stage.isBottleneck || stage.status === 'error') return 'is-bottleneck'
  return `stage-tone-${stage.toneIndex}`
}

function StageRow({
  item,
  depth = 0,
  expandedDescriptions,
  onToggleDescription,
  panelId
}: {
  item: LatencyStageBreakdownItem
  depth?: number
  expandedDescriptions: ReadonlySet<string>
  onToggleDescription: (key: string) => void
  panelId: string
}) {
  const toneClass = stageToneClass(item)
  const isDescriptionOpen = expandedDescriptions.has(item.key)
  const descriptionId = `${panelId}-desc-${item.key.replace(/[^a-z0-9-_]/gi, '-')}`
  const activeChildren = item.children.filter((child) => child.status !== 'skipped' && child.durationMs > 0)

  return (
    <div
      className={`query-plan-stage-banner ${depth > 0 ? 'is-substage' : ''} ${item.isBottleneck ? 'is-bottleneck' : ''} ${item.status === 'skipped' ? 'is-skipped' : ''} ${item.status === 'error' ? 'is-error' : ''}`}
      data-stage={item.label}
      data-bottleneck={item.isBottleneck ? 'true' : 'false'}
    >
      <div className="query-plan-stage-header">
        <div className="query-plan-stage-title-group">
          <span className={`query-plan-stage-swatch ${toneClass}`} aria-hidden="true" />
          <strong className="query-plan-stage-name">{item.label}</strong>
          <button
            type="button"
            className={`query-plan-info-button ${isDescriptionOpen ? 'is-active' : ''}`}
            aria-expanded={isDescriptionOpen}
            aria-controls={descriptionId}
            aria-label={`About ${item.label}`}
            title={item.description}
            onClick={() => onToggleDescription(item.key)}
          >
            <span aria-hidden="true">i</span>
          </button>
          {item.isBottleneck && (
            <span className="query-plan-bottleneck-badge" title="Slowest stage in this pipeline level">
              Bottleneck
            </span>
          )}
          {item.status === 'error' && !item.isBottleneck && (
            <span className="query-plan-error-badge">Issue</span>
          )}
        </div>
        <div className="query-plan-stage-metrics">
          {item.summaryBadges.map((badge) => (
            <span className="query-plan-metric-pill" key={badge}>
              {badge}
            </span>
          ))}
          <span className={`query-plan-stage-duration ${item.isBottleneck ? 'is-bottleneck' : ''}`}>
            {item.durationLabel}
          </span>
          {item.status !== 'skipped' && item.durationMs > 0 && (
            <span className="query-plan-stage-share">{item.percentLabel}</span>
          )}
        </div>
      </div>

      {isDescriptionOpen && (
        <p className="query-plan-stage-description" id={descriptionId}>
          {item.description}
        </p>
      )}

      {item.children.length > 0 && (
        <div className="query-plan-substages">
          {activeChildren.length > 1 && (
            <div
              className="query-plan-stacked-bar is-subbar"
              role="img"
              aria-label={`${item.label} sub-stage time breakdown`}
            >
              {activeChildren.map((child) => (
                <button
                  key={child.key}
                  type="button"
                  className={`query-plan-bar-segment ${stageToneClass(child)}`}
                  style={{ width: `${Math.max(child.widthPercent, 4)}%`, flexGrow: Math.max(child.durationMs, 1) }}
                  title={`${child.label}: ${child.durationLabel} (${child.percentLabel})${child.isBottleneck ? ' — Bottleneck' : ''}`}
                  aria-label={`${child.label}: ${child.durationLabel} (${child.percentLabel})${child.isBottleneck ? ', bottleneck' : ''}`}
                  onClick={() => onToggleDescription(child.key)}
                />
              ))}
            </div>
          )}
          <div className="query-plan-substage-list">
            {item.children.map((child) => (
              <StageRow
                key={child.key}
                item={child}
                depth={depth + 1}
                expandedDescriptions={expandedDescriptions}
                onToggleDescription={onToggleDescription}
                panelId={panelId}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export function QueryPlanPanel({ trace, wallTimeMs }: { trace?: LatencyTrace; wallTimeMs?: number }) {
  const [open, setOpen] = useState(false)
  const [expandedDescriptions, setExpandedDescriptions] = useState<Set<string>>(() => new Set())
  const panelId = useId()
  const text = useMemo(() => (trace ? latencyPlanText(trace, wallTimeMs) : undefined), [trace, wallTimeMs])
  const breakdown = useMemo(() => (trace ? latencyPlanBreakdown(trace) : undefined), [trace])
  if (!trace || !text || !breakdown) return null

  const hasWallTime = typeof wallTimeMs === 'number' && Number.isFinite(wallTimeMs) && wallTimeMs >= 0
  const allStageKeys = useMemo(() => {
    const keys: string[] = []
    for (const stage of breakdown.stages) {
      keys.push(stage.key)
      for (const child of stage.children) {
        keys.push(child.key)
      }
    }
    return keys
  }, [breakdown])

  const allDescriptionsOpen = allStageKeys.length > 0 && allStageKeys.every((key) => expandedDescriptions.has(key))

  function toggleDescription(key: string) {
    setExpandedDescriptions((current) => {
      const next = new Set(current)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  function toggleAllDescriptions() {
    setExpandedDescriptions(() => (allDescriptionsOpen ? new Set() : new Set(allStageKeys)))
  }

  return (
    <div className="query-plan">
      <button
        className="source-chip"
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        <Activity size={16} aria-hidden="true" />
        <span>Latency plan</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {open && (
        <div className="query-plan-body" id={panelId} role="region" aria-label="Latency plan">
          <div className="query-plan-top-row">
            <dl className="query-plan-totals">
              <div>
                <dt>Measured stages</dt>
                <dd>{formatSeconds(trace.totalDurationMs)}</dd>
              </div>
              {hasWallTime && (
                <div>
                  <dt>Wall time</dt>
                  <dd>{formatSeconds(wallTimeMs)}</dd>
                </div>
              )}
              {breakdown.bottleneck && (
                <div className="query-plan-bottleneck-summary">
                  <dt>Bottleneck</dt>
                  <dd>
                    {breakdown.bottleneck.label} ({breakdown.bottleneck.durationLabel} · {breakdown.bottleneck.percentLabel})
                    {breakdown.bottleneck.substageLabel
                      ? ` — ${breakdown.bottleneck.substageLabel} (${breakdown.bottleneck.substageDurationLabel})`
                      : ''}
                  </dd>
                </div>
              )}
            </dl>
            <button
              type="button"
              className="query-plan-explain-all-btn"
              aria-pressed={allDescriptionsOpen}
              onClick={toggleAllDescriptions}
            >
              <span className="query-plan-info-icon-inline" aria-hidden="true">i</span>
              <span>{allDescriptionsOpen ? 'Hide stage guides' : 'Explain all stages'}</span>
            </button>
          </div>

          {breakdown.barSegments.length > 0 && (
            <div className="query-plan-bar-container">
              <div
                className="query-plan-stacked-bar"
                role="img"
                aria-label={`Horizontal stacked latency bar. Bottleneck: ${breakdown.bottleneck?.label ?? 'none'}`}
              >
                {breakdown.barSegments.map((segment) => (
                  <button
                    key={segment.key}
                    type="button"
                    className={`query-plan-bar-segment ${stageToneClass(segment)}`}
                    style={{
                      width: `${Math.max(segment.widthPercent, 4)}%`,
                      flexGrow: Math.max(segment.durationMs, 1)
                    }}
                    title={`${segment.label}: ${segment.durationLabel} (${segment.percentLabel})${segment.isBottleneck ? ' — Bottleneck' : ''}`}
                    aria-label={`${segment.label}: ${segment.durationLabel} (${segment.percentLabel})${segment.isBottleneck ? ', bottleneck' : ''}`}
                    onClick={() => toggleDescription(segment.key)}
                  />
                ))}
              </div>
              <div className="query-plan-bar-legend" aria-label="Stage time legend">
                {breakdown.barSegments.map((segment) => (
                  <button
                    key={segment.key}
                    type="button"
                    className={`query-plan-legend-item ${segment.isBottleneck ? 'is-bottleneck' : ''}`}
                    onClick={() => toggleDescription(segment.key)}
                    title={`Click to view description for ${segment.label}`}
                  >
                    <span className={`query-plan-stage-swatch ${stageToneClass(segment)}`} aria-hidden="true" />
                    <span className="query-plan-legend-label">{segment.label}</span>
                    <span className="query-plan-legend-time">{segment.durationLabel}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="query-plan-stages" aria-label="Answer generation stages">
            {breakdown.stages.map((stage) => (
              <StageRow
                key={stage.key}
                item={stage}
                expandedDescriptions={expandedDescriptions}
                onToggleDescription={toggleDescription}
                panelId={panelId}
              />
            ))}
          </div>

          <details className="query-plan-raw-details">
            <summary>Raw execution trace</summary>
            <pre className="query-plan-text">{text}</pre>
          </details>
        </div>
      )}
    </div>
  )
}
