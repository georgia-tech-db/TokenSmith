import { useState } from 'react'
import { formatDuration, type LatencyPlan, type LatencyStageBreakdownItem } from './latency-plan'
import './query-plan.css'

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
                <span
                  key={child.key}
                  aria-hidden="true"
                  className={`query-plan-bar-segment ${stageToneClass(child)}`}
                  style={{ flex: `0 0 ${child.widthPercent}%` }}
                  title={`${child.label}: ${child.durationLabel} (${child.percentLabel})${child.isBottleneck ? ' — Bottleneck' : ''}`}
                  aria-label={`${child.label}: ${child.durationLabel} (${child.percentLabel})${child.isBottleneck ? ', bottleneck' : ''}`}
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

function stageKeys(stages: LatencyStageBreakdownItem[]): string[] {
  return stages.flatMap((stage) => [stage.key, ...stageKeys(stage.children)])
}

export function QueryPlanPanel({ plan, panelId, answerReadyMs }: {
  plan: LatencyPlan
  panelId: string
  answerReadyMs?: number
}) {
  const [expandedDescriptions, setExpandedDescriptions] = useState<Set<string>>(() => new Set())
  const { text, breakdown } = plan
  const hasAnswerTime = typeof answerReadyMs === 'number' && Number.isFinite(answerReadyMs) && answerReadyMs >= 0
  const allStageKeys = stageKeys(breakdown.stages)
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
      <div className="query-plan-body" id={panelId} role="region" aria-label="Query plan">
        <p className="query-plan-note">Time spent preparing this answer. Measured stages include background follow-up suggestions; answer ready time ends when the answer appears.</p>
        <div className="query-plan-top-row">
          <dl className="query-plan-totals">
            <div>
              <dt>Measured stages</dt>
              <dd>{breakdown.totalDurationLabel}</dd>
            </div>
            {hasAnswerTime && (
              <div>
                <dt>Answer ready in</dt>
                <dd>{formatDuration(answerReadyMs!)}</dd>
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
                <span
                  key={segment.key}
                  aria-hidden="true"
                  className={`query-plan-bar-segment ${stageToneClass(segment)}`}
                  style={{
                    flex: `0 0 ${segment.widthPercent}%`
                  }}
                  title={`${segment.label}: ${segment.durationLabel} (${segment.percentLabel})${segment.isBottleneck ? ' — Bottleneck' : ''}`}
                  aria-label={`${segment.label}: ${segment.durationLabel} (${segment.percentLabel})${segment.isBottleneck ? ', bottleneck' : ''}`}
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
    </div>
  )
}
