import { getLatencyStageInfo, measuredDurationMs, mergeLatencySpans, type LatencyDetail, type LatencySpan, type LatencySpanStatus, type LatencyTrace } from '../../shared/latency-trace'

export interface LatencyStageBreakdownItem {
  key: string
  stage: string
  label: string
  description: string
  durationMs: number
  durationLabel: string
  share: number
  widthPercent: number
  percentLabel: string
  status: LatencySpanStatus
  isBottleneck: boolean
  toneIndex: number
  inCount?: number
  outCount?: number
  summaryBadges: string[]
  details?: Record<string, LatencyDetail>
  children: LatencyStageBreakdownItem[]
}

export interface LatencyBottleneckSummary {
  stage: string
  label: string
  durationMs: number
  durationLabel: string
  percentLabel: string
  substageLabel?: string
  substageDurationLabel?: string
}

export interface LatencyBreakdown {
  totalDurationMs: number
  totalDurationLabel: string
  bottleneck?: LatencyBottleneckSummary
  barSegments: LatencyStageBreakdownItem[]
  stages: LatencyStageBreakdownItem[]
}

function finiteNonNegative(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

export function formatDuration(durationMs: number): string {
  const safeDuration = finiteNonNegative(durationMs) ?? 0
  if (safeDuration >= 1000) return `${(safeDuration / 1000).toFixed(2)} s`
  if (safeDuration >= 100) return `${Math.round(safeDuration)} ms`
  if (safeDuration >= 10) return `${safeDuration.toFixed(1)} ms`
  return `${safeDuration.toFixed(2)} ms`
}

function formatPercent(share: number): string {
  if (!Number.isFinite(share) || share <= 0) return '0%'
  const percent = share * 100
  if (percent >= 99.5 && percent < 100) return '99%'
  if (percent < 1) return '<1%'
  return `${Math.round(percent)}%`
}

function findBottleneckIndex(spans: ReadonlyArray<LatencySpan>): number {
  let bestIndex = -1
  let bestDuration = 0
  spans.forEach((span, index) => {
    const duration = finiteNonNegative(span.durationMs) ?? 0
    if (span.status === 'skipped' || duration <= 0) return
    if (duration > bestDuration) {
      bestDuration = duration
      bestIndex = index
    }
  })
  return bestIndex
}

function buildSummaryBadges(span: LatencySpan): string[] {
  const badges: string[] = []
  if (span.inCount !== undefined && span.outCount !== undefined) {
    badges.push(`${span.inCount} → ${span.outCount}`)
  } else if (span.outCount !== undefined) {
    badges.push(`out: ${span.outCount}`)
  } else if (span.inCount !== undefined) {
    badges.push(`in: ${span.inCount}`)
  }

  const details = span.details
  if (details) {
    if (typeof details.prompt_tokens === 'number') {
      badges.push(`${details.prompt_tokens} prompt tok`)
    }
    if (typeof details.generated_tokens === 'number') {
      badges.push(`${details.generated_tokens} output tok`)
    }
    if (typeof details.mode === 'string' && details.mode) {
      badges.push(`mode: ${details.mode}`)
    }
    if (typeof details.fits === 'boolean') {
      badges.push(details.fits ? 'fits window' : 'resized/trimmed')
    }
  }
  return badges
}

function buildStageItems(
  spans: ReadonlyArray<LatencySpan>,
  referenceTotalMs: number,
  markBottleneck: boolean,
  parentKey = 'stage'
): LatencyStageBreakdownItem[] {
  const normalized = [...spans]
  const bottleneckIndex = markBottleneck ? findBottleneckIndex(normalized) : -1
  const activeTotal = measuredDurationMs(
    normalized.filter((span) => span.status !== 'skipped' && (finiteNonNegative(span.durationMs) ?? 0) > 0)
  )
  const denominator = referenceTotalMs > 0 ? referenceTotalMs : activeTotal

  return normalized.map((span, index) => {
    const info = getLatencyStageInfo(span.stage)
    const durationMs = finiteNonNegative(span.durationMs) ?? 0
    const status: LatencySpanStatus = span.status ?? 'ok'
    const isActive = status !== 'skipped' && durationMs > 0
    const share = isActive && denominator > 0 ? Math.min(1, durationMs / denominator) : 0
    const widthPercent = isActive && activeTotal > 0 ? (durationMs / activeTotal) * 100 : 0
    const isBottleneck = index === bottleneckIndex
    const durationLabel = status === 'skipped'
      ? 'Skipped'
      : status === 'error'
        ? `Error · ${formatDuration(durationMs)}`
        : formatDuration(durationMs)
    const key = `${parentKey}-${index}`
    const childTotal = measuredDurationMs(span.children ?? [])
    const children = span.children?.length
      ? buildStageItems(span.children, Math.max(durationMs, childTotal), isBottleneck, key)
      : []

    return {
      key,
      stage: info.label,
      label: info.label,
      description: info.description,
      durationMs,
      durationLabel,
      share,
      widthPercent,
      percentLabel: isActive ? formatPercent(share) : status === 'skipped' ? '—' : '0%',
      status,
      isBottleneck,
      toneIndex: index % 5,
      ...(span.inCount !== undefined ? { inCount: span.inCount } : {}),
      ...(span.outCount !== undefined ? { outCount: span.outCount } : {}),
      summaryBadges: buildSummaryBadges(span),
      ...(span.details ? { details: span.details } : {}),
      children
    }
  })
}

export function buildLatencyBreakdown(trace: LatencyTrace): LatencyBreakdown {
  const orderedSpans = mergeLatencySpans(trace.spans)
  const totalDurationMs = measuredDurationMs(orderedSpans)
  const stages = buildStageItems(orderedSpans, totalDurationMs, true)
  const barSegments = stages.filter((stage) => stage.status !== 'skipped' && stage.durationMs > 0)
  const bottleneckStage = stages.find((stage) => stage.isBottleneck)
  const bottleneckSubstage = bottleneckStage?.children.find((child) => child.isBottleneck)

  return {
    totalDurationMs,
    totalDurationLabel: formatDuration(totalDurationMs),
    ...(bottleneckStage
      ? {
          bottleneck: {
            stage: bottleneckStage.stage,
            label: bottleneckStage.label,
            durationMs: bottleneckStage.durationMs,
            durationLabel: formatDuration(bottleneckStage.durationMs),
            percentLabel: bottleneckStage.percentLabel,
            ...(bottleneckSubstage
              ? {
                  substageLabel: bottleneckSubstage.label,
                  substageDurationLabel: formatDuration(bottleneckSubstage.durationMs)
                }
              : {})
          }
        }
      : {}),
    barSegments,
    stages
  }
}

function formatDetailValue(value: LatencyDetail): string {
  if (typeof value === 'string' && /[\s,=]/.test(value)) return JSON.stringify(value)
  return String(value)
}

function formatSpan(span: LatencySpan, depth: number): string[] {
  const prefix = `${' '.repeat(1 + depth * 5)}-> `
  const label = getLatencyStageInfo(span.stage).label
  const timing = span.status === 'skipped'
    ? 'skipped'
    : span.status === 'error'
      ? `error after ${formatDuration(span.durationMs)}`
      : `actual ${formatDuration(span.durationMs)}`
  const counts = [
    span.inCount !== undefined ? `in=${span.inCount}` : undefined,
    span.outCount !== undefined ? `out=${span.outCount}` : undefined
  ].filter((value): value is string => Boolean(value))
  const details = Object.entries(span.details ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}=${formatDetailValue(value)}`)
  const suffix = [...counts, ...details]

  return [
    `${prefix}${label} (${timing})${suffix.length ? ` ${suffix.join(', ')}` : ''}`,
    ...(span.children ?? []).flatMap((child) => formatSpan(child, depth + 1))
  ]
}

export function formatLatencyTrace(trace: LatencyTrace, answerReadyMs?: number): string {
  const breakdown = buildLatencyBreakdown(trace)
  const summary = [`measured stages ${formatDuration(breakdown.totalDurationMs)}`]
  const wallTime = finiteNonNegative(answerReadyMs)
  if (wallTime !== undefined) summary.push(`answer ready in ${formatDuration(wallTime)}`)
  if (breakdown.bottleneck) {
    summary.push(`bottleneck ${breakdown.bottleneck.label} ${breakdown.bottleneck.durationLabel}`)
  }
  return [
    `Answer (${summary.join('; ')})`,
    ...mergeLatencySpans(trace.spans).flatMap((span) => formatSpan(span, 0))
  ].join('\n')
}

export interface LatencyPlan {
  text: string
  breakdown: LatencyBreakdown
}

/** Saved timing metadata is optional and must never prevent an answer from rendering. */
export function prepareLatencyPlan(trace?: LatencyTrace, answerReadyMs?: number): LatencyPlan | undefined {
  try {
    if (!trace) return undefined
    const breakdown = buildLatencyBreakdown(trace)
    if (!breakdown.stages.length) return undefined
    return { breakdown, text: formatLatencyTrace(trace, answerReadyMs) }
  } catch {
    return undefined
  }
}
