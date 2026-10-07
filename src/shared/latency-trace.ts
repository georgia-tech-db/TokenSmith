export type LatencyDetail = string | number | boolean | null

export type LatencySpanStatus = 'ok' | 'error' | 'skipped'

export interface LatencySpan {
  stage: string
  startedAt: number
  durationMs: number
  inCount?: number
  outCount?: number
  status?: LatencySpanStatus
  details?: Record<string, LatencyDetail>
  children?: LatencySpan[]
}

export interface LatencyTrace {
  requestId?: string
  totalDurationMs: number
  spans: LatencySpan[]
}

interface LatencySpanFields {
  inCount?: number
  outCount?: number
  status?: LatencySpanStatus
  details?: Record<string, unknown>
  children?: LatencySpan[]
}

interface LatencyTimer {
  finish: (fields?: LatencySpanFields) => LatencySpan
}

const maxDetailCount = 24
const maxDetailKeyLength = 64
const maxDetailStringLength = 160
const privateDetailKeys = new Set([
  'answer',
  'apikey',
  'authorization',
  'excerpt',
  'headers',
  'messages',
  'prompt',
  'query',
  'rawpayload',
  'rawresponse',
  'rewrittenquery',
  'sourcetext',
  'systemmessage'
])
const canonicalStageOrder = [
  'Rewriting',
  'Retrieval',
  'Prompt preparation',
  'Generation',
  'Suggestions'
]

function finiteNonNegative(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

function boundedCount(value: unknown): number | undefined {
  const count = finiteNonNegative(value)
  return count === undefined ? undefined : Math.floor(count)
}

function sanitizeDetail(value: unknown): LatencyDetail | undefined {
  if (value === null || typeof value === 'boolean') return value
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined
  if (typeof value !== 'string') return undefined
  return value.length <= maxDetailStringLength
    ? value
    : `${value.slice(0, maxDetailStringLength - 1)}…`
}

export function sanitizeLatencyDetails(
  details?: Record<string, unknown>
): Record<string, LatencyDetail> | undefined {
  if (!details) return undefined

  const sanitized: Record<string, LatencyDetail> = {}
  const entries = Object.entries(details)
    .filter(([key]) => {
      const normalizedKey = key.replace(/[^a-z]/gi, '').toLowerCase()
      return key.length > 0
        && key.length <= maxDetailKeyLength
        && !privateDetailKeys.has(normalizedKey)
    })
    .sort(([left], [right]) => left.localeCompare(right))
    .slice(0, maxDetailCount)

  for (const [key, rawValue] of entries) {
    const value = sanitizeDetail(rawValue)
    if (value !== undefined) sanitized[key] = value
  }

  return Object.keys(sanitized).length > 0 ? sanitized : undefined
}

export function startLatencySpan(
  stage: string,
  fields: Omit<LatencySpanFields, 'status' | 'children'> = {}
): LatencyTimer {
  const startedAt = Date.now()
  const started = performance.now()
  let completed: LatencySpan | undefined

  return {
    finish(finishFields = {}) {
      if (completed) return completed
      const durationMs = Math.max(0, performance.now() - started)
      const inCount = boundedCount(finishFields.inCount ?? fields.inCount)
      const outCount = boundedCount(finishFields.outCount ?? fields.outCount)
      const details = sanitizeLatencyDetails({ ...fields.details, ...finishFields.details })
      completed = {
        stage,
        startedAt,
        durationMs,
        ...(inCount !== undefined ? { inCount } : {}),
        ...(outCount !== undefined ? { outCount } : {}),
        ...(finishFields.status ? { status: finishFields.status } : {}),
        ...(details ? { details } : {}),
        ...(finishFields.children?.length ? { children: finishFields.children } : {})
      }
      return completed
    }
  }
}

function stageRank(stage: string): number {
  const rank = canonicalStageOrder.indexOf(stage)
  return rank === -1 ? canonicalStageOrder.length : rank
}

export function mergeLatencySpans(...groups: Array<ReadonlyArray<LatencySpan> | undefined>): LatencySpan[] {
  const byStage = new Map<string, { span: LatencySpan; sequence: number }>()
  let sequence = 0

  for (const group of groups) {
    for (const span of group ?? []) {
      byStage.set(span.stage, { span, sequence })
      sequence += 1
    }
  }

  return [...byStage.values()]
    .sort((left, right) => {
      const rankDifference = stageRank(left.span.stage) - stageRank(right.span.stage)
      if (rankDifference !== 0) return rankDifference
      const startDifference = left.span.startedAt - right.span.startedAt
      return startDifference || left.sequence - right.sequence
    })
    .map(({ span }) => span)
}

export function measuredDurationMs(spans: ReadonlyArray<LatencySpan>): number {
  return spans.reduce((total, span) => total + (finiteNonNegative(span.durationMs) ?? 0), 0)
}

export function createLatencyTrace(
  spans: ReadonlyArray<LatencySpan>,
  requestId?: string
): LatencyTrace {
  const orderedSpans = mergeLatencySpans(spans)
  return {
    ...(requestId ? { requestId } : {}),
    totalDurationMs: measuredDurationMs(orderedSpans),
    spans: orderedSpans
  }
}

function formatDuration(durationMs: number): string {
  const safeDuration = finiteNonNegative(durationMs) ?? 0
  if (safeDuration >= 1000) return `${(safeDuration / 1000).toFixed(2)} s`
  if (safeDuration >= 100) return `${Math.round(safeDuration)} ms`
  if (safeDuration >= 10) return `${safeDuration.toFixed(1)} ms`
  return `${safeDuration.toFixed(2)} ms`
}

function formatDetailValue(value: LatencyDetail): string {
  if (typeof value === 'string' && /[\s,=]/.test(value)) return JSON.stringify(value)
  return String(value)
}

function formatSpan(span: LatencySpan, depth: number): string[] {
  const prefix = `${' '.repeat(1 + depth * 5)}-> `
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
    `${prefix}${span.stage} (${timing})${suffix.length ? ` ${suffix.join(', ')}` : ''}`,
    ...(span.children ?? []).flatMap((child) => formatSpan(child, depth + 1))
  ]
}

export function formatLatencyTrace(trace: LatencyTrace, wallTimeMs?: number): string {
  const summary = [`measured stages ${formatDuration(trace.totalDurationMs)}`]
  const wallTime = finiteNonNegative(wallTimeMs)
  if (wallTime !== undefined) summary.push(`wall time ${formatDuration(wallTime)}`)
  return [
    `Answer (${summary.join('; ')})`,
    ...trace.spans.flatMap((span) => formatSpan(span, 0))
  ].join('\n')
}

/**
 * Combine renderer-side spans with the engine's answer trace for one request.
 * Tracing is best-effort metadata, so any failure yields no trace rather than an error.
 */
export function combineLatencyTrace(
  requestId: string | undefined,
  clientSpans: ReadonlyArray<LatencySpan> | undefined,
  engineTrace?: LatencyTrace
): LatencyTrace | undefined {
  try {
    const engineMatches = engineTrace
      && (!engineTrace.requestId || !requestId || engineTrace.requestId === requestId)
    const spans = mergeLatencySpans(clientSpans, engineMatches ? engineTrace.spans : undefined)
    return spans.length > 0 ? createLatencyTrace(spans, requestId ?? engineTrace?.requestId) : undefined
  } catch {
    return undefined
  }
}
