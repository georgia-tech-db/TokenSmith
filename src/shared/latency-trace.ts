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

export interface LatencyStageInfo {
  id: string
  label: string
  description: string
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

interface StageCatalogEntry extends LatencyStageInfo {
  aliases: ReadonlyArray<string>
  topLevelRank?: number
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

const stageCatalog: ReadonlyArray<StageCatalogEntry> = [
  {
    id: 'Question understanding',
    label: 'Question understanding',
    description:
      'Checks whether your question builds on earlier messages or a highlighted passage, turns follow-ups into a self-contained search query, and decides whether step-by-step reasoning is needed.',
    aliases: ['Rewriting', 'Question rewrite', 'Question routing'],
    topLevelRank: 0
  },
  {
    id: 'Library search',
    label: 'Library search',
    description:
      'Searches your selected course materials in the Library to find the passages most relevant to your question.',
    aliases: ['Retrieval', 'Source retrieval'],
    topLevelRank: 1
  },
  {
    id: 'Context packing',
    label: 'Context packing',
    description:
      'Selects and trims retrieved course passages so your question, evidence, and reserved answer space fit comfortably inside the model context budget.',
    aliases: ['Prompt preparation', 'Source packing'],
    topLevelRank: 2
  },
  {
    id: 'Answer generation',
    label: 'Answer generation',
    description:
      'Runs the AI model on your question and packed course passages to produce a source-grounded answer.',
    aliases: ['Generation'],
    topLevelRank: 3
  },
  {
    id: 'Follow-up suggestions',
    label: 'Follow-up suggestions',
    description:
      'Generates optional follow-up questions in the background after your answer appears so you can continue exploring the topic.',
    aliases: ['Suggestions'],
    topLevelRank: 4
  },
  {
    id: 'Query encoding',
    label: 'Query encoding',
    description:
      'Converts your search question into a numerical vector (embedding) so the library can match related concepts, not just exact words.',
    aliases: ['Query embedding']
  },
  {
    id: 'Semantic search',
    label: 'Semantic search',
    description:
      'Compares your encoded question vector against indexed course chunks to find passages with similar meaning.',
    aliases: ['Vector search']
  },
  {
    id: 'Keyword search',
    label: 'Keyword search',
    description:
      'Matches exact technical terms, formulas, and identifiers from your question against the course text index.',
    aliases: []
  },
  {
    id: 'Passage ranking',
    label: 'Passage ranking',
    description:
      'Combines semantic and keyword matches using reciprocal rank fusion and selects the strongest passages to cite.',
    aliases: ['Fusion and selection']
  },
  {
    id: 'Context check',
    label: 'Context check',
    description:
      'Runs a quick 1-token check with the local model to verify that the packed passages and question fit within the context window before generating the full answer.',
    aliases: ['Prompt preflight']
  },
  {
    id: 'Model loading',
    label: 'Model loading',
    description:
      'Loads or warms up the AI model weights in GPU or system memory before reading your prompt.',
    aliases: ['Model load']
  },
  {
    id: 'Reading context',
    label: 'Reading context',
    description:
      'Reads and processes your question, instructions, and packed course passages in the model before generating the first answer word.',
    aliases: ['Prompt evaluation']
  },
  {
    id: 'Writing answer',
    label: 'Writing answer',
    description:
      'Generates the final answer text token by token from the model.',
    aliases: ['Token generation']
  }
]

const stageLookup = new Map<string, StageCatalogEntry>()
for (const entry of stageCatalog) {
  stageLookup.set(entry.id.toLowerCase(), entry)
  stageLookup.set(entry.label.toLowerCase(), entry)
  for (const alias of entry.aliases) {
    stageLookup.set(alias.toLowerCase(), entry)
  }
}

const topLevelStageCount = stageCatalog.filter((entry) => entry.topLevelRank !== undefined).length

export function normalizeLatencyStageName(stage: string): string {
  const trimmed = typeof stage === 'string' ? stage.trim() : ''
  const matched = stageLookup.get(trimmed.toLowerCase())
  return matched ? matched.label : trimmed || 'Stage'
}

export function getLatencyStageInfo(stage: string): LatencyStageInfo {
  const trimmed = typeof stage === 'string' ? stage.trim() : ''
  const matched = stageLookup.get(trimmed.toLowerCase())
  if (matched) {
    return {
      id: matched.id,
      label: matched.label,
      description: matched.description
    }
  }
  const fallbackLabel = trimmed || 'Stage'
  return {
    id: fallbackLabel,
    label: fallbackLabel,
    description: 'Measures time spent in this step of the answer generation process.'
  }
}

export function listLatencyStageInfo(): LatencyStageInfo[] {
  return stageCatalog.map(({ id, label, description }) => ({ id, label, description }))
}

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
  if (!details || typeof details !== 'object' || Array.isArray(details)) return undefined

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

/** Validate persisted and cross-process metadata before merging it into a trace. */
function normalizeSpan(span: LatencySpan, depth = 0): LatencySpan | undefined {
  if (!span || typeof span !== 'object' || typeof span.stage !== 'string' || depth > 8) return undefined
  const durationMs = finiteNonNegative(span.durationMs)
  if (durationMs === undefined) return undefined
  const inCount = boundedCount(span.inCount)
  const outCount = boundedCount(span.outCount)
  const status = ['ok', 'error', 'skipped'].includes(span.status ?? '') ? span.status : undefined
  const details = sanitizeLatencyDetails(span.details)
  const children = Array.isArray(span.children)
    ? span.children.slice(0, 128).map((child) => normalizeSpan(child, depth + 1)).filter((child): child is LatencySpan => Boolean(child))
    : []
  return {
    stage: normalizeLatencyStageName(span.stage).slice(0, 128),
    startedAt: finiteNonNegative(span.startedAt) ?? 0,
    durationMs: status === 'skipped' ? 0 : durationMs,
    ...(inCount !== undefined ? { inCount } : {}),
    ...(outCount !== undefined ? { outCount } : {}),
    ...(status ? { status } : {}),
    ...(details ? { details } : {}),
    ...(children.length ? { children } : {})
  }
}

export function startLatencySpan(
  stage: string,
  fields: Omit<LatencySpanFields, 'status' | 'children'> = {}
): LatencyTimer {
  const normalizedStage = normalizeLatencyStageName(stage)
  const startedAt = Date.now()
  const started = performance.now()
  let completed: LatencySpan | undefined

  return {
    finish(finishFields = {}) {
      if (completed) return completed
      const durationMs = finishFields.status === 'skipped' ? 0 : Math.max(0, performance.now() - started)
      const inCount = boundedCount(finishFields.inCount ?? fields.inCount)
      const outCount = boundedCount(finishFields.outCount ?? fields.outCount)
      const details = sanitizeLatencyDetails({ ...fields.details, ...finishFields.details })
      const children = finishFields.children?.map((child) => normalizeSpan(child)).filter((child): child is LatencySpan => Boolean(child))
      completed = {
        stage: normalizedStage,
        startedAt,
        durationMs,
        ...(inCount !== undefined ? { inCount } : {}),
        ...(outCount !== undefined ? { outCount } : {}),
        ...(finishFields.status ? { status: finishFields.status } : {}),
        ...(details ? { details } : {}),
        ...(children?.length ? { children } : {})
      }
      return completed
    }
  }
}

function stageRank(stage: string): number {
  const matched = stageLookup.get(stage.trim().toLowerCase())
  return matched?.topLevelRank ?? topLevelStageCount
}

export function mergeLatencySpans(...groups: Array<ReadonlyArray<LatencySpan> | undefined>): LatencySpan[] {
  const byStage = new Map<string, { span: LatencySpan; sequence: number }>()
  let sequence = 0

  for (const group of groups) {
    for (const rawSpan of Array.isArray(group) ? group : []) {
      if (!rawSpan || typeof rawSpan !== 'object' || typeof rawSpan.stage !== 'string') continue
      const span = normalizeSpan(rawSpan)
      if (!span) continue
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
