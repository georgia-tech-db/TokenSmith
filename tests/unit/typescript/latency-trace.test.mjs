import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  combineLatencyTrace,
  createLatencyTrace,
  getLatencyStageInfo,
  listLatencyStageInfo,
  measuredDurationMs,
  mergeLatencySpans,
  normalizeLatencyStageName,
  sanitizeLatencyDetails,
  startLatencySpan
} = requireTranspiledTs('src/shared/latency-trace.ts')

const { buildLatencyBreakdown, formatLatencyTrace, prepareLatencyPlan } = requireTranspiledTs('src/renderer/src/latency-plan.ts')

const span = (stage, durationMs, fields = {}) => ({
  stage,
  startedAt: 1_700_000_000_000,
  durationMs,
  ...fields
})

test('formats a stable EXPLAIN-style latency artifact with rebranded stages and bottleneck summary', () => {
  const trace = createLatencyTrace([
    span('Follow-up suggestions', 900, { outCount: 3 }),
    span('Library search', 700, {
      outCount: 12,
      details: { sources: 12, mode: 'hybrid' },
      children: [
        span('Query encoding', 50, { details: { models: 1 } }),
        span('Semantic search', 300, { outCount: 20 }),
        span('Keyword search', 250, { outCount: 20 }),
        span('Passage ranking', 100, { inCount: 32, outCount: 12 })
      ]
    }),
    span('Question understanding', 150, { details: { history_turns: 3 } }),
    span('Context packing', 20, {
      inCount: 12,
      outCount: 6,
      details: { estimated_prompt_tokens: 2840, context_tokens: 4096 }
    }),
    span('Answer generation', 5030, {
      details: { prompt_tokens: 2840, generated_tokens: 310 },
      children: [
        span('Context check', 130, { inCount: 6, outCount: 6, details: { fits: true, prompt_tokens: 2840 } }),
        span('Model loading', 400),
        span('Reading context', 600),
        span('Writing answer', 3900)
      ]
    })
  ], 'generation-1')

  assert.equal(trace.totalDurationMs, 6800)
  assert.equal(formatLatencyTrace(trace, 7300), [
    'Answer (measured stages 6.80 s; answer ready in 7.30 s; bottleneck Answer generation 5.03 s)',
    ' -> Question understanding (actual 150 ms) history_turns=3',
    ' -> Library search (actual 700 ms) out=12, mode=hybrid, sources=12',
    '      -> Query encoding (actual 50.0 ms) models=1',
    '      -> Semantic search (actual 300 ms) out=20',
    '      -> Keyword search (actual 250 ms) out=20',
    '      -> Passage ranking (actual 100 ms) in=32, out=12',
    ' -> Context packing (actual 20.0 ms) in=12, out=6, context_tokens=4096, estimated_prompt_tokens=2840',
    ' -> Answer generation (actual 5.03 s) generated_tokens=310, prompt_tokens=2840',
    '      -> Context check (actual 130 ms) in=6, out=6, fits=true, prompt_tokens=2840',
    '      -> Model loading (actual 400 ms)',
    '      -> Reading context (actual 600 ms)',
    '      -> Writing answer (actual 3.90 s)',
    ' -> Follow-up suggestions (actual 900 ms) out=3'
  ].join('\n'))
})

test('normalizes legacy stage names such as Prompt preflight to intuitive rebranded names and descriptions', () => {
  assert.equal(normalizeLatencyStageName('Prompt preflight'), 'Context check')
  assert.equal(normalizeLatencyStageName('Prompt preparation'), 'Context packing')
  assert.equal(normalizeLatencyStageName('Rewriting'), 'Question understanding')
  assert.equal(normalizeLatencyStageName('Retrieval'), 'Library search')
  assert.equal(normalizeLatencyStageName('Generation'), 'Answer generation')
  assert.equal(normalizeLatencyStageName('Suggestions'), 'Follow-up suggestions')
  assert.equal(normalizeLatencyStageName('Query embedding'), 'Query encoding')
  assert.equal(normalizeLatencyStageName('Vector search'), 'Semantic search')
  assert.equal(normalizeLatencyStageName('Fusion and selection'), 'Passage ranking')
  assert.equal(normalizeLatencyStageName('Model load'), 'Model loading')
  assert.equal(normalizeLatencyStageName('Prompt evaluation'), 'Reading context')
  assert.equal(normalizeLatencyStageName('Token generation'), 'Writing answer')

  const preflightInfo = getLatencyStageInfo('Prompt preflight')
  assert.equal(preflightInfo.label, 'Context check')
  assert.match(preflightInfo.description, /context window/i)

  const catalog = listLatencyStageInfo()
  assert.ok(catalog.length >= 13)
  for (const entry of catalog) {
    assert.ok(entry.label.length > 0)
    assert.ok(entry.description.length > 20)
  }
})

test('builds horizontal stacked bar segments and marks the slowest stage and sub-stage as the bottleneck', () => {
  const trace = createLatencyTrace([
    span('Rewriting', 0, { status: 'skipped' }),
    span('Retrieval', 800, {
      outCount: 6,
      children: [
        span('Query embedding', 100),
        span('Vector search', 500),
        span('Keyword search', 200)
      ]
    }),
    span('Prompt preparation', 40, { inCount: 6, outCount: 4 }),
    span('Generation', 3160, {
      details: { prompt_tokens: 1800, generated_tokens: 240 },
      children: [
        span('Prompt preflight', 260, { inCount: 4, outCount: 4, details: { fits: true } }),
        span('Model load', 100),
        span('Prompt evaluation', 400),
        span('Token generation', 2400)
      ]
    }),
    span('Suggestions', 1000, { outCount: 3 })
  ])

  const breakdown = buildLatencyBreakdown(trace)
  assert.equal(breakdown.totalDurationMs, 5000)
  assert.equal(breakdown.barSegments.length, 4, 'Skipped stages are omitted from the stacked bar')
  assert.deepEqual(
    breakdown.barSegments.map((segment) => segment.label),
    ['Library search', 'Context packing', 'Answer generation', 'Follow-up suggestions']
  )
  assert.equal(breakdown.bottleneck.label, 'Answer generation')
  assert.equal(breakdown.bottleneck.percentLabel, '63%')
  assert.equal(breakdown.bottleneck.substageLabel, 'Writing answer')

  const bottleneckSegments = breakdown.barSegments.filter((segment) => segment.isBottleneck)
  assert.equal(bottleneckSegments.length, 1)
  assert.equal(bottleneckSegments[0].label, 'Answer generation')

  const generationStage = breakdown.stages.find((stage) => stage.label === 'Answer generation')
  assert.equal(generationStage.children[0].label, 'Context check')
  assert.equal(generationStage.children.find((child) => child.isBottleneck).label, 'Writing answer')

  const retrievalStage = breakdown.stages.find((stage) => stage.label === 'Library search')
  assert.equal(
    retrievalStage.children.some((child) => child.isBottleneck),
    false,
    'Non-bottleneck parent stages do not mark sub-stages as overall bottlenecks'
  )
})

test('formats skipped, failed, sub-millisecond, and unavailable stages honestly', () => {
  const trace = createLatencyTrace([
    span('Rewriting', 0, { status: 'skipped' }),
    span('Generation', 0.4),
    span('Suggestions', 12.25, {
      status: 'error',
      details: { reason: 'provider unavailable' }
    })
  ])

  assert.equal(formatLatencyTrace(trace), [
    'Answer (measured stages 12.7 ms; bottleneck Follow-up suggestions 12.3 ms)',
    ' -> Question understanding (skipped)',
    ' -> Answer generation (actual 0.40 ms)',
    ' -> Follow-up suggestions (error after 12.3 ms) reason="provider unavailable"'
  ].join('\n'))
  assert.doesNotMatch(formatLatencyTrace(trace), /ttft|undefined|NaN/)
})

test('merges partial and final spans in pipeline order with later stages replacing earlier ones', () => {
  const partialGeneration = span('Generation', 1000, { details: { phase: 'partial' } })
  const finalGeneration = span('Answer generation', 1200, { details: { phase: 'final' } })
  const merged = mergeLatencySpans(
    [span('Retrieval', 20), span('Rewriting', 10)],
    [partialGeneration],
    [span('Suggestions', 50), finalGeneration]
  )

  assert.deepEqual(merged.map(({ stage }) => stage), [
    'Question understanding', 'Library search', 'Answer generation', 'Follow-up suggestions'
  ])
  assert.deepEqual(merged.find(({ stage }) => stage === 'Answer generation').details, { phase: 'final' })
})

test('sanitizes detail values, ordering, and size without serializing nested content', () => {
  const longValue = 'x'.repeat(300)
  const details = sanitizeLatencyDetails({
    z: true,
    nested: { source: 'private text' },
    invalid: Number.POSITIVE_INFINITY,
    long: longValue,
    count: 4,
    empty: null,
    prompt: 'private question',
    system_message: 'private instructions',
    apiKey: 'private key',
    prompt_tokens: 2800
  })

  assert.deepEqual(Object.keys(details), ['count', 'empty', 'long', 'prompt_tokens', 'z'])
  assert.equal(details.long.length, 160)
  assert.match(details.long, /…$/)
  assert.doesNotMatch(JSON.stringify(details), /private text|private question|private instructions|private key|Infinity/)

  const manyDetails = Object.fromEntries(
    Array.from({ length: 40 }, (_, index) => [`field_${String(index).padStart(2, '0')}`, index])
  )
  assert.equal(Object.keys(sanitizeLatencyDetails(manyDetails)).length, 24)
})

test('timer captures safe counts and returns the same completed span on repeated finish calls', () => {
  const timer = startLatencySpan('Retrieval', {
    inCount: 3.8,
    details: { mode: 'hybrid', unsupported: ['private'] }
  })
  const completed = timer.finish({
    outCount: 2.9,
    details: { candidates: 12 }
  })

  assert.equal(completed.stage, 'Library search')
  assert.equal(completed.inCount, 3)
  assert.equal(completed.outCount, 2)
  assert.ok(completed.durationMs >= 0)
  assert.deepEqual(completed.details, { candidates: 12, mode: 'hybrid' })
  assert.equal(timer.finish({ status: 'error' }), completed)
})

test('measured total ignores invalid durations', () => {
  assert.equal(measuredDurationMs([
    span('Question understanding', 10),
    span('Library search', Number.NaN),
    span('Answer generation', -5)
  ]), 10)
})

test('combines renderer spans with the engine trace for one request', () => {
  const client = [span('Library search', 700), span('Question understanding', 150)]
  const partial = { requestId: 'r1', totalDurationMs: 5050, spans: [span('Context packing', 20), span('Answer generation', 5030)] }
  const final = { ...partial, spans: [...partial.spans, span('Follow-up suggestions', 900)] }

  const early = combineLatencyTrace('r1', client, partial)
  assert.deepEqual(early.spans.map(s => s.stage), ['Question understanding', 'Library search', 'Context packing', 'Answer generation'])
  assert.equal(early.requestId, 'r1')
  assert.equal(early.totalDurationMs, 150 + 700 + 20 + 5030)

  const done = combineLatencyTrace('r1', client, final)
  assert.deepEqual(done.spans.map(s => s.stage).at(-1), 'Follow-up suggestions')
  assert.equal(done.spans.filter(s => s.stage === 'Answer generation').length, 1)
})

test('builds partial plans and ignores mismatched or missing engine traces', () => {
  const rewrite = [span('Question understanding', 120)]
  assert.deepEqual(combineLatencyTrace('r1', rewrite).spans.map(s => s.stage), ['Question understanding'])
  const engineOnly = combineLatencyTrace('r1', undefined, {
    requestId: 'r1', totalDurationMs: 30, spans: [span('Context packing', 10), span('Answer generation', 20)]
  })
  assert.deepEqual(engineOnly.spans.map(s => s.stage), ['Context packing', 'Answer generation'])
  const stale = combineLatencyTrace('r2', rewrite, { requestId: 'r1', totalDurationMs: 5, spans: [span('Answer generation', 5)] })
  assert.deepEqual(stale.spans.map(s => s.stage), ['Question understanding'])
  assert.equal(combineLatencyTrace('r1', undefined, undefined), undefined)
  assert.equal(combineLatencyTrace('r1', [null]), undefined)
})

test('traced messages round-trip through JSON and old messages need no trace', () => {
  const trace = combineLatencyTrace('r1', [span('Question understanding', 10, { details: { mode: 'standalone' } })])
  const saved = JSON.parse(JSON.stringify({ id: 'a1', role: 'assistant', text: 'x', latencyTrace: trace }))
  assert.equal(formatLatencyTrace(saved.latencyTrace, 1234), formatLatencyTrace(trace, 1234))
  assert.equal(JSON.parse(JSON.stringify({ id: 'a0', role: 'assistant', text: 'old' })).latencyTrace, undefined)
})


test('saved or cross-process malformed metadata cannot break the answer plan', () => {
  for (const trace of [undefined, null, {}, { spans: null }, { spans: 'bad' }, { spans: [null, {}] }]) {
    assert.equal(prepareLatencyPlan(trace), undefined)
  }
  const plan = prepareLatencyPlan({ totalDurationMs: 999999, spans: [
    null, span('Generation', -10), span('Generation', 40, {
      startedAt: NaN, inCount: -1, outCount: 2.9, status: 'unexpected',
      details: { prompt: 'private', prompt_tokens: 12 },
      children: [null, span('Model load', Infinity), span('Token generation', 30)]
    }), span('Suggestions', 100, { status: 'skipped' })
  ] }, 50)
  assert.equal(plan.breakdown.totalDurationMs, 40, 'Total is recomputed from valid measured stages')
  assert.equal(plan.breakdown.stages[0].outCount, 2)
  assert.equal(plan.breakdown.stages[0].children.length, 1)
  assert.doesNotMatch(plan.text, /private|NaN|Infinity|undefined/)
  assert.match(plan.text, /answer ready in 50.0 ms/)
})

test('nested stage identities are unique even when stage names repeat across parents', () => {
  const trace = createLatencyTrace([
    span('Library search', 10, { children: [span('Repeated', 5, { children: [span('Repeated', 2)] })] }),
    span('Answer generation', 20, { children: [span('Repeated', 10)] })
  ])
  const keys = stages => stages.flatMap(stage => [stage.key, ...keys(stage.children)])
  const all = keys(buildLatencyBreakdown(trace).stages)
  assert.equal(new Set(all).size, all.length)
})

test('skipped work contributes no elapsed time', () => {
  const skipped = startLatencySpan('Suggestions').finish({ status: 'skipped' })
  assert.equal(skipped.durationMs, 0)
  assert.equal(createLatencyTrace([span('Generation', 30), skipped]).totalDurationMs, 30)
})
