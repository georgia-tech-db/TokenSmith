import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  createLatencyTrace,
  formatLatencyTrace,
  measuredDurationMs,
  mergeLatencySpans,
  sanitizeLatencyDetails,
  startLatencySpan
} = requireTranspiledTs('src/shared/latency-trace.ts')

const span = (stage, durationMs, fields = {}) => ({
  stage,
  startedAt: 1_700_000_000_000,
  durationMs,
  ...fields
})

test('formats a stable EXPLAIN-style latency artifact', () => {
  const trace = createLatencyTrace([
    span('Suggestions', 900, { outCount: 3 }),
    span('Retrieval', 700, {
      outCount: 12,
      details: { sources: 12, mode: 'hybrid' },
      children: [
        span('Query embedding', 50, { details: { models: 1 } }),
        span('Vector search', 300, { outCount: 20 }),
        span('Keyword search', 250, { outCount: 20 }),
        span('Fusion and selection', 100, { inCount: 32, outCount: 12 })
      ]
    }),
    span('Rewriting', 150, { details: { history_turns: 3 } }),
    span('Prompt preparation', 20, {
      inCount: 12,
      outCount: 6,
      details: { estimated_prompt_tokens: 2840, context_tokens: 4096 }
    }),
    span('Generation', 5030, {
      details: { prompt_tokens: 2840, generated_tokens: 310 },
      children: [
        span('Model load', 400),
        span('Prompt evaluation', 600),
        span('Token generation', 3900)
      ]
    })
  ], 'generation-1')

  assert.equal(trace.totalDurationMs, 6800)
  assert.equal(formatLatencyTrace(trace, 7300), [
    'Answer (measured stages 6.80 s; wall time 7.30 s)',
    ' -> Rewriting (actual 150 ms) history_turns=3',
    ' -> Retrieval (actual 700 ms) out=12, mode=hybrid, sources=12',
    '      -> Query embedding (actual 50.0 ms) models=1',
    '      -> Vector search (actual 300 ms) out=20',
    '      -> Keyword search (actual 250 ms) out=20',
    '      -> Fusion and selection (actual 100 ms) in=32, out=12',
    ' -> Prompt preparation (actual 20.0 ms) in=12, out=6, context_tokens=4096, estimated_prompt_tokens=2840',
    ' -> Generation (actual 5.03 s) generated_tokens=310, prompt_tokens=2840',
    '      -> Model load (actual 400 ms)',
    '      -> Prompt evaluation (actual 600 ms)',
    '      -> Token generation (actual 3.90 s)',
    ' -> Suggestions (actual 900 ms) out=3'
  ].join('\n'))
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
    'Answer (measured stages 12.7 ms)',
    ' -> Rewriting (skipped)',
    ' -> Generation (actual 0.40 ms)',
    ' -> Suggestions (error after 12.3 ms) reason="provider unavailable"'
  ].join('\n'))
  assert.doesNotMatch(formatLatencyTrace(trace), /ttft|undefined|NaN/)
})

test('merges partial and final spans in pipeline order with later stages replacing earlier ones', () => {
  const partialGeneration = span('Generation', 1000, { details: { phase: 'partial' } })
  const finalGeneration = span('Generation', 1200, { details: { phase: 'final' } })
  const merged = mergeLatencySpans(
    [span('Retrieval', 20), span('Rewriting', 10)],
    [partialGeneration],
    [span('Suggestions', 50), finalGeneration]
  )

  assert.deepEqual(merged.map(({ stage }) => stage), [
    'Rewriting', 'Retrieval', 'Generation', 'Suggestions'
  ])
  assert.equal(merged.find(({ stage }) => stage === 'Generation'), finalGeneration)
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

  assert.equal(completed.stage, 'Retrieval')
  assert.equal(completed.inCount, 3)
  assert.equal(completed.outCount, 2)
  assert.ok(completed.durationMs >= 0)
  assert.deepEqual(completed.details, { candidates: 12, mode: 'hybrid' })
  assert.equal(timer.finish({ status: 'error' }), completed)
})

test('measured total ignores invalid durations', () => {
  assert.equal(measuredDurationMs([
    span('Rewriting', 10),
    span('Retrieval', Number.NaN),
    span('Generation', -5)
  ]), 10)
})
