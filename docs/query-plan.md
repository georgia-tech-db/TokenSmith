# Study chat query plans

Answers with timing metadata offer **Show query plan** immediately after
**Explain more simply** in the answer action row. The native button toggles
**Hide query plan**, exposes its state to assistive technology, and opens the
plan for the currently selected Basic, Simpler explanation, or Reasoning answer.
Older answers without a trace have no query-plan action.

## Reading the plan

- **Answer ready in** is the existing elapsed response time up to the answer
  appearing. It includes time outside instrumented stages.
- **Measured stages** sums the recorded top-level durations, including follow-up
  suggestions that may finish after the answer appears. Child durations are
  breakdowns of their parent; they are never added to the total again.
- The largest measured stage is marked **Bottleneck**. Proportional bars show
  measured work, not an absolute timeline. Provider timings may leave some
  parent time unattributed.
- Stage guides describe the work. The raw execution trace is optional text,
  rendered without HTML interpretation.
- An answer can show a partial plan while suggestions are still running. The
  final response updates that same answer and its persisted trace. Failed
  suggestions retain the successful answer and record an error stage.

These are application pipeline timings, not a database optimizer's SQL plan.
No time to first token is reported: the current answer path is non-streaming.

## Implementation

`src/shared/latency-trace.ts` owns the JSON contract, timers, normalization,
sanitization, stage catalog, and merging. Spans contain a stage, epoch start,
monotonic duration, optional counts/status/bounded scalar details, and children.
Skipped work contributes zero time. Invalid spans are discarded and nested
metadata is bounded. Instrumentation callers supply only operational metadata,
never prompts, passages, answers, credentials, or complete provider payloads.

`src/renderer/src/latency-plan.ts` derives labels, proportions, stage identities,
and text from validated spans. Rendering treats missing or malformed traces as
optional metadata; an unusable plan cannot prevent the answer from displaying.
`AnswerExplanation.tsx` owns the toggle and selects the displayed answer's trace.
`QueryPlanPanel.tsx` renders the plan with styles in `query-plan.css`.

The measured stages are:

1. **Question understanding**: renderer question rewriting/routing when needed.
2. **Library search**: the renderer's complete search call. The Python worker
   supplies query encoding, semantic search, keyword search, and passage ranking
   children for the applicable retrieval mode.
3. **Context packing**: source selection and context budgeting in the engine.
4. **Answer generation**: provider completion, including local context checks.
   Ollama supplies loading, prompt evaluation, and token-generation durations;
   remote providers supply token usage when available. Missing metrics are not
   estimated.
5. **Follow-up suggestions**: optional work after the answer is available.

Both normal answers and answer variants carry their own trace through the engine,
bridge, and saved message. Clarification and simpler answers display only the
stages that ran. Request IDs prevent mixing another request's engine trace into
an answer. Practice consumes the richer library-search result's `sources` field;
practice generation itself is not instrumented by this feature.

## Reproduce the checks

```sh
npm run build
npm run test:unit:ts
npm run test:unit:python
npm run test:integration
```

For the deterministic browser regression, start the fixture server:

```sh
npx vite --config tests/ui/vite.config.mjs --host 127.0.0.1 --port 5196
node tests/ui/query-plan-check.mjs
```

Set `PLAYWRIGHT_MODULE` and `CHROME_PATH` if Playwright and Chrome are installed
externally. `QUERY_PLAN_UI_URL` can override the preview origin. The check covers
button order, keyboard toggling, late/partial traces, malformed and legacy saved
messages, answer variants, persistence, and desktop/mobile layout. Fixtures use
no real model calls or user data.
