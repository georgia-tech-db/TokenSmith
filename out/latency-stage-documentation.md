# Latency-stage changes for PR #156

## Purpose

This document records the implementation currently on `kausar/latency-log` for
PR [#156](https://github.com/georgia-tech-db/TokenSmith/pull/156). The change
adds a compact, persisted latency plan to study-chat answers. It is intended
to make request bottlenecks visible in the application without requiring
application logs or an external profiler.

The trace is best-effort metadata. A tracing, formatting, or follow-up
instrumentation failure must not prevent a successful answer from being
displayed.

## User-visible behavior

- An assistant answer with a trace shows an always-available **Latency plan**
  disclosure below the answer.
- The disclosure is a native, keyboard-accessible button with
  `aria-expanded`, a stable generated id, and a chevron state indicator.
- Expanded output is rendered as safe, wrapping monospace text. Trace details
  are formatted as text rather than injected as HTML.
- The panel reports measured stage time separately from the existing wall-clock
  response time (`responseDurationMs`).
- The panel can show partial plans. For example, an answer may appear with
  generation data before follow-up suggestions finish; the same message is
  subsequently updated with the completed suggestions span.
- Clarification answers and “explain more simply” answers show only the stages
  that actually ran. They do not receive fabricated rewrite or retrieval
  stages.
- Existing saved messages without `latencyTrace` continue to render normally,
  without an empty disclosure.
- The existing Sources disclosure and follow-up controls remain independent of
  the latency panel.

## Trace model and safety rules

`src/shared/latency-trace.ts` defines the shared JSON-safe contract:

- `LatencySpan`: stage name, epoch `startedAt`, measured `durationMs`,
  optional input/output counts, status (`ok`, `error`, or `skipped`), bounded
  scalar details, and nested children.
- `LatencyTrace`: optional `requestId`, measured top-level
  `totalDurationMs`, and ordered top-level spans.
- Timing helpers use `performance.now()` in TypeScript and
  `time.perf_counter()` in Python. Epoch timestamps are used only to order
  stages across process boundaries; duration calculations never subtract epoch
  clocks.
- Trace construction clamps invalid negative durations and counts, sanitizes
  details, bounds strings and collections, drops unsupported values, and
  preserves deterministic stage order.
- The formatter and breakdown helpers support nested spans, skipped stages,
  unavailable metrics, errors, rounding, and partial traces.
- Trace data contains only operational metadata: identifiers, counts,
  timings, provider/model names, modes, budgets, finish reasons, and token
  counts. It does not contain prompts, rewritten questions, answers, source
  excerpts, system messages, credentials, headers, or raw provider payloads.

## End-to-end request instrumentation

### Rewriting and retrieval

`src/shared/study-chat-pipeline.ts` now returns the rewrite/retrieval spans
alongside the existing resolution and duration values. This retains
compatibility for callers that still consume `rewriteMs` and `searchMs`.

The retrieval plan is measured across the TypeScript/Python boundary:

- `src/main/python/python-engine-service.ts` measures embedding-device setup
  and the complete library call.
- `python_engine/tokensmith_engine.py` records only the operations that apply
  to the selected search mode:
  - query encoding;
  - vector search;
  - keyword preparation and keyword search;
  - fusion and selection;
  - source fetch/expansion and final selection where applicable.
- The Python result returns safe latency spans and counts with the sources.
  The renderer-side Retrieval span contains those children without
  reconstructing a plan from log messages or double-counting child duration.
- Empty-library and no-match outcomes retain safe reason details.
- Vector, keyword, and hybrid searches display different applicable child
  stages; irrelevant children are omitted.

### Prompt/context preparation

The prompt-preparation stage is represented as **Context packing** in the
current implementation. It reports measured packing time and safe packing
metadata, including retrieved/included/truncated source counts, estimated
prompt tokens, source-budget usage, context limits, answer reserve, and
safety margin. The stage is measured at the engine call site rather than
reconstructed from renderer wall time.

### Answer generation

Both Ollama and remote OpenAI-compatible providers produce optional engine
traces:

- Ollama response metadata is captured for model load, prompt evaluation,
  token generation, total provider duration, prompt-token count, and
  generated-token count. Ollama nanoseconds are converted to milliseconds
  exactly once.
- Remote providers expose prompt/completion usage and finish reason when the
  provider returns them. Missing provider metrics remain unavailable instead
  of being estimated.
- Gemma one-token verification probes are represented as bounded
  **Prompt preflight** children, and the final answer request is not counted
  twice.
- Answer generation and follow-up suggestion generation are measured
  independently. A follow-up error is marked on the Suggestions span while
  preserving the successful answer.
- Existing engine log events receive the request id where relevant, allowing
  correlation without matching prompt text.

## Persistence and lifecycle

The trace is carried through the existing engine and bridge contracts and is
stored on `ChatMessage`. The simpler-explanation persisted variant also
accepts an optional trace.

The normal lifecycle is:

1. The renderer creates a request id.
2. Rewrite and retrieval return client-side spans.
3. The answer callback publishes a partial trace containing the stages
   available at answer time.
4. The final engine response contributes suggestions and any provider
   metadata.
5. `combineLatencyTrace` merges spans by request id and replaces the same
   assistant message with the completed trace.
6. Saving and reloading the conversation restores the compact trace.

The existing `responseDurationMs` remains the wall-clock value for
compatibility. The trace total is computed only from measured stages and
does not include the intentional renderer search-status delay.

## Files changed

### Runtime and contracts

- `src/shared/latency-trace.ts` — shared types, sanitization, merge,
  breakdown, formatting, and safe timing helpers.
- `src/shared/study-chat-pipeline.ts` — returns rewrite/retrieval spans and
  combines the cross-process retrieval result.
- `src/shared/engine.ts` — adds request ids and optional engine traces to
  chat request/response contracts.
- `src/shared/app-state.ts` — persists optional traces on assistant messages
  and simpler-explanation messages.
- `src/main/python/python-engine-service.ts` — measures the Python library
  call and attaches retrieval children.
- `python_engine/tokensmith_engine.py` — measures retrieval operations and
  returns `latencySpans`.
- `src/main/engine/study-engine-latency.ts` — builds prompt, generation,
  suggestion, and provider metadata spans.
- `src/main/engine/ollama-service.ts` — captures Ollama phase/tokens,
  Gemma preflight, answer generation, and suggestions.
- `src/main/engine/remote-chat-service.ts` — captures remote usage,
  finish-reason, answer, and suggestion spans.
- `src/renderer/src/App.tsx` — creates request ids, merges partial/final
  traces, persists them with answers, and renders the panel.

### UI

- `src/renderer/src/QueryPlanPanel.tsx` — expandable latency-plan component,
  safe text formatting, measured/wall-time breakdown, and graceful fallback
  behavior.
- `src/renderer/src/styles.css` — panel layout, typography, spacing, nested
  rows, status styling, wrapping, and responsive behavior.

### Tests and fixtures

- `tests/unit/typescript/latency-trace.test.mjs` — trace sanitization,
  merging, formatting, breakdown, stage ordering, and partial/error cases.
- `tests/unit/typescript/gemma-answer-budget.test.mjs` —
  preflight/generation accounting.
- `tests/unit/typescript/generation-events.test.mjs` — answer and follow-up
  event behavior.
- `tests/unit/typescript/question-rewrite.test.mjs` — rewrite trace changes.
- `tests/unit/typescript/remote-chat-service.test.mjs` — remote metadata and
  trace behavior.
- `tests/python/test_tokensmith_engine.py` — retrieval span names and
  returned metadata.
- `tests/ui/reasoning-retry.jsx` — UI fixture coverage for traced reasoning
  and retry behavior.

The test updates also normalize expected stage names and update fixtures for
the richer response shape. No production retrieval or answer semantics are
intended to change.

## Validation and follow-up

The intended validation commands for this branch are:

```sh
npm run test:unit:ts
npm run test:unit:python
npm run typecheck
npm run test:integration
npm run build
```

Manual validation should cover one Ollama chat, one remote-provider chat,
reload persistence, a clarification, and an “explain more simply” response.
Streaming TTFT, cold/warm comparisons, benchmark aggregation, and quiz/starter
question tracing remain outside this PR because the current generation path is
non-streaming or uses separate flows.
