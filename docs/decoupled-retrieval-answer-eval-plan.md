# Plan: Decouple Retrieval From the Answer-Quality Pipeline

## 1. Objective

Split the retrieval step out of the answer-quality (Jev) pipeline so that it is
produced once by the retrieval benchmark and reused for generation. Concretely:

1. A **new retrieval step** (reusing the engine's `search_library` core, the same
   core the BuzzDB benchmark exercises) indexes the corpus and retrieves chunks
   per case, emits a **pass/fail** gate, **and** writes the retrieved chunks to a
   shared file.
2. **Answer-quality pipeline** consumes those chunks directly for generation and
   Jev grading, **skipping** the Gemini embedding/index/cache path entirely.

Goal, in the requester's words: *"use retrieved chunks from the benchmark for
answer generation ... avoid the caching and Gemini API call."*

> **This plan is supplemental, not a replacement.** The existing BuzzDB retrieval
> benchmark (`tests/benchmarks/test_buzzdb_embeddings.py`,
> `test_buzzdb_retrieval.mjs`, `buzzdb_grounding_cases.json`) and its per-PR CI
> gate in `.github/workflows/accuracy.yml` remain **exactly as they are** and keep
> running unchanged. Everything described here is purely additive: new scripts and
> a new flag that live alongside the current tests. No existing test, fixture,
> dataset, or workflow is modified or removed.

```
Step 1 (retrieval)
  index corpus (Ollama nomic-embed-text)
    -> engine.search_library(query) per case
      -> weak gate (sources non-empty, reason == null, vector hit >= 1)  -> PASS / FAIL
      -> emit full `sources` per case -> retrieved-sources.json

Step 2 (generation + grading)
  read retrieved-sources.json
    -> map sources -> ChatSource[] -> request.retrievedSources
      -> studyChatMessages(request) -> OpenRouter (Nemotron 3 Super)
        -> Jev grade vs ideal_answer -> pass-rate gate
```

## 2. Why This Is Feasible

Both pipelines already converge on the same artifact: a ranked list of
retrieved **sources** (chunks). The split point is where the retrieval benchmark
already has the full search result in hand.

### Verified contracts (read from source)

- **Retrieval core**: `engine.search_library(payload)` in
  `python_engine/tokensmith_engine.py` returns
  `{ "sources": [...], "reason", "queryTerms", "keywordTerms" }`. Each source is
  produced by `source_from_sqlite_chunk` and already carries `context`,
  `sectionHeader`, `sourceChunkIds`, `tokensmithChunkId`, title/collection, and
  rank metadata.
- **Benchmark already captures sources**: in
  `tests/benchmarks/test_buzzdb_embeddings.py`, each case result builds an
  `evidence` list from `response["sources"]`. The full `sources` are available at
  that point; the report currently slims them for display only.
- **Generator input**: `studyChatMessages(request)` in
  `src/main/engine/study-chat-format.ts` reads `request.retrievedSources`
  (type `ChatSource[]`) and packs them via `sourceContext`/`packSourceContext`.
- **`ChatSource` shape** (`src/shared/app-state.ts`): the prompt builder only
  needs the chunk text (`context` or `excerpt`) plus optional display fields
  (`title`/`documentTitle`/`collectionName`, `path`, `sectionHeader`). See
  `sourcePrefix` and `sourceText` in `study-chat-format.ts`.

Because the generator consumes `ChatSource[]` and the benchmark produces
`search_library` sources, the only glue required is a field mapping plus a shared
file contract.

## 3. The One Constraint: Same Dataset + Same Corpus

Today the two pipelines use **different corpora**:

| | Corpus | Cases |
| :-- | :-- | :-- |
| Retrieval benchmark | BuzzDB book Markdown fixture (`tests/fixtures/buzzdb/buzzdb-book.tokensmith.md`) | `tests/benchmarks/buzzdb_grounding_cases.json` |
| Answer-quality (Jev) | `tests/dataset/A Distributed Object Model for the Java System.pdf` | `tests/dataset/DistributedObjectModel.json` |

Retrieved chunks only make sense for the dataset they came from, so the retrieval
step and the generation step **must run over the same corpus + cases**.

**Chosen approach (Option A):** run retrieval over the Jev dataset's corpus (the
PDF) using Ollama `nomic-embed-text` instead of Gemini. This keeps the existing
`DistributedObjectModel.json` cases and reference answers, drops the Gemini API
call, and removes the index/cache step from the answer pipeline — exactly the
stated goal. Because this corpus has no retrieval ground truth, Step 1 uses a
**weak gate** (see Decision 3 and Section 4a).

Alternative (Option B, not chosen): keep the BuzzDB corpus and grade its
`referenceAnswer` fields with Jev. Viable (grounding cases already carry
`referenceAnswer`), but it evaluates a different document than the current Jev
target, so it does not satisfy "reuse for the answer-quality pipeline" as-is.

## 4. Decisions

1. **Query rewrite is dropped for this flow.** `run-eval-pipeline.mjs` currently
   rewrites the query via `prepareRewrittenStudyChat` (`resolveRemoteChatQuestion`)
   before searching; the benchmark searches the raw question. Moving retrieval
   into the benchmark means retrieval uses the raw benchmark question. For
   single-turn benchmark questions this is acceptable and more reproducible. This
   is a deliberate behavior change and is called out here.
2. **Do not mutate the committed BuzzDB benchmark gate.** Add a dedicated
   retrieval emitter that reuses `search_library`, rather than overloading
   `test_buzzdb_embeddings.py`. This keeps the existing per-PR CI gate stable.
3. **Use a weak retrieval gate for the RMI corpus (for now).**
   `DistributedObjectModel.json` has no retrieval ground truth
   (`expectedChunkIds`/`requiredContext`), so Step 1 does **not** do a full
   evidence-coverage check for this corpus. Instead the Step 1 gate is the
   minimal check: **the vector leg produced at least one hit and `search_library`
   returned non-empty `sources`** (no `reason`) for every case. A real
   evidence-coverage gate (adding ground-truth fields to the dataset) is deferred
   to future work. See Section 4a.
4. **Integrate the RMI corpus by indexing the PDF directly (no chunk fixture).**
   Step 1 indexes the raw PDF through the engine worker `index_material`; chunk
   IDs are generated at index time. Building a `tokensmith`-marked Markdown
   fixture with stable chunk IDs (as BuzzDB has) is **not** required for the weak
   gate and is deferred.

### 4a. What the weak gate does and does not check

| Checked by weak gate | NOT checked (deferred) |
| :-- | :-- |
| `search_library` returned `reason == null` | Whether the retrieved chunks are the *expected* chunks |
| `sources` is non-empty for every case | Whether required evidence phrases appear in the context |
| Vector leg produced ≥1 hit (not keyword-only fallback) | Whether forbidden chunks were excluded |

The weak gate confirms retrieval **ran and produced usable chunks**, not that it
retrieved the *correct* evidence. Answer quality is still measured downstream by
Jev in Step 2, so a bad retrieval tends to surface as a low Jev score rather than
a Step 1 failure. This is an accepted limitation of the weak gate.

## 5. Shared File Contract: `retrieved-sources.json`

Written by Step 1, read by Step 2. One entry per case.

```jsonc
{
  "createdAt": "ISO-8601",
  "corpus": "tests/dataset/A Distributed Object Model for the Java System.pdf",
  "dataset": "tests/dataset/DistributedObjectModel.json",
  "embedder": { "provider": "ollama", "model": "nomic-embed-text:latest" },
  "searchMode": "hybrid",
  "limit": 4,
  "cases": [
    {
      "id": "1",
      "question": "…",
      "referenceAnswer": "…",        // carried through so Jev needs no cases file
      "retrieval": {                  // weak gate result (see Section 4a)
        "passed": true,               // sources non-empty, reason == null, vector hit >= 1
        "vectorHits": 7,
        "reason": null,
        "failures": []
      },
      "sources": [                    // full search_library sources (not slimmed)
        {
          "context": "…chunk text…",
          "sectionHeader": "…",
          "sourceChunkIds": ["…"],
          "tokensmithChunkId": "…",
          "documentTitle": "…",
          "collectionName": "…",
          "path": "…",
          "score": 0.0,
          "retrievalMode": "hybrid"
        }
      ]
    }
  ]
}
```

Notes:
- `referenceAnswer` is carried inline so the grading step needs no separate cases
  file (mirrors how `generate-openrouter-answers.mjs` inlines `referenceAnswer`).
- `sources` are the **full** search results; the Markdown report in Step 1 can
  still show shortened excerpts without affecting this file.

## 6. Implementation

### Phase 1 — Retrieval emitter (Step 1)

Add `tests/experiments/retrieve-sources.mjs` (Node driver) plus a small Python
entry that reuses the engine, **or** a single Python script that reuses
`engine.search_library`. Reuse the exact retrieval call the benchmark uses so the
gate semantics match.

Responsibilities:
1. Index the target PDF **directly** with Ollama `nomic-embed-text` into an
   isolated `tmp/dom/profile` via the engine worker `index_material` (mirroring
   the current `run-eval-pipeline.mjs` worker setup). No chunk-marked fixture is
   built; chunk IDs are generated at index time.
2. For each case in `DistributedObjectModel.json`, call `search`
   (`searchMode: 'hybrid'`, `limit`) and capture the full `sources`.
3. Apply the **weak gate** (Decision 3 / Section 4a) per case: pass only if
   `search_library` returned no `reason`, `sources` is non-empty, and the vector
   leg produced at least one hit. To observe the vector-hit count without
   changing engine behavior, either read it from the engine's
   `search_results_ranked` log event or wrap `vector_search` the way
   `test_buzzdb_embeddings.py::retrieve_sources` does. Record
   `passed`/`vectorHits`/`reason`/`failures` per case.
4. Write `retrieved-sources.json` (Section 5) and a human-readable Markdown
   summary. Exit non-zero if any case fails the weak gate.

This script owns the Ollama embedder and the index/cache concern. The
answer-quality pipeline no longer touches embeddings.

### Phase 2 — `--sources-from` in the answer pipeline (Step 2)

Add a `--sources-from <retrieved-sources.json>` flag to
`tests/experiments/run-eval-pipeline.mjs`. When present:

- **Skip** `startWorker`, `index_material`, the Gemini embedder spec, the
  `list_materials` cache lookup, and `prepareRewrittenStudyChat` entirely. The
  `GEMINI_API_KEY` requirement is dropped in this mode.
- For each case, build `request.retrievedSources` from the file's `sources` by
  mapping to `ChatSource` (text + display fields), then:
  - `messages = studyChatMessages(request)`
  - `callOpenRouter(messages)` (Nemotron 3 Super, reasoning enabled — unchanged)
  - `callJev(...)` against the inline `referenceAnswer` (unchanged gate logic)
- Keep the existing report/summary/exit-code behavior.

Source mapping (benchmark source -> `ChatSource`):

| ChatSource field | From search_library source |
| :-- | :-- |
| `context` / `excerpt` | `context` (required) |
| `sectionHeader` | `sectionHeader` |
| `sourceChunkIds` | `sourceChunkIds` |
| `tokensmithChunkId` | `tokensmithChunkId` |
| `documentTitle` / `collectionName` / `title` | same fields |
| `path` | `path` |
| `score`, `retrievalMode` | same fields (optional, for reporting) |

### Phase 3 — Orchestration

Wire the two steps so Step 2 consumes Step 1's output:

```sh
# Step 1: retrieve (Ollama, no Gemini). Emits gate + chunks.
node tests/experiments/retrieve-sources.mjs \
  --dataset tests/dataset/DistributedObjectModel.json \
  --pdf "tests/dataset/A Distributed Object Model for the Java System.pdf" \
  --embedder nomic-embed-text:latest \
  --limit 4 \
  --out tmp/dom/retrieved-sources.json \
  --summary tmp/dom/retrieval-summary.md
# exit 0 only if the evidence-coverage gate passes

# Step 2: generate + grade from the saved chunks (no embeddings, no Gemini).
OPENROUTER_API_KEY=... JEV_API_TOKEN=... \
node tests/experiments/run-eval-pipeline.mjs \
  --sources-from tmp/dom/retrieved-sources.json \
  --generator-model "nvidia/nemotron-3-super-120b-a12b:free" \
  --threshold 0.7 --pass-rate 0.85 \
  --out tmp/dom/evaluation-report.json \
  --summary tmp/dom/evaluation-summary.md
```

### Phase 4 — Verification

- **Dry run end-to-end**: Step 1 with Ollama against the PDF produces non-empty
  `sources` for all 6 cases and a sane pass/fail; `run-eval-pipeline.mjs
  --sources-from … --dry-run` consumes it, builds prompts, and exits cleanly with
  no `GEMINI_API_KEY`/network.
- **Prompt parity check**: confirm `studyChatMessages` output from
  `--sources-from` matches the shape produced by the live path for the same
  chunks (same `### Source unit` blocks), so moving retrieval out does not change
  the generated prompt given identical chunks.
- **Gate independence**: force a weak-gate failure (e.g. point at a case whose
  search returns empty `sources`/a `reason`, or disable the vector leg) and
  confirm Step 1 exits non-zero so Step 2 is not run.

## 7. Scope Summary

| Area | Change |
| :-- | :-- |
| `tests/experiments/retrieve-sources.mjs` (+Python entry) | **New.** Index the PDF with Ollama, run `search_library` per case, apply the weak gate, emit `retrieved-sources.json`. |
| `tests/experiments/run-eval-pipeline.mjs` | Add `--sources-from`; when set, bypass worker/index/Gemini/rewrite and build `retrievedSources` from file. |
| `tests/dataset/DistributedObjectModel.json` | **Unchanged** for now (weak gate). Adding `expectedChunkIds`/`requiredContext` for a real gate is deferred. |
| `tests/benchmarks/test_buzzdb_embeddings.py` | **Unchanged** (committed CI gate stays stable). |
| `docs/testing.md` / `tests/experiments/README.md` | Document the two-step decoupled flow. |

## 8. Deferred / Future Work

1. **Upgrade the weak gate to real evidence coverage.** Add `expectedChunkIds`
   and/or `requiredContext` (seeded from each case's existing `key_evidence_quote`)
   plus `topK`/`minExpectedHits` to `DistributedObjectModel.json`, then reuse the
   benchmark's `validate_grounding_case` logic in Step 1. Note that stable
   `expectedChunkIds` effectively also requires a chunk-marked RMI fixture.
2. **CI integration.** Decide whether the decoupled flow eventually runs in CI
   (opt-in/scheduled) or stays a local two-step experiment.
