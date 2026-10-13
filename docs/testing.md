# Testing And Benchmarks

Complete the [developer setup](development.md#local-setup) first. Run all commands
below from the repository root.

## Local Checks

With Ollama running, install the benchmark model and run tests:

```sh
ollama pull nomic-embed-text:latest
npm run typecheck
npm test
```

The benchmark uses the bundled Python runtime and TokenSmith's production Ollama
embedding function. The exact model digest and Ollama version are pinned in
[embedding-model.json](../tests/benchmarks/embedding-model.json); use that version
of Ollama locally. A changed model tag fails validation instead of silently
changing the benchmark.

## CI And Benchmarks

Every pull request, push to `main`, and merge-queue change runs:

- **CI:** typechecking, production build, TypeScript/Python unit tests, and Python-worker/conversation integration tests.
- **Accuracy benchmark (retrieval):** BuzzDBBook questions using the app's `nomic-embed-text:latest` (137M F16) through Ollama and its hybrid search, including source expansion and final source selection. Conversation-contract checks run alongside it and are reported separately.

The benchmark badge is a pass/fail regression check, **not generated-answer accuracy**.
Its Actions summary shows per-question evidence coverage and the measured pass rate.
Each Details cell shows the question, an authored reference answer (not generated
or automatically graded), and ranked retrieved-source excerpts. Conversation
checks label their prior exchange and evidence as fixtures, not live retrieval.
The conversation checks mock the rewrite model and search; they do not grade live
follow-up understanding or generated answers. CI starts its own CPU-only Ollama
service with no cloud keys or paid model calls. The CPU runtime, model weights,
and book embeddings are cached, but queries are
embedded afresh and the search index is rebuilt on every run. PR checks do not
upload packaged apps or other build artifacts.

Run just the benchmark:

```sh
npm run test:benchmark
```

The first benchmark run embeds the book using the installed Nomic model.
For repeat runs, set `TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH` to a cache produced by
`npm run build:embedding-benchmark-cache -- <path>`. Stale caches fail validation.
Changes to the model, parsed chunks, or embedding input preparation invalidate
the cache; question or ranking changes alone can reuse the book vectors.
Set `TOKENSMITH_BENCHMARK_REPORT_PATH` to save the combined JSON results locally.

### Harder BuzzDB Questions

The regular benchmark has 16 live hybrid-retrieval questions (10 conceptual and
6 application/reasoning questions), plus 16 mocked conversation-contract checks.
The six harder families cover buffer safety/WAL, 2Q traces, aggregate initialization
bugs, column-page reads, aggregation trade-offs, and batching speedups. Each has
three turns and four evaluator-only criteria per turn in
[buzzdb_reasoning_cases.json](../tests/benchmarks/buzzdb_reasoning_cases.json).
Required evidence groups check each prerequisite, not just one matching passage.

To also generate the 18 new answers using the installed `gemma4:e4b` model:

```sh
TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH=tmp/buzzdb-nomic-cache.npz npm run test:benchmark -- --live
```

Select a different installed Ollama model without changing the app's settings:

```sh
TOKENSMITH_BENCHMARK_MODEL=gemma4:26b TOKENSMITH_BENCHMARK_SEED=19 TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH=tmp/buzzdb-nomic-cache.npz npm run test:benchmark -- --live
```

The optional seed is applied to every chat API request (rewriter, prompt probe,
and answer), recorded with the actual requests, and never saved to application
settings. An unset seed keeps the runtime default. A fixed seed aids paired
comparisons but does not guarantee identical outputs across runtime versions.
The model must already be installed; the runner never downloads it.

Set `TOKENSMITH_BENCHMARK_REASONING=auto`, `on`, or `off` to compare reasoning
policies. The benchmark defaults to `off` for continuity with earlier runs;
new app settings default to `auto`. Auto adds a non-thinking structured planning
call for first questions on supported Ollama models and shares the existing
rewrite call for follow-ups. Its boolean decision is independent of contextual
versus standalone mode. The requested task and previous exchange determine the
decision, without question-length rules or subject-specific keywords.

Reasoning answers reserve at least 4,096 generated tokens (reasoning plus final
answer) before packing sources; larger configured allowances are preserved.
The same context-fit checks still apply. Unsupported models stay non-thinking
in Auto; forcing On reports unsupported reasoning. Cloud generation settings are
unchanged. Logs record model capability, planner decision, effective reasoning
mode, and output allowance. Compare actual answers, limits, and total latency,
not just routing labels or successful completion.

`TOKENSMITH_BENCHMARK_CASES_PATH=tests/benchmarks/buzzdb_reasoning_variants.json`
selects six additional live turns covering slot relocation, leaf/internal splits,
and SIMD masks. These alternate live cases do not replace or enlarge the 16 CI
retrieval questions. They were authored before the MoE prompt experiment and
include explicit changes to the preceding scenario. Like the original chains,
they are fixed diagnostic questions, not adaptively generated student follow-ups.

This is the same benchmark entry point, with an opt-in live-answer stage. It uses
production rewriting, real Nomic hybrid search, source packing/preflight, and
generation. The baseline profile uses 8,192 context, 1,536 output,
temperature 0.7, top-p 0.4, top-k 40, repeat penalty 1.18, reasoning off, and no
custom system prompt. The production model-aware retrieval limit is recorded
per run (7 candidates for E4B and 8 for 26B with these settings). Suggestions are
disabled to isolate the answer path.
It does not modify the installed app, its library, or saved model settings.

All Ollama chat models use a measured prompt preflight, not just E4B. The packer
reserves the requested output allowance, keeps source units intact, and removes
whole trailing units if Ollama's token count leaves insufficient answer space.
An impossible fit fails explicitly. This verifies context headroom, not answer
correctness, and adds a one-token probe before generation (additional probes
only when a smaller prompt must be measured). Remote models reserve their full
output allowance too, but continue to use estimated packing without this probe.

Each family starts fresh, and later turns use its own newly generated answers.
These are fixed diagnostic probes, not adaptive student questions. Expected answers
and rubrics are never sent to the model. An execution failure blocks the remaining
dependent turns; no retry or reference answer replaces it. A wrong but completed
answer remains in the conversation. Reference traces and arithmetic are authored
expectations, not an automatic prose-grading mechanism.

The run saves full questions, answers, ranked evidence, actual model requests,
preflight/final token counts, timings, model digest, and source hashes in a fresh
`tmp/buzzdb-reasoning-<timestamp>/` directory. Set `TOKENSMITH_BENCHMARK_LIVE_DIR`
to choose a different **new** directory. Existing runs are never overwritten.
`answers.md` is readable; `results.json` retains the full evidence and requests.
The evaluated TypeScript application is copied to `app/src` in that directory
and loaded from that snapshot, so editing the working tree during a run cannot
change its pipeline. For a controlled replay of an earlier implementation, set
`TOKENSMITH_BENCHMARK_APP_SOURCE_ROOT` to its saved `app` directory. Python
retrieval still uses the current checkout and validated embedding cache; its
source hashes are recorded. This is not a full historical-environment replay.
Evidence checks are also applied to complete passages identified in the final
model request, so packing losses are distinguishable from retrieval omissions.
When replaying a legacy packer that clips passages, partially included sources
do not count as fully preserved; their actual text remains in the saved requests.
The live report tracks each family's canonical reference passages across its
turns. Missing one is a diagnostic, not proof that no equivalent evidence exists
or that the answer is wrong; follow-ups can also need only a subset of that
family's evidence. Review the actual passages before attributing an answer error
to retrieval. The initial CI questions require all their specified groups.

Completed answers are explicitly **ungraded**, not accuracy passes. Review all
four criteria (0 / 0.5 / 1 each) and record additional substantive errors
separately; a correct number or keyword alone is not a correct explanation.
Label manual reviews with their reviewer and run identity. The live stage is
not part of every-PR CI and does not change the retrieval badge's meaning.

Configure branch protection to require `Build and tests` and
`BuzzDB hybrid retrieval and conversation contracts` before merging. The README
badges track `main`, while each PR has its own check results.

## Coverage

The two coverage badges report **unit-test line coverage** separately for
frontend/Electron TypeScript and backend Python. TypeScript includes all of
`src/`: UI, Electron main/preload, and shared modules, including untested files
but excluding declarations. Python covers `python_engine/`. CI shows language
totals and per-file coverage. These are reporting metrics, not accuracy scores
or minimum-coverage gates. Only a successful CI
push on `main` publishes the badges and small report to `codex/coverage-badge`;
PRs cannot overwrite it. No coverage artifacts or external-service secrets are needed.

To run coverage locally without modifying the packaged Python runtime:

```sh
python3.12 -m pip install --target tmp/coverage-tools -r requirements-coverage.txt
npm run coverage
```
