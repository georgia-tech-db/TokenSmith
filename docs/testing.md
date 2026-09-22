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
