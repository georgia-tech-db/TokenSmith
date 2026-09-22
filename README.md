<p align="center">
  <img src="src/renderer/src/assets/tokensmith-mark.png" alt="TokenSmith icon" width="120" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="src/renderer/src/assets/tokensmith-logo.png" alt="TokenSmith" width="120" />
</p>

# TokenSmith

[![CI](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/ci.yml)
[![Accuracy benchmark (retrieval)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml/badge.svg?branch=main&event=push)](https://github.com/georgia-tech-db/TokenSmith/actions/workflows/accuracy.yml)
[![Code coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeorgia-tech-db%2FTokenSmith%2Fcodex%2Fcoverage-badge%2Fcoverage.json)](https://github.com/georgia-tech-db/TokenSmith/blob/codex/coverage-badge/coverage.md)

TokenSmith is a desktop app for students to ask questions on your course documents (PDFs). 

It runs locally on your machine, retrieves passages relevant to your question from your documents, and shows the **page sources** with each answer.

<p align="center">
<img width="1348" height="838" alt="tokensmith" src="https://github.com/user-attachments/assets/ca1ecc04-73ea-4190-b7ed-e58ebeb25a01" />
</p>

## Student Workflow

1. Install and start Ollama.
2. Download the recommended local embedder and chat models.
3. Add a folder containing your course PDFs.
4. Ask questions in Chat or pick a suggested question.
5. Use page source cards to explore where an answer came from within the document and **skim through the page**.
6. Continue with your own questions or suggested follow-up questions to study deeper.

## What TokenSmith Does

- Indexes PDFs for local search using the embedder model.
- Retrieves relevant passages before answering using a vector index.
- Answers with page source cards for cross-checking with the documents.
- Suggests follow-up questions.

## Install

Download the latest app from the GitHub Releases page: https://github.com/georgia-tech-db/TokenSmith/releases

On first launch, TokenSmith will guide you through installing Ollama, downloading models, and adding PDFs.

## Developer Setup

Install dependencies:

```sh
npm install
```

Download the pinned, platform-specific Python runtime and install TokenSmith's
Python dependencies inside it:

```sh
npm run setup:python-runtime
```

This creates `app_runtime/python` inside the repository. It does not require,
modify, or install packages into your system Python. The download is selected
for Linux x64, Windows x64, or macOS ARM64 and verified with SHA-256 before it
is used.

The runtime comes from Astral's
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
`install_only_stripped` archives. TokenSmith pins the Python version, release,
target platform, and checksum rather than copying the developer's Python
installation.

Start the app locally:

```sh
npm run dev
```

With Ollama running, install the benchmark model and run tests:

```sh
ollama pull nomic-embed-text:latest
npm run typecheck
npm test
```

The benchmark uses the bundled Python runtime and TokenSmith's production Ollama
embedding function. The exact model digest and Ollama version are pinned in
[embedding-model.json](tests/benchmarks/embedding-model.json); use that version
of Ollama locally. A changed model tag fails validation instead of silently
changing the benchmark.

## CI And Benchmarks

Every pull request, push to `main`, and merge-queue change runs:

- **CI:** typechecking, production build, TypeScript/Python unit tests, and Python-worker/conversation integration tests.
- **Accuracy benchmark (retrieval):** BuzzDBBook questions using the app's `nomic-embed-text:latest` (137M F16) through Ollama and its hybrid search, including source expansion and final source selection. Conversation-contract checks run alongside it and are reported separately.

The benchmark badge is a pass/fail regression check, **not generated-answer accuracy**.
Its Actions summary shows per-question evidence coverage and the measured pass rate.
The conversation checks mock the rewrite model and search; they do not grade live
follow-up understanding or generated answers. CI starts its own CPU-only Ollama
service with no cloud keys or paid model calls. The CPU runtime, model weights,
and book embeddings are cached, but queries are
embedded afresh and the search index is rebuilt on every run. PR checks do not
upload packaged apps or other build artifacts.

Run just the benchmark after the developer setup above:

```sh
npm run test:benchmark
```

The first benchmark run embeds the book using the installed Nomic model.
For repeat runs, set `TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH` to a cache produced by
`npm run build:embedding-benchmark-cache -- <path>`. Stale caches fail validation.
Changes to the model, parsed chunks, or embedding input preparation invalidate
the cache; question or ranking changes alone can reuse the book vectors.
Set `TOKENSMITH_BENCHMARK_REPORT_PATH` to save the combined JSON results locally.

After the first successful GitHub runs, configure branch protection to require
`Build and tests` and `BuzzDB hybrid retrieval and conversation contracts` before
merging. The README badges track `main`, while each PR has its own check results.

The coverage badge reports **unit-test line coverage** across `src/` (TypeScript
and TSX, including untested UI files, excluding declarations) and `python_engine/`.
CI shows separate language totals and per-file coverage. The combined percentage
is weighted by line counts, not averaged across languages. It is a reporting
metric, not an accuracy score or a minimum-coverage gate. Only a successful CI
push on `main` publishes the badge and small report to `codex/coverage-badge`;
PRs cannot overwrite it. No coverage artifacts or external-service secrets are needed.

To run coverage locally without modifying the packaged Python runtime:

```sh
python3.12 -m pip install --target tmp/coverage-tools -r requirements-coverage.txt
npm run coverage
```

## Packaging

Packaging requires `npm run setup:python-runtime` first. Release workflows run
that command automatically and include the private runtime in each application,
so students do not need to install Python.

Create a macOS DMG:

```sh
npm run package:mac
```

Create a Windows portable zip:

```sh
npm run package:win
```
