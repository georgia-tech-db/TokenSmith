# Independent holdout evaluation of PR #157

This directory contains the frozen test questions, corpus, scripts, raw outputs, blind scoring, and results discussed on [PR #157](https://github.com/georgia-tech-db/TokenSmith/pull/157#issuecomment-6168678756).

The tested PR commit is **`d628f8322c09911306c923d655db1139aab035e0`**. This branch adds evaluation artifacts on top of that commit; it does not change application code. Measurements were collected on 14 October 2026.

## Results

| Measure | Off | On |
| --- | ---: | ---: |
| Relevant target retrieved, casual questions | 28/48 | 32/48 |
| All target evidence groups retrieved, casual questions | 20/48 | 22/48 |
| Fully correct and source-supported answers | 11/24 | 12/24 |
| Partial or complete source-supported answers | 19/24 | 21/24 |
| Initial preparation | 18.1 seconds | 741.3 seconds |
| Unchanged cached preparation | 1.10 seconds | 1.47 seconds |

The primary answer result had two wins and one regression: +4.2 percentage points, with an exploratory paired bootstrap interval of -8.3 to +16.7 points. This supports retaining the optional, off-by-default design. It does not establish a general benefit large enough to repay preparation time.

These are **new synthetic student-style questions and assistant grading of blinded A/B answers**, not actual student submissions or a human study. Questions, evidence targets, and criteria were frozen before enrichment. The four complete chapters contain 416 original passages; all 251 eligible passages were enriched, including distractors. There were 24 concepts, each with two casual formulations and one technical control. Every first casual formulation was used in the 24-pair answer test. The sample is one textbook and one local model pair. Read [protocol.txt](protocol.txt) for the design and limits.

## Inspect the evidence

- [cases.json](cases.json): all 72 questions, original evidence groups, and frozen answer criteria.
- [freeze.json](freeze.json): pre-run timestamp and hashes of the unmodified inputs.
- [report.html](report.html): full report with expandable per-question sources, answers, and scores. Download the directory and open this file locally; GitHub does not render arbitrary HTML pages.
- [analysis.json](analysis.json), [retrieval-summary.json](retrieval-summary.json), and [answer-summary.json](answer-summary.json): machine-readable results.
- [retrieval.json](retrieval.json): all 576 searches, including retrieved source text and IDs.
- [answers.json](answers.json): all 48 answers, complete inference requests/responses, source budgets, and timings.
- [blind-review.json](blind-review.json) and [manual-scores.json](manual-scores.json): anonymous review packet and individual scores. Score 1 can still be useful; score 2 requires both frozen details without a material false assertion.
- [blind-key.json](blind-key.json): arm decoding. **Do not inspect this before an independent regrade.**
- [preparation.json](preparation.json), the two `*-cold-calls.jsonl` logs, and [environment.json](environment.json): preparation timings, generation calls, hardware, exact model digests, and engine hashes.
- `off-stored-chunks.json`, `on-stored-chunks.json`, and `query-embeddings.json`: original/indexed passages, generated terms, and recorded vectors. Large files are downloadable through GitHub's Raw button.

## Verify the saved result without model calls

Clone the evaluation branch into a separate directory:

```sh
git clone --single-branch --branch codex/pr157-independent-evaluation https://github.com/georgia-tech-db/TokenSmith.git TokenSmith-pr157-eval
cd TokenSmith-pr157-eval
python3 tests/experiments/pr157-holdout-2026-10-14/verify.py --source .
```

This uses only the Python standard library. It checks the publication manifest, original freeze hashes, PR source hashes, answer completion, original passage identity, and the reported primary counts. Verification does not contact Ollama or change the published results.

## Reproduce the experiment

The measured environment used Python 3.12.14, Node 20.19.6, TypeScript 6.0.3, Ollama 0.31.1, Apple M1 Max, and 32 GiB memory. Python package versions are pinned in [requirements.txt](requirements.txt); JavaScript dependencies are pinned by the repository lockfile. Different hardware or model digests can change results and timing.

From the cloned repository root:

```sh
python3 -m venv .venv-pr157
. .venv-pr157/bin/activate
python -m pip install -r tests/experiments/pr157-holdout-2026-10-14/requirements.txt
npm ci --ignore-scripts
node tests/experiments/pr157-holdout-2026-10-14/verify-packing.mjs --source .
python tests/experiments/pr157-holdout-2026-10-14/new_run.py --source . --name pr157-holdout-rerun
```

The offline packing check recreates all 48 recorded prompts, source counts, and budget limits. Historical heuristic token estimates include path metadata and can change when local paths are shortened; the recorded estimates remain intact. `new_run.py` checks the engine source and frozen inputs, creates a new sibling experiment directory and fresh temporary indexes, and prints the exact commands for preparation, retrieval, answers, and blind review. It makes no model calls. It refuses an existing destination, preventing accidental replacement of saved evidence. **Run the printed commands in order.** They require local Ollama at `http://127.0.0.1:11434` with these models already installed:

| Model | Recorded digest |
| --- | --- |
| `gemma4:e4b` | `c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb` |
| `nomic-embed-text:latest` | `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` |

Compare the digests returned by `curl http://127.0.0.1:11434/api/tags` with the table. A matching model name alone is insufficient. The runners record the actual digests but do not automatically download or enforce model versions. A mismatch is a new condition, not an exact reproduction.

Do not rerun `build_cases.py`: it documents input construction, while the reproduction should retain the original freeze. Both arms use the same corpus path. Preparation receives no question or gold-answer file. The fresh-run script must remain under `tests/experiments` in the checkout. Cold means fresh application caches, not cleared OS caches; the reference run began with no resident Ollama models. Initial preparation ran off first, then on. Search and answer arms were interleaved as recorded in the scripts.

After generation, grade `blind-review.json` before opening `blind-key.json`. Save one record per answer in `manual-scores.json`:

```json
[
  {"reviewId": "flexible_rows.casual_a.A", "score": 2, "supported": true, "note": "Reason tied to the frozen criteria and supplied sources."}
]
```

That example illustrates the schema, not a score to reuse. Supply all 48 records, then compute the new statistics:

```sh
python tests/experiments/pr157-holdout-rerun/summarize.py
```

`summarize.py` writes fresh statistics to `rerun-summary.json`, including concept-cluster bootstrap intervals. It also works before grading and reports answer scoring as pending. The original `build_report.py` recreates the **published run's** report and includes result-specific explanatory prose; do not use that prose as the report for a different run.

## Publication and portability notes

The [artifact manifest](artifact-manifest.json) hashes every published file except itself. The original four frozen files, questions, source text, generated descriptions, numerical measurements, answers, and scores are preserved. Local absolute workspace paths in metadata were replaced with `${REPO_ROOT}`, `${PR_SOURCE}`, `${WORK_DIR}`, and `${PYTHON_ENV}`. Unused installed models were removed from the environment inventory. [reproduction.txt](reproduction.txt) preserves the original run notes with those path placeholders; this README provides the portable commands.

Two execution-only adaptations were made after measurement:

1. `run.py` now collects hardware metadata without macOS-only commands on other platforms. The original is saved in [original-scripts/run.py](original-scripts/run.py).
2. `answers.mjs` imports the bundled `ts-module-loader.mjs` rather than a helper from the evaluator's other checkout. This is the exact helper used originally; its difference from the PR helper concerns temporary-file isolation and atomic writes, not model settings or source packing. The original runner is saved in [original-scripts/answers.mjs](original-scripts/answers.mjs).

The additions `verify.py`, `verify-packing.mjs`, `new_run.py`, and `summarize.py` support audit and reproduction. They were added after scoring; they did not author questions or select results. Nothing in this package claims that rerunning a local language model on different hardware must produce byte-identical answers.
