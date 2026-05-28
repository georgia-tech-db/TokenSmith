# Benchmark Evaluation

Evaluates TokenSmith's RAG pipeline using two separate benchmark runners, each designed for a different input format and evaluation context. Both runners share the same judge infrastructure, output structure, and Streamlit viewers.

---

## Overview

```
src/benchmark_eval/
    run_benchmark_synthetic-QA-benchmark.py     ← Runner for synthetic QAC files
    run_benchmark_excQAFromBook-benchmark.py     ← Runner for external exercise QA files
    results_viewer--synthetic-QA-benchmark.py   ← Streamlit viewer for synthetic results
    results_viewer--excQAFromBook-benchmark.py  ← Streamlit viewer for external results
    judge.py                                    ← Five LLM judge evaluations (synthetic)
    judge_client.py                             ← Unified judge backend (local + OpenRouter)
    metrics.py                                  ← Deterministic metrics + aggregation
    report.py                                   ← Markdown report generation
    runner.py                                   ← TokenSmith interface
    regenerate_report_from_existing_results.py  ← Regenerate report without re-running
    BOOK-EXC_QARCs.json                         ← Exercise QA benchmark dataset
```

All results are written to `benchmark_results/{run_label}/` in the project root.

---

## Prerequisites

```bash
pip install nltk pyyaml
```

TokenSmith must have a built index before running either benchmark:
```bash
python3 main.py index
```

Both runners use `config/config.yaml` for all TokenSmith settings (model paths, retrieval parameters, generation settings). No separate benchmark config is needed.

---

---

# Synthetic QA Benchmark

**Script:** `src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py`

**Viewer:** `streamlit run src/benchmark_eval/results_viewer--synthetic-QA-benchmark.py`

## What it does

Runs manually reviewed QAC (Question-Answer-Chunk) triplets through TokenSmith and evaluates retrieval and answer quality. This benchmark has ground-truth gold chunks — verbatim sentences from the textbook that are needed to answer each question — so it can measure retrieval coverage in addition to answer quality.

## Input

Verified QAC JSONL files produced by the review tool, located in:
```
synthetic_qac_data/manually_verified/Verified--chapter_XX_qac_*.jsonl
```

Only records with `annotation_status` of `approved` or `edited_approved` are used. Rejected and skipped records are ignored.

Each record must have:

| Field | Description |
|---|---|
| `question` | The question to run through TokenSmith |
| `gold_chunks` | Verbatim textbook sentences needed to answer the question |
| `rubric` | List of criteria a correct answer must satisfy |
| `mock_answer` | Reference answer for BLEU scoring and correctness evaluation |
| `difficulty` | `easy`, `medium`, or `hard` |

## Sub-commands

### `run` — Single benchmark run

```bash
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py run [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--qac_file` | *(required)* | Path to a `Verified--*.jsonl` file, a directory of them, or `all` to use everything in `synthetic_qac_data/manually_verified/` |
| `--config` | `config/config.yaml` | Path to TokenSmith config YAML |
| `--output_dir` | `benchmark_results` | Root directory for output files |
| `--label` | auto (`run_YYYYMMDD_HHMMSS`) | Human-readable label for this run |
| `--chunk_judge_mode` | `group` | Chunk relevance judging: `individual` (one call per chunk) or `group` (one call per 3 chunks) |
| `--rubric_judge_mode` | `all` | Rubric judging: `individual` (one call per criterion) or `all` (one call for all criteria) |
| `--no_judge` | off | Skip all LLM judge evaluations — deterministic metrics only (faster) |
| `--no_resume` | off | Rerun from scratch, ignoring existing results |
| `--dry_run` | off | Print execution plan without running anything |

#### Examples

```bash
# Basic run — one chapter file, all defaults
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py run \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_google_gemini-2.5-pro-preview.jsonl \
    --label chapter8_baseline

# All verified files at once
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py run \
    --qac_file all \
    --label full_benchmark

# Skip LLM judge — retrieval + deterministic metrics only (much faster)
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py run \
    --qac_file all \
    --label full_benchmark_no_judge \
    --no_judge

# Individual rubric judging (one LLM call per criterion instead of all together)
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py run \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_google_gemini-2.5-pro-preview.jsonl \
    --label chapter8_individual_rubric \
    --rubric_judge_mode individual

# Dry run — see the plan without spending any compute
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py run \
    --qac_file all \
    --dry_run
```

### `ab-test` — Grid search over config parameters

```bash
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py ab-test [OPTIONS]
```

All `run` flags are supported, plus:

| Flag | Default | Description |
|---|---|---|
| `--params` | *(required)* | One or more parameter specs: `"param=val1,val2,val3"`. Multiple params produce a full grid search. |

Only **query-time** parameters can be varied. Index-affecting parameters are rejected with a clear error.

**Supported AB test parameters:**

| Parameter | Type | Example |
|---|---|---|
| `top_k` | int | `"top_k=5,10,20"` |
| `num_candidates` | int | `"num_candidates=30,50,80"` |
| `rerank_mode` | str | `"rerank_mode=cross_encoder,"` (trailing comma = empty string = no reranking) |
| `rerank_top_k` | int | `"rerank_top_k=3,5,10"` |
| `ensemble_method` | str | `"ensemble_method=rrf,linear"` |
| `rrf_k` | int | `"rrf_k=30,60"` |
| `gen_model` | str | `"gen_model=models/qwen2.5-3b.gguf,models/qwen2.5-7b.gguf"` |
| `max_gen_tokens` | int | `"max_gen_tokens=400,800"` |
| `system_prompt_mode` | str | `"system_prompt_mode=baseline,tutor,new_v1"` |
| `use_hyde` | bool | `"use_hyde=true,false"` |
| `use_double_prompt` | bool | `"use_double_prompt=true,false"` |

> ⚠️ If more than 8 combinations are specified, a warning is printed. The run proceeds unless you cancel.

#### Examples

```bash
# Compare top_k values
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py ab-test \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_google_gemini-2.5-pro-preview.jsonl \
    --params "top_k=5,10,20" \
    --label topk_sweep

# Compare system prompt modes
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py ab-test \
    --qac_file all \
    --params "system_prompt_mode=baseline,tutor,new_v1" \
    --label prompt_comparison

# Grid search: top_k × rerank_mode (6 combinations)
python3 src/benchmark_eval/run_benchmark_synthetic-QA-benchmark.py ab-test \
    --qac_file all \
    --params "top_k=5,10,20" "rerank_mode=cross_encoder," \
    --label topk_rerank_grid \
    --dry_run
```

## Judge evaluations

The local Qwen GGUF model (`gen_model` from `config/config.yaml`) is used for all judge calls.

| # | Evaluation | Input | Output |
|---|---|---|---|
| 1 | Chunk Relevance | Question + retrieved chunks | `relevant` / `not_relevant` / `uncertain` per chunk |
| 2 | Rubric Satisfaction | Question + TS answer + rubric | `met` / `not_met` / `partial` per criterion |
| 3 | Answer Correctness | Question + TS answer + rubric (± mock answer) | `-1` / `0` / `1`, run twice |
| 4 | Gold Chunk Presence | Gold chunks + retrieved chunks | Deterministic substring check — no LLM call |
| 5 | Answer Faithfulness | TS answer + retrieved chunks | `faithful` / `partially_faithful` / `unfaithful` |

## Output

```
benchmark_results/{label}/
    raw_results.jsonl        ← TS answers + retrieved chunks (one line per QAC)
    judge_results.jsonl      ← All 5 judge outputs (one line per QAC)
    metrics_detail.csv       ← One row per QAC, all metrics as columns
    metrics_summary.json     ← Aggregated: overall + by_difficulty + by_chapter
    report.md                ← Full human-readable report

benchmark_results/{ab_label}/
    {combo_label}/           ← Same structure as above per combination
    comparative_report.md    ← Cross-combination comparison + rankings
    ab_test_summary.json
```

## Regenerating a report from existing results

```bash
python3 src/benchmark_eval/regenerate_report_from_existing_results.py \
    --run_dir benchmark_results/chapter8_baseline
```

Reads `raw_results.jsonl` and `judge_results.jsonl`, recomputes all metrics and examples, overwrites `metrics_detail.csv`, `metrics_summary.json`, and `report.md`. No LLM or TokenSmith calls are made.

---

---

# Exercise QA Benchmark (Book Exercises)

**Script:** `src/benchmark_eval/run_benchmark_excQAFromBook-benchmark.py`

**Viewer:** `streamlit run src/benchmark_eval/results_viewer--excQAFromBook-benchmark.py`

**Benchmark file:** `src/benchmark_eval/BOOK-EXC_QARCs.json`

## What it does

Runs textbook exercise questions through TokenSmith and evaluates the answers using an LLM judge. Unlike the synthetic benchmark, this dataset has no gold chunks — evaluation is based entirely on rubric coverage and answer correctness. Supports both a local Qwen judge and an OpenRouter-hosted judge (e.g. Qwen 72B).

## Input

A JSON array at `src/benchmark_eval/BOOK-EXC_QARCs.json`:

```json
[
  {
    "id": "2.10",
    "question": "Describe the differences in meaning between the terms relation and relation schema.",
    "answer": "A relation is a set of tuples...",
    "must_rubric": [
      "Define 'relation' as a set or collection of tuples.",
      "Define 'relation schema' as the structural definition..."
    ],
    "optional_rubric": [
      "Distinguish between the static nature of a schema and the dynamic nature of a relation instance."
    ]
  },
  ...
]
```

| Field | Description |
|---|---|
| `id` | Unique question identifier (e.g. `"2.10"`, `"6.5"`) |
| `question` | The exercise question |
| `answer` | Reference / mock answer used for the `with_reference` correctness judge call |
| `must_rubric` | Required criteria — counted in the overall rubric coverage score |
| `optional_rubric` | Bonus criteria — scored and shown but not counted in the overall score |

## Flags

```bash
python3 src/benchmark_eval/run_benchmark_excQAFromBook-benchmark.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--benchmark` | *(required)* | Path to the benchmark JSON file |
| `--config` | `config/config.yaml` | Path to TokenSmith config YAML |
| `--output-dir` | `benchmark_results` | Root directory for output files |
| `--label` | auto (`ext_YYYYMMDD_HHMMSS`) | Human-readable label for this run |
| `--judge-backend` | `local` | Judge backend: `local` (Qwen GGUF via llama.cpp) or `openrouter` |
| `--judge-model` | `qwen/qwen-2.5-72b-instruct` | OpenRouter model ID (only used with `--judge-backend openrouter`) |
| `--openrouter-key` | from `.env` | OpenRouter API key (overrides `OPENROUTER_API_KEY` env var) |
| `--chunk-judge-mode` | `group` | Chunk relevance judging: `individual` or `group` (3 per call) |
| `--rubric-judge-mode` | `all` | Rubric judging: `individual` (one per criterion) or `all` (all together) |
| `--no-judge` | off | Skip all LLM judge evaluations — retrieval only |
| `--no-faithfulness` | off | Skip faithfulness evaluation (recommended for local judge to avoid context window issues) |
| `--disable-chunks` | off | Run without retrieval — TokenSmith answers from model knowledge only |
| `--reuse-run` | off | Reuse TokenSmith answers + chunks from a previous run label — only re-runs the judge |
| `--no-resume` | off | Rerun from scratch, ignoring existing results |
| `--dry-run` | off | Print execution plan without running anything |

## OpenRouter API key

The script reads `OPENROUTER_API_KEY` from `.env` in the project root automatically:

```dotenv
OPENROUTER_API_KEY=sk-or-your-key-here
```

No other configuration needed for the OpenRouter backend.

## Examples

```bash
# Local judge — full run with retrieval
python3 src/benchmark_eval/run_benchmark_excQAFromBook-benchmark.py \
    --benchmark src/benchmark_eval/BOOK-EXC_QARCs.json \
    --label book_exc_local \
    --no-faithfulness

# OpenRouter judge — full run with retrieval
python3 src/benchmark_eval/run_benchmark_excQAFromBook-benchmark.py \
    --benchmark src/benchmark_eval/BOOK-EXC_QARCs.json \
    --label book_exc_qwen72b \
    --judge-backend openrouter \
    --judge-model qwen/qwen-2.5-72b-instruct \
    --no-faithfulness

# No retrieval — answers from model knowledge only
python3 src/benchmark_eval/run_benchmark_excQAFromBook-benchmark.py \
    --benchmark src/benchmark_eval/BOOK-EXC_QARCs.json \
    --label book_exc_no_retrieval \
    --disable-chunks \
    --judge-backend openrouter \
    --judge-model qwen/qwen-2.5-72b-instruct \
    --no-faithfulness

# Re-judge a previous run with a different judge (no re-running TokenSmith)
python3 src/benchmark_eval/run_benchmark_excQAFromBook-benchmark.py \
    --benchmark src/benchmark_eval/BOOK-EXC_QARCs.json \
    --label book_exc_rejudge_72b \
    --reuse-run book_exc_local \
    --judge-backend openrouter \
    --judge-model qwen/qwen-2.5-72b-instruct \
    --no-faithfulness

# Retrieval only — no judge calls at all
python3 src/benchmark_eval/run_benchmark_excQAFromBook-benchmark.py \
    --benchmark src/benchmark_eval/BOOK-EXC_QARCs.json \
    --label book_exc_retrieval_only \
    --no-judge

# Dry run
python3 src/benchmark_eval/run_benchmark_excQAFromBook-benchmark.py \
    --benchmark src/benchmark_eval/BOOK-EXC_QARCs.json \
    --dry-run
```

## Judge evaluations

| # | Evaluation | Input | Output |
|---|---|---|---|
| 1 | Must Rubric Satisfaction | Question + TS answer + must_rubric | `met` / `not_met` / `partial` per criterion |
| 2 | Optional Rubric Satisfaction | Question + TS answer + optional_rubric | `met` / `not_met` / `partial` per criterion (informational only) |
| 3 | Answer Correctness (no ref) | Question + TS answer + must_rubric | `-1` / `0` / `1` |
| 4 | Answer Correctness (with ref) | Question + TS answer + must_rubric + mock answer | `-1` / `0` / `1` |
| 5 | Chunk Relevance | Question + retrieved chunks | `relevant` / `not_relevant` / `uncertain` per chunk |
| 6 | Answer Faithfulness | TS answer + retrieved chunks | `faithful` / `partially_faithful` / `unfaithful` (skippable) |

**Must rubric** results count toward the overall rubric coverage score. **Optional rubric** results are stored and displayed but not included in the summary score.

## Output

```
benchmark_results/{label}/
    full_results.json        ← Everything: TS answers, retrieved chunks, all judge outputs,
                                summary scores per question. Main save artifact.
    raw_results.json         ← TS answers + retrieved chunks only (no judge data)
    judge_results.json       ← Judge outputs + summary scores only (no chunks)
    report.md                ← Human-readable report with aggregate tables + per-question breakdown
```

All three JSON files are regenerated from `full_results.json` at the end of each run. `full_results.json` is written after every question for crash safety — re-running the same command with the same `--label` resumes from where it left off.

---

---

# Viewing Results

## Synthetic QA Benchmark Viewer

```bash
streamlit run src/benchmark_eval/results_viewer--synthetic-QA-benchmark.py
```

Reads from `benchmark_results/` — shows any run directory containing `raw_results.jsonl` + `judge_results.jsonl`.

**Features:** Run selector, filters (chapter, difficulty, gen status, gold coverage, correctness, faithfulness), question table, detail panel showing TS answer, rubric with per-criterion judge verdicts, gold chunks with 🟢/🔴 coverage indicators, retrieved chunks with relevance verdicts, faithfulness verdict + unsupported claims.

## Exercise QA Benchmark Viewer

```bash
streamlit run src/benchmark_eval/results_viewer--excQAFromBook-benchmark.py
```

Reads from `benchmark_results/` — shows any run directory containing `full_results.json`.

**Features:** Run selector, filters (correctness, must rubric coverage, faithfulness), question table, detail panel with prev/next navigation. Left panel: generated answer, mock answer, judge scores with explanations. Right panel: must rubric and optional rubric breakdowns, retrieved chunks with relevance verdicts, faithfulness.

---

---

# Notes on the Judge Model

**Local backend** (`--judge-backend local`):
- Uses `gen_model` from `config/config.yaml` — the same Qwen GGUF model as TokenSmith's generator
- Uses Qwen 2.5's native chat template (`<|im_start|>` / `<|im_end|>`) with JSON output priming
- All judge calls use `temperature=0.0` for deterministic output
- Context window is limited — avoid faithfulness evaluation with 10+ long chunks (`--no-faithfulness`)
- The `LlamaRAMCache` in `src/generator.py` should be disabled during benchmark runs to prevent stale outputs when changing prompts (comment out the `set_cache` lines)

**OpenRouter backend** (`--judge-backend openrouter`):
- Supports any model available on OpenRouter
- Reads `OPENROUTER_API_KEY` from `.env` in the project root
- No context window concerns — 72B models have 32k+ context
- Recommended for the exercise benchmark where questions and answers can be long

**JSON parsing:** Both backends handle markdown fences, truncated output, and unclosed braces. If a judge call fails, the metric is recorded as `None` / `uncertain` and the pipeline continues.