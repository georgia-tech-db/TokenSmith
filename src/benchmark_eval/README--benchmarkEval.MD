# Benchmark Evaluation

Evaluates TokenSmith's RAG pipeline against a verified QAC (Question-Answer-Chunk) benchmark dataset. Runs questions through TokenSmith, computes retrieval and answer quality metrics, runs a five-part LLM judge evaluation using the local Qwen model, and generates comprehensive markdown reports. Supports AB testing across TokenSmith config parameters.

---

## Overview

The evaluation pipeline has four stages:

1. **Run** — Each benchmark question is sent through TokenSmith's full retrieval and generation pipeline
2. **Judge** — Five LLM-as-judge evaluations run on the results using the local Qwen GGUF model
3. **Metrics** — Deterministic metrics are computed and aggregated at QAC, difficulty, and chapter level
4. **Report** — A verbose, human-readable markdown report is generated with tables, scores, and illustrative examples

---

## Directory Structure

```
src/benchmark_eval/
    __init__.py
    run_benchmark.py     ← CLI entry point
    runner.py            ← TokenSmith interface + result collection
    judge.py             ← Five LLM judge evaluations
    metrics.py           ← Deterministic metrics + aggregation + example selection
    report.py            ← Markdown report generation (single-run + AB comparative)

benchmark_results/
    {run_label}/
        raw_results.jsonl        ← TokenSmith answers + retrieved chunks
        judge_results.jsonl      ← All 5 judge evaluation outputs
        metrics_detail.csv       ← One row per QAC, all metrics as columns
        metrics_summary.json     ← Aggregated: overall + by_difficulty + by_chapter
        report.md                ← Full human-readable benchmark report

    {ab_test_label}/
        {combo_label}/           ← Same structure as above per combination
        comparative_report.md    ← Cross-combination comparison table + rankings
        ab_test_summary.json     ← All combination metrics in one file
```

---

## Prerequisites

```bash
pip install nltk pyyaml
```

TokenSmith must already have a built index (`python3 main.py index` must have been run). The benchmark evaluation imports TokenSmith's pipeline directly — no separate server needed.

The local Qwen GGUF model (`gen_model` in `config/config.yaml`) must be present. It is used for all judge evaluations.

---

## Input: Verified QAC Files

The benchmark reads from verified QAC files produced by the review tool in `synthetic_qac_data/manually_verified/`. Only records with `annotation_status` of `approved` or `edited_approved` are used — rejected and skipped records are ignored.

Each QAC record must have:
- `question` — the question to run through TokenSmith
- `gold_chunks` — verbatim sentences from the textbook needed to answer the question
- `rubric` — list of criteria a correct answer must satisfy
- `mock_answer` — reference answer for BLEU scoring and correctness evaluation
- `difficulty` — `easy`, `medium`, or `hard`

---

## Sub-commands

### `run` — Single Benchmark Run

```bash
python3 src/benchmark_eval/run_benchmark.py run [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--qac_file` | *(required)* | Path to a `Verified--*.jsonl` file, a directory of them, or `all` to use everything in `synthetic_qac_data/manually_verified/` |
| `--config` | `config/config.yaml` | Path to TokenSmith config YAML |
| `--output_dir` | `benchmark_results` | Root directory for output files |
| `--label` | auto (`run_YYYYMMDD_HHMMSS`) | Human-readable label for this run |
| `--chunk_judge_mode` | `group` | How to judge chunk relevance: `individual` (one call per chunk) or `group` (one call per 3 chunks) |
| `--rubric_judge_mode` | `all` | How to judge rubric satisfaction: `individual` (one call per criterion) or `all` (one call for all criteria) |
| `--no_judge` | off | Skip all LLM judge evaluations — runs only deterministic metrics (much faster) |
| `--no_resume` | off | Rerun everything from scratch, ignoring existing results |
| `--dry_run` | off | Print execution plan without running anything |

#### Examples

```bash
# Basic run with defaults
python3 src/benchmark_eval/run_benchmark.py run \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_gemini.jsonl \
    --label chapter8_baseline

# Fast run — deterministic metrics only, no LLM judge calls
python3 src/benchmark_eval/run_benchmark.py run \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_gemini.jsonl \
    --label chapter8_no_judge \
    --no_judge

# Individual chunk judging (more granular, more API calls)
python3 src/benchmark_eval/run_benchmark.py run \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_gemini.jsonl \
    --label chapter8_individual \
    --chunk_judge_mode individual \
    --rubric_judge_mode individual

# All verified files at once
python3 src/benchmark_eval/run_benchmark.py run \
    --qac_file all \
    --label full_benchmark

# Dry run — see the plan first
python3 src/benchmark_eval/run_benchmark.py run \
    --qac_file all --dry_run

# Force full rerun ignoring existing results
python3 src/benchmark_eval/run_benchmark.py run \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_gemini.jsonl \
    --label chapter8_fresh \
    --no_resume
```

---

### `ab-test` — Grid Search AB Testing

```bash
python3 src/benchmark_eval/run_benchmark.py ab-test [OPTIONS]
```

All flags from `run` are supported plus:

| Flag | Default | Description |
|---|---|---|
| `--params` | *(required)* | One or more parameter specs: `"param=val1,val2,val3"`. Multiple params produce a full grid search |

#### Supported AB Test Parameters

Only **query-time** parameters can be varied. Parameters that require re-indexing or re-embedding are rejected with a clear error.

| Parameter | Type | Example Values |
|---|---|---|
| `top_k` | int | `5,10,20` |
| `num_candidates` | int | `30,50,80` |
| `rerank_mode` | str | `cross_encoder,` (trailing comma = include empty string) |
| `rerank_top_k` | int | `3,5,10` |
| `ensemble_method` | str | `rrf,linear` |
| `rrf_k` | int | `30,60` |
| `ranker_weights` | dict | Specify as JSON string |
| `gen_model` | str | `models/qwen2.5-3b.gguf,models/qwen2.5-7b.gguf` |
| `max_gen_tokens` | int | `200,400,600` |
| `system_prompt_mode` | str | `baseline,tutor,concise` |
| `use_hyde` | bool | `true,false` |
| `hyde_max_tokens` | int | `200,300` |
| `use_double_prompt` | bool | `true,false` |

**Rejected parameters** (require re-indexing — will error with a clear message):
`chunk_size_in_chars`, `chunk_overlap`, `chunk_mode`, `embed_model`, `embedding_model_context_window`, `do_llm_chunk_reorg`, `do_llm_coref_res`, `use_indexed_chunks`, `extracted_index_path`, `page_to_chunk_map_path`

**Warning:** If more than 8 combinations are specified, a warning is printed about runtime. The run still proceeds unless you cancel.

#### Examples

```bash
# Vary top_k only — 3 combinations
python3 src/benchmark_eval/run_benchmark.py ab-test \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_gemini.jsonl \
    --params "top_k=5,10,20" \
    --label topk_sweep

# Vary top_k and rerank_mode — 6 combinations (3 × 2)
python3 src/benchmark_eval/run_benchmark.py ab-test \
    --qac_file synthetic_qac_data/manually_verified/Verified--chapter_08_qac_gemini.jsonl \
    --params "top_k=5,10,20" "rerank_mode=cross_encoder," \
    --label topk_vs_rerank

# Compare system prompt modes
python3 src/benchmark_eval/run_benchmark.py ab-test \
    --qac_file all \
    --params "system_prompt_mode=baseline,tutor,concise,detailed" \
    --label prompt_mode_comparison

# Dry run to check combinations before running
python3 src/benchmark_eval/run_benchmark.py ab-test \
    --qac_file all \
    --params "top_k=5,10" "rerank_mode=cross_encoder," \
    --label test_grid \
    --dry_run
```

---

## Judge Evaluations

All five evaluations use the local Qwen GGUF model (`gen_model` from `config/config.yaml`) via `llama_cpp`. The model path is read directly from the TokenSmith config — no separate configuration needed.

| Evaluation | Input | Output |
|---|---|---|
| **1. Chunk Relevance** | Question + retrieved chunks | `relevant` / `not_relevant` / `uncertain` per chunk |
| **2. Rubric Satisfaction** | Question + TS answer + rubric | `met` / `not_met` / `partial` per criterion |
| **3. Answer Correctness** | Question + TS answer + rubric (+ optional mock answer) | `-1` / `0` / `1` score, run twice (with and without reference) |
| **4. Gold Chunk Presence** | Gold chunks + retrieved chunks | Deterministic substring check — no LLM call |
| **5. Answer Faithfulness** | TS answer + retrieved chunks | `faithful` / `partially_faithful` / `unfaithful` / `uncertain` |

**Chunk relevance modes:**
- `individual` — one judge call per retrieved chunk
- `group` — one judge call per 3 retrieved chunks (default, cheaper)

**Rubric satisfaction modes:**
- `individual` — one judge call per rubric criterion
- `all` — one judge call for all criteria together (default, cheaper)

---

## Metrics

Computed at three levels of granularity: per-QAC, by difficulty, and overall.

### Retrieval Metrics

| Metric | Description |
|---|---|
| `gold_chunk_coverage_rate` | Fraction of gold chunks found in any retrieved chunk |
| `perfect_gold_coverage_pct` | % of QACs where ALL gold chunks were retrieved |
| `retrieval_precision` | Fraction of retrieved chunks containing at least one gold chunk |
| `n_retrieved_chunks` | Number of chunks TokenSmith retrieved |

### Answer Quality Metrics

| Metric | Description |
|---|---|
| `rubric_met_rate` | Fraction of rubric criteria fully met |
| `rubric_avg_score` | Average score (1=met, 0.5=partial, 0=not_met) |
| `correctness_score_no_ref` | -1/0/1 correctness score without mock answer reference |
| `correctness_score_with_ref` | -1/0/1 correctness score with mock answer reference |
| `bleu_score` | BLEU score between TS answer and mock answer |
| `faithfulness_score` | 1=faithful, 0.5=partially faithful, 0=unfaithful |
| `answer_word_count` | Word count of TokenSmith's answer |
| `chunk_relevance_rate` | Fraction of retrieved chunks judged relevant |

---

## Output Files

### `raw_results.jsonl`
One line per QAC. Contains the original QAC record plus:
- `ts_answer` — TokenSmith's generated answer
- `retrieved_chunks` — list of dicts with `chunk_id`, `rank`, `content`, retrieval scores
- `hyde_query` — the HyDE-expanded query if HyDE was enabled
- `error` — error message if get_answer() raised

### `judge_results.jsonl`
One line per QAC. Contains `record_id` and a `judgements` dict with all five evaluation outputs.

### `metrics_detail.csv`
One row per QAC. All metric fields as columns. Suitable for analysis in pandas, Excel, etc.

### `metrics_summary.json`
Aggregated metrics with structure:
```json
{
  "overall": {"n": 50, "gold_chunk_coverage_rate_mean": 0.82, ...},
  "by_difficulty": {
    "easy":   {"n": 25, ...},
    "medium": {"n": 15, ...},
    "hard":   {"n": 10, ...}
  },
  "by_chapter": {
    "chapter_8": {"n": 11, ...}
  }
}
```

### `report.md`
Full human-readable report including:
- Executive summary with plain-English performance descriptions
- Configuration used for the run
- Retrieval performance tables (overall + by difficulty)
- Answer quality tables (rubric, correctness, BLEU, faithfulness)
- Per-chapter breakdown
- Illustrative examples: best/worst answer, best/worst retrieval, faithful/unfaithful, rubric pass/fail, rubric-high-but-wrong
- Judge model name in the header

### `comparative_report.md` (AB test only)
- Side-by-side comparison table of all combinations across all key metrics
- Per-metric rankings from best to worst
- Win count per combination + recommendation
- Per-difficulty breakdown across all combinations

---

## Regenerating a Report Without Re-Running

If you want to regenerate the report after code changes without re-running the judge:

```bash
python3 regenerate_report.py --run_dir benchmark_results/chapter8_baseline
```

This reads `raw_results.jsonl` and `judge_results.jsonl` from the run directory, recomputes all metrics and examples, overwrites `metrics_detail.csv`, `metrics_summary.json`, and `report.md`. No LLM or TokenSmith calls are made.

---

## Crash Safety

Both `raw_results.jsonl` and `judge_results.jsonl` are written one line at a time immediately after each QAC is processed. If a run crashes, re-running the same command will automatically skip already-completed QACs (unless `--no_resume` is set).

For AB tests, each combination's results are similarly crash-safe, and already-completed combinations are skipped on resume.

---

## Notes on the Judge Model

- The judge uses **the same local Qwen GGUF model** as TokenSmith's generator (`gen_model` in `config/config.yaml`)
- Judge prompts use Qwen 2.5's native chat template (`<|im_start|>` / `<|im_end|>`)
- All judge calls use `temperature=0.0` for deterministic output
- JSON responses are parsed with fallback handling for markdown fences, truncated output, and unclosed braces
- If a judge call fails or returns unparseable output, the metric is recorded as `None` / `uncertain` and the pipeline continues