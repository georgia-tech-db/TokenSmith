# LLM Benchmark Generation

Generates a synthetic QAC (Question-Answer-Chunk) benchmark dataset from a textbook markdown file using large language models via OpenRouter. Also includes a Streamlit-based manual review and annotation tool.

---

## Overview

The pipeline has three stages:

1. **QAC Generation** — A powerful LLM reads windows of textbook pages and produces question/answer/rubric/gold-chunk triplets at three difficulty levels (easy, medium, hard)
2. **Agentic Refinement** — Each generated QAC goes through self-critique, refinement, and dual-model verification automatically
3. **Manual Review** — A Streamlit app lets you annotate, edit, flag, and approve QAC pairs before they are used for benchmarking

---

## Directory Structure

```
src/llm_benchmark_generation/
    main.py                        ← CLI entry point
    generator.py                   ← Step 1: window-level generation orchestrator
    pipeline.py                    ← Steps 2-6: per-QAC refinement + verification
    prompts.py                     ← All LLM prompt builders
    llm_client.py                  ← OpenRouter HTTP client + JSON parsing
    markdown_utils.py              ← Page extraction and windowing
    verification.py                ← Substring verification for gold chunks
    review_tool.py                 ← Streamlit manual annotation app
    book_chapters_page_info.json   ← Chapter page boundary definitions

config/
    benchmark_qac_gen_config.yaml  ← All pipeline configuration

synthetic_qac_data/
    qacs/                          ← Generated QAC JSONL files (one per chapter)
    manually_verified/             ← Verified/annotated QAC files from review tool
    logs_for_qac_generation/       ← Per-window pipeline logs (JSON)
```

---

## Prerequisites

```bash
pip install httpx pyyaml python-dotenv
```

Set your OpenRouter API key either in `.env` at the project root:
```
OPENROUTER_API_KEY=sk-or-...
```
or in `config/benchmark_qac_gen_config.yaml` under `openrouter_api_key`.

---

## Configuration

**`config/benchmark_qac_gen_config.yaml`**

| Field | Default | Description |
|---|---|---|
| `openrouter_api_key` | `""` | API key (leave blank to use env var) |
| `generation_model` | `google/gemini-2.5-pro-preview` | Model for QAC generation and refinement |
| `verifier_1_model` | `anthropic/claude-sonnet-4-5` | First verification model |
| `verifier_2_model` | `openai/gpt-4o` | Second verification model |
| `book_info_path` | `src/llm_benchmark_generation/book_chapters_page_info.json` | Chapter boundary definitions |
| `default_markdown_path` | `data/textbook--extracted_markdown.md` | Textbook markdown source |
| `window_size` | `25` | Pages per generation window |
| `max_tokens_generation` | `16000` | Output token budget for QAC generation |
| `max_tokens_critique` | `2000` | Output token budget for self-critique |
| `max_tokens_refine` | `4000` | Output token budget for refinement |
| `max_tokens_verify` | `1500` | Output token budget per verifier call |
| `http_connect_timeout` | `10` | TCP connect timeout in seconds |
| `http_read_timeout` | `180` | Response read timeout in seconds |
| `max_retries` | `3` | LLM call retry attempts |
| `retry_delay_base` | `10` | Base retry delay in seconds (multiplied by attempt number) |
| `output_dir` | `synthetic_qac_data` | Root output directory |
| `qac_subdir` | `qacs` | Subdirectory for QAC JSONL files |
| `log_subdir` | `logs` | Subdirectory for pipeline logs |
| `output_filename` | `""` | Override output filename (auto-generated if blank) |
| `non_interactive_default` | `append` | Behaviour when file exists and `--no-interactive` is set (`append` or `overwrite`) |

---

## Book Chapter Definitions

**`src/llm_benchmark_generation/book_chapters_page_info.json`**

Defines the page ranges for each chapter. The `--- Page N ---` markers in the markdown are used for extraction.

```json
{
  "markdown_path": "data/textbook--extracted_markdown.md",
  "chapters": {
    "1": {"content_start": 43, "content_end": 91, "rt_start": 91, "rt_end": 94},
    "2": {"content_start": 101, "content_end": 131, "rt_start": 131, "rt_end": 132}
  }
}
```

If this file is missing or has no `chapters` key, the entire markdown is treated as one chapter spanning all found page markers.

---

## Sub-commands

### `generate` — Run QAC Generation

```bash
python3 src/llm_benchmark_generation/main.py generate [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/benchmark_qac_gen_config.yaml` | Path to config YAML |
| `--chapters` | `all` | Chapters to process: `all` or space-separated numbers e.g. `"1 2 3"` |
| `--windows` | `first` | Windows to run: `first`, `all`, or 1-based indices e.g. `"1 2"` |
| `--output` | auto | Override output filename (without directory) |
| `--no-interactive` | off | Skip the already-done prompt, use `non_interactive_default` from config |
| `--dry-run` | off | Print execution plan without making any API calls |

**Windowing logic:** Each chapter is split into non-overlapping 25-page windows. The final window always ends at the last chapter page and back-fills if needed to maintain exactly 25 pages. If a chapter is shorter than 25 pages it becomes a single window.

**Output files** (written to `synthetic_qac_data/qacs/`):
```
chapter_01_qac_google_gemini-2.5-pro-preview.jsonl
chapter_02_qac_google_gemini-2.5-pro-preview.jsonl
...
```
Auto-generated filename format: `qac_{model}_{date}.jsonl` if `--output` is not specified.

**Log files** (written to `synthetic_qac_data/logs/`):
```
c01_w43_67_pipeline.json     ← full per-window agentic pipeline log
c01_w68_91_pipeline.json
...
```

**Crash safety:** Each QAC is written to the JSONL immediately after processing. Re-running the same command resumes from where it left off, skipping already-completed windows.

#### Examples

```bash
# Run first window only for all chapters (good for testing)
python3 src/llm_benchmark_generation/main.py generate \
    --chapters all --windows first

# Run all windows for chapters 8, 9, and 10
python3 src/llm_benchmark_generation/main.py generate \
    --chapters "8 9 10" --windows all

# Dry run — see what would run without spending API credits
python3 src/llm_benchmark_generation/main.py generate \
    --chapters all --windows all --dry-run

# Run with a specific output filename, no interactive prompts
python3 src/llm_benchmark_generation/main.py generate \
    --chapters all --windows first \
    --output chapter_first_windows_v1 \
    --no-interactive
```

---

### `estimate-cost` — Estimate API Cost Before Running

```bash
python3 src/llm_benchmark_generation/main.py estimate-cost [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/benchmark_qac_gen_config.yaml` | Path to config YAML |
| `--chapters` | `all` | Chapters to include in the estimate |
| `--windows` | `all` | Windows to include in the estimate |

Fetches live per-token pricing from the OpenRouter `/api/v1/models` endpoint, samples one random window to measure prompt sizes, and extrapolates to the full requested run. Prints a per-model cost breakdown and a total estimate with a ±range.

```bash
# Estimate cost for first window of all chapters
python3 src/llm_benchmark_generation/main.py estimate-cost \
    --chapters all --windows first

# Estimate cost for all windows of chapters 1-5
python3 src/llm_benchmark_generation/main.py estimate-cost \
    --chapters "1 2 3 4 5" --windows all
```

---

## Agentic Refinement Pipeline

Each generated QAC pair automatically goes through up to 6 steps. Pages are sent to the LLM at most **twice** per window (Step 1 always, Step 3b only if chunks are insufficient).

| Step | Model | Input | Purpose |
|---|---|---|---|
| 1 — Generate | Gemini | Full pages | Produce all QAC pairs for the window |
| Substring check | None (deterministic) | QAC + source pages | Verify gold chunks exist verbatim in pages |
| 2 — Self-critique | Gemini | QAC only | Structured flags: phrasing, terminology, redundant chunks, rubric issues |
| 3 — Refine | Gemini | QAC + flags | Fix all flagged issues |
| 3b — Fallback | Gemini | QAC + targeted pages | Only fires if `chunks_insufficient=true`; finds missing chunk or scopes question down |
| 4 — Verify | Claude + GPT | QAC only | Dual-model verification on 5 criteria |
| 5 — Second refine | Gemini | QAC + verifier flags | Fix remaining issues flagged by verifiers |
| 6 — Final verify | Claude + GPT | QAC only | Last verification pass |

**QAC status after pipeline:**
- `passed` — cleared all verification steps
- `manual_review` — failed substring check OR still flagged after Step 6

---

## QAC Output Format

Each line in the output JSONL is one QAC record:

```json
{
  "chapter": 8,
  "window_pages": [561, 585],
  "generation_model": "google/gemini-2.5-pro-preview",
  "status": "passed",
  "difficulty": "easy",
  "question": "What is a key-value store?",
  "mock_answer": "A key-value store is ...",
  "rubric": [
    "Must define a key-value store as ...",
    "Must mention that ..."
  ],
  "gold_chunks": [
    "A key-value store is a simple data model ...",
    "Each key is associated with exactly one value ..."
  ],
  "chunk_relationships": {
    "composites": [["sentence A", "sentence B"]],
    "substitutes": []
  },
  "verification": null
}
```

---

## Manual Review Tool

```bash
streamlit run src/llm_benchmark_generation/review_tool.py
```

Run from the **project root**. The app reads from `synthetic_qac_data/qacs/` and writes to `synthetic_qac_data/manually_verified/`.

### Features

- **File selector** — choose which generated JSONL file to work with
- **Table view** — all QAC pairs with colour-coded difficulty and annotation status badges
- **Filters** — filter by chapter, difficulty, annotation status, generation status
- **Next unannotated** button — jump to the first unreviewed record matching current filters
- **Next record** button — advance to the next record in the filtered list
- **Detail panel** — editable question, mock answer, difficulty, rubric, and gold chunks
- **Chunk flags** — per-chunk flags: Gold, Optional, Example/Analogy, Confusing
- **Rubric flags** — per-criterion flags: Gold, Optional, Example/Analogy
- **Chunk relationships** — define composite (all needed) and substitute (any one sufficient) groups using chunk multiselect
- **Substring indicator** — live 🟢/🔴 check on each gold chunk against the source pages
- **Load Original** — restore the pre-annotation version of a reviewed record

### Annotation Actions

| Button | Status Set | Description |
|---|---|---|
| ✅ Approve | `approved` or `edited_approved` | Save as-is; auto-detects edits |
| ✏️ Edit & Approve | `edited_approved` | Always saves as edited |
| ❌ Reject | `rejected` | Mark as rejected, save to file |
| ⏭ Skip | `skipped` | Mark as skipped, advance to next unannotated |
| → Next | — | Move to next record without saving |

### Verified Output Format

Each saved record in `manually_verified/Verified--{source_filename}` includes all original QAC fields plus:

```json
{
  "record_id": "c08_w561_585_q003",
  "annotation_status": "edited_approved",
  "annotator_note": "Simplified the question wording",
  "annotation_timestamp": "2025-04-25T14:30:00",
  "edited_fields": {"question": "...", "rubric": [...]},
  "rubric_flags": [
    {"gold": true, "optional": false, "example_analogy": false}
  ],
  "gold_chunk_flags": [
    {"gold": true, "optional": false, "example_analogy": false, "confusing": false}
  ]
}
```

Only `approved` and `edited_approved` records are used by the benchmark evaluation pipeline.

---

## Models Used

All models are hardcoded in `pipeline.py` and `llm_client.py` — they are not surfaced in the config because they are not intended to vary:

| Role | Model |
|---|---|
| Generation + Refinement | `google/gemini-2.5-pro-preview` |
| Verifier 1 | `anthropic/claude-sonnet-4-5` |
| Verifier 2 | `openai/gpt-4o` |

All calls go through OpenRouter at `https://openrouter.ai/api/v1/chat/completions`.