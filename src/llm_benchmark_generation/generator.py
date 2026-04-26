"""
src/llm_benchmark_generation/generator.py

Orchestrates generation and refinement for one chapter window.

Step 1 : Call the LLM with the full page text to generate all QAC pairs.
Steps 2-6 : Call pipeline.run_qac_pipeline() for each generated pair.

Writes:
  - Per-QAC records to the output JSONL (appended atomically after each QAC)
  - A full window log JSON to the log directory (updated after every QAC)
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

from llm_client    import call_llm, parse_json_response
from pipeline      import run_qac_pipeline
from prompts       import QAC_SYSTEM, build_qac_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Log helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_window_log(chapter: int, window_start: int, window_end: int) -> dict:
    return {
        "chapter":      chapter,
        "window_pages": [window_start, window_end],
        "generation":   None,
        "qac_logs":     [],
    }


def _save_window_log(log: dict, log_path: pathlib.Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def get_window_log_path(log_dir: pathlib.Path, chapter: int,
                        window_start: int, window_end: int) -> pathlib.Path:
    return log_dir / f"c{chapter:02d}_w{window_start}_{window_end}_pipeline.json"


# ─────────────────────────────────────────────────────────────────────────────
# Window pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_window(
    chapter:          int,
    window_start:     int,
    window_end:       int,
    pages_text:       str,
    full_md:          str,
    offsets:          dict[int, int],
    output_jsonl:     pathlib.Path,
    log_dir:          pathlib.Path,
    generation_model: str,
    verifier_1:       str,
    verifier_2:       str,
    api_key:          str,
    cfg:              dict,
) -> list[dict]:
    """
    Run the full generation + refinement pipeline for one window.

    Returns the list of final QAC records that were written to output_jsonl.
    Skips generation if this (chapter, window_start, window_end) combination
    already exists in output_jsonl.
    """
    print(f"\n{'='*60}")
    print(f"WINDOW  Chapter {chapter}  pages {window_start}-{window_end}")
    print(f"{'='*60}")

    log_path   = get_window_log_path(log_dir, chapter, window_start, window_end)
    window_log = _init_window_log(chapter, window_start, window_end)

    # ── Check if already done ─────────────────────────────────────────────────
    existing     = _load_jsonl(output_jsonl)
    done_windows = {
        (r["chapter"], r["window_pages"][0], r["window_pages"][1])
        for r in existing
    }
    if (chapter, window_start, window_end) in done_windows:
        print("Window already in output file — skipping")
        return [
            r for r in existing
            if r.get("chapter") == chapter
            and r.get("window_pages") == [window_start, window_end]
        ]

    # ── Step 1: Generate ──────────────────────────────────────────────────────
    print(f"\n[Step 1] Generating QAC pairs ...", end=" ", flush=True)
    gen_prompt     = build_qac_prompt(chapter, window_start, window_end, pages_text)
    raw_g, usage_g = call_llm(
        [{"role": "system", "content": QAC_SYSTEM},
         {"role": "user",   "content": gen_prompt}],
        model=generation_model,
        api_key=api_key,
        max_tokens=cfg.get("max_tokens_generation", 16000),
        connect_timeout=cfg.get("http_connect_timeout", 10),
        read_timeout=cfg.get("http_read_timeout", 180),
        retries=cfg.get("max_retries", 3),
        retry_delay=cfg.get("retry_delay_base", 10),
    )

    # Log generation immediately (before parse) for crash safety
    window_log["generation"] = {
        "stage":     "generation",
        "model":     generation_model,
        "prompt":    gen_prompt,
        "response":  raw_g,
        "usage":     usage_g,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_window_log(window_log, log_path)

    if raw_g is None:
        print("FAILED (LLM error)")
        return []

    gen_parsed = parse_json_response(raw_g)
    if not isinstance(gen_parsed, dict) or "qac_pairs" not in gen_parsed:
        print("FAILED (JSON parse)")
        print(f"--- RAW (first 400 chars) ---\n{(raw_g or '')[:400]}\n---")
        return []

    raw_pairs = gen_parsed["qac_pairs"]
    print(f"[Step 1] Generated {len(raw_pairs)} QAC pairs")

    # ── Steps 2-6: Process each pair ─────────────────────────────────────────
    final_records: list[dict] = []
    for i, pair in enumerate(raw_pairs):
        final_qac, qac_log = run_qac_pipeline(
            qac=pair,
            qac_index=i,
            pages_text=pages_text,
            full_md=full_md,
            offsets=offsets,
            window_start=window_start,
            window_end=window_end,
            generation_model=generation_model,
            verifier_1=verifier_1,
            verifier_2=verifier_2,
            api_key=api_key,
            cfg=cfg,
        )

        record = {
            "chapter":          chapter,
            "window_pages":     [window_start, window_end],
            "generation_model": generation_model,
            "status":           qac_log["status"],
            **final_qac,
        }
        final_records.append(record)

        # Append to JSONL immediately (atomic per-QAC write for crash safety)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Update window log after every QAC
        window_log["qac_logs"].append(qac_log)
        _save_window_log(window_log, log_path)

        time.sleep(0.5)

    passed = sum(1 for r in final_records if r.get("status") == "passed")
    manual = sum(1 for r in final_records if r.get("status") == "manual_review")
    print(
        f"\n[DONE] {len(final_records)} QACs — "
        f"passed={passed}  manual_review={manual}"
    )
    print(f"[LOG]  {log_path}")

    return final_records


# ─────────────────────────────────────────────────────────────────────────────
# JSONL helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records