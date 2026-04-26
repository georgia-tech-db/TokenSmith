#!/usr/bin/env python3
"""
src/llm_benchmark_generation/main.py

Command-line entry point for the QAC benchmark generation pipeline.

Usage
-----
Generate QAC pairs:
    python src/llm_benchmark_generation/main.py generate \\
        --config  config/benchmark_qac_gen_config.yaml \\
        --chapters all \\
        --windows  first

Estimate cost before running:
    python src/llm_benchmark_generation/main.py estimate-cost \\
        --config config/benchmark_qac_gen_config.yaml

Sub-command reference
---------------------
generate
    --config PATH       Path to YAML config  (default: config/benchmark_qac_gen_config.yaml)
    --chapters LIST     Chapter numbers to process, e.g. "1 2 3" or "all"  (default: all)
    --windows MODE      Which windows to run: "first", "all", or "1 2" (1-based indices)
                        (default: first)
    --output NAME       Override output filename (without directory)
    --no-interactive    Skip "already done?" prompt and use non_interactive_default from config
    --dry-run           Print what would run without making any API calls

estimate-cost
    --config PATH       Path to YAML config
    --chapters LIST     Which chapters to include in the estimate (default: all)
    --windows MODE      Which windows (default: all)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

import yaml

# ── Make sure sibling modules are importable when invoked as a script ─────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from generator      import run_window, _load_jsonl
from llm_client     import fetch_model_pricing
from markdown_utils import (
    load_markdown, extract_pages, get_page_windows,
    load_book_info, resolve_chapters,
)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG   = "config/benchmark_qac_gen_config.yaml"
DEFAULT_MD_PATH  = "data/textbook--extracted_markdown.md"
DEFAULT_GEN_MODEL = "google/gemini-2.5-pro-preview"
DEFAULT_VER1      = "anthropic/claude-sonnet-4-5"
DEFAULT_VER2      = "openai/gpt-4o"


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load YAML config and apply defaults for any missing keys."""
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        print(f"[WARN] Config not found at {cfg_path} — using all defaults")
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_api_key(cfg: dict) -> str:
    """Get API key from config or environment variable."""
    key = cfg.get("openrouter_api_key", "").strip()
    if not key:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print(
            "ERROR: No OpenRouter API key found.\n"
            "  Set it in the config file under 'openrouter_api_key'\n"
            "  or export the OPENROUTER_API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


# ─────────────────────────────────────────────────────────────────────────────
# Output file resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_output_path(
    cfg:          dict,
    output_name:  str | None,
    no_interactive: bool,
    chapters_to_run: list[int],
) -> pathlib.Path:
    """
    Determine the output JSONL path.

    Priority:
      1. --output arg from CLI
      2. output_filename in config
      3. Auto-generated name: qac_{model}_{date}.jsonl
    """
    output_dir = pathlib.Path(cfg.get("output_dir", "synthetic_qac_data"))
    qac_subdir = output_dir / cfg.get("qac_subdir", "qacs")
    qac_subdir.mkdir(parents=True, exist_ok=True)

    if output_name:
        filename = output_name if output_name.endswith(".jsonl") else output_name + ".jsonl"
    elif cfg.get("output_filename", "").strip():
        filename = cfg["output_filename"].strip()
    else:
        model_slug = cfg.get("generation_model", DEFAULT_GEN_MODEL).replace("/", "_")
        date_str   = time.strftime("%Y-%m-%d")
        filename   = f"qac_{model_slug}_{date_str}.jsonl"

    out_path = qac_subdir / filename

    # Handle existing file
    if out_path.exists():
        existing = _load_jsonl(out_path)
        if not existing:
            return out_path

        done_chapters = sorted({r.get("chapter") for r in existing})
        overlap = [c for c in chapters_to_run if c in done_chapters]
        if not overlap:
            return out_path

        print(f"\n[!] Output file already exists: {out_path}")
        print(f"    Chapters already present: {done_chapters}")
        print(f"    Overlap with requested chapters: {overlap}")

        if no_interactive:
            default = cfg.get("non_interactive_default", "append")
            print(f"    --no-interactive: defaulting to '{default}'")
            if default == "overwrite":
                out_path.unlink()
            return out_path

        print("\n  What would you like to do?")
        print("  [a] Append new windows to the existing file (skip already-done ones)")
        print("  [o] Overwrite the existing file entirely")
        print("  [n] Enter a new filename")
        choice = input("  Choice (a/o/n): ").strip().lower()

        if choice == "o":
            out_path.unlink()
            print(f"  Deleted existing file. Will write fresh to {out_path}")
        elif choice == "n":
            new_name = input("  New filename (without directory): ").strip()
            if not new_name.endswith(".jsonl"):
                new_name += ".jsonl"
            out_path = qac_subdir / new_name
            print(f"  Using new file: {out_path}")
        else:
            print(f"  Appending to existing file: {out_path}")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Window selection
# ─────────────────────────────────────────────────────────────────────────────

def select_windows(
    chapters:    dict[int, dict],
    window_mode: str,
    window_size: int,
) -> dict[int, list[tuple[int, int]]]:
    """
    Build a mapping of chapter → list of (window_start, window_end) to process.

    window_mode options:
      "first"   — only the first window of each chapter
      "all"     — all windows of each chapter
      "1 2 3"   — 1-based window indices (same indices applied to all chapters)
    """
    result: dict[int, list[tuple[int, int]]] = {}

    for chap_num, bounds in chapters.items():
        all_windows = get_page_windows(
            bounds["content_start"], bounds["content_end"], window_size
        )
        if window_mode == "first":
            result[chap_num] = all_windows[:1]
        elif window_mode == "all":
            result[chap_num] = all_windows
        else:
            # Interpret as space-separated 1-based indices
            try:
                indices = [int(x) - 1 for x in window_mode.split()]
                result[chap_num] = [
                    all_windows[i] for i in indices if 0 <= i < len(all_windows)
                ]
            except ValueError:
                print(
                    f"[WARN] Could not parse --windows '{window_mode}'. "
                    "Defaulting to 'first'."
                )
                result[chap_num] = all_windows[:1]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: generate
# ─────────────────────────────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # Models
    gen_model = cfg.get("generation_model", DEFAULT_GEN_MODEL)
    ver1      = cfg.get("verifier_1_model", DEFAULT_VER1)
    ver2      = cfg.get("verifier_2_model", DEFAULT_VER2)
    api_key   = resolve_api_key(cfg)
    win_size  = int(cfg.get("window_size", 25))

    # Book info
    book_info   = load_book_info(cfg.get("book_info_path", ""))
    default_md  = cfg.get("default_markdown_path", DEFAULT_MD_PATH)
    md_path, all_chapters = resolve_chapters(book_info, default_md)

    # Chapter selection
    if args.chapters == "all":
        chapters_to_run = sorted(all_chapters.keys())
    else:
        try:
            chapters_to_run = [int(c) for c in args.chapters.split()]
        except ValueError:
            print(f"ERROR: Could not parse --chapters '{args.chapters}'", file=sys.stderr)
            sys.exit(1)

    # Validate
    invalid = [c for c in chapters_to_run if c not in all_chapters]
    if invalid:
        print(f"ERROR: Unknown chapter(s): {invalid}", file=sys.stderr)
        sys.exit(1)

    selected_chapters = {c: all_chapters[c] for c in chapters_to_run}

    # Window selection
    window_plan = select_windows(selected_chapters, args.windows, win_size)

    # Output path
    output_jsonl = resolve_output_path(
        cfg, args.output, args.no_interactive, chapters_to_run
    )
    log_dir = (
        pathlib.Path(cfg.get("output_dir", "synthetic_qac_data"))
        / cfg.get("log_subdir", "logs")
    )

    # Print plan
    total_windows = sum(len(v) for v in window_plan.values())
    print(f"\n{'='*60}")
    print(f"QAC GENERATION PLAN")
    print(f"{'='*60}")
    print(f"  Config          : {args.config}")
    print(f"  Generation model: {gen_model}")
    print(f"  Verifier 1      : {ver1}")
    print(f"  Verifier 2      : {ver2}")
    print(f"  Markdown        : {md_path}")
    print(f"  Output JSONL    : {output_jsonl}")
    print(f"  Log directory   : {log_dir}")
    print(f"  Window size     : {win_size} pages")
    print(f"  Total windows   : {total_windows}")
    print()
    for chap_num, windows in window_plan.items():
        bounds = all_chapters[chap_num]
        print(
            f"  C{chap_num:02d}  pages {bounds['content_start']}-{bounds['content_end']}"
            f"  ->  {len(windows)} window(s): "
            + "  ".join(f"[{ws}-{we}]" for ws, we in windows)
        )

    if args.dry_run:
        print("\n[DRY RUN] No API calls made.")
        return

    # Load markdown once
    print(f"\nLoading markdown from {md_path} ...")
    full_md, offsets = load_markdown(md_path)
    print(f"  {len(offsets)} pages indexed  ({len(full_md):,} chars)")

    # Run
    grand_passed = grand_manual = 0

    for chap_num, windows in window_plan.items():
        for ws, we in windows:
            try:
                pages_text = extract_pages(full_md, offsets, ws, we)
            except ValueError as e:
                print(f"\n[ERROR] Page extraction failed for C{chap_num} [{ws}-{we}]: {e}")
                continue

            records = run_window(
                chapter=chap_num,
                window_start=ws,
                window_end=we,
                pages_text=pages_text,
                full_md=full_md,
                offsets=offsets,
                output_jsonl=output_jsonl,
                log_dir=log_dir,
                generation_model=gen_model,
                verifier_1=ver1,
                verifier_2=ver2,
                api_key=api_key,
                cfg=cfg,
            )
            grand_passed += sum(1 for r in records if r.get("status") == "passed")
            grand_manual += sum(1 for r in records if r.get("status") == "manual_review")
            time.sleep(2.0)

    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"  Total passed       : {grand_passed}")
    print(f"  Total manual review: {grand_manual}")
    print(f"  Output             : {output_jsonl}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: estimate-cost
# ─────────────────────────────────────────────────────────────────────────────

def cmd_estimate_cost(args: argparse.Namespace) -> None:
    """
    Estimate the cost of running the pipeline by:
      1. Fetching live per-token pricing from the OpenRouter API.
      2. Sampling one random window and measuring actual prompt sizes.
      3. Extrapolating to the full set of requested windows.

    Assumptions for the extrapolation (conservative):
      - Each window generates 8 QAC pairs (midpoint of typical 6-10 range)
      - Each QAC goes through: critique + refine + 2x verify = 4 extra calls
      - Token counts for refinement/verification are estimated from prompt sizes
    """
    cfg     = load_config(args.config)
    api_key = resolve_api_key(cfg)
    win_size = int(cfg.get("window_size", 25))

    gen_model = cfg.get("generation_model", DEFAULT_GEN_MODEL)
    ver1      = cfg.get("verifier_1_model", DEFAULT_VER1)
    ver2      = cfg.get("verifier_2_model", DEFAULT_VER2)

    # Book info
    book_info  = load_book_info(cfg.get("book_info_path", ""))
    default_md = cfg.get("default_markdown_path", DEFAULT_MD_PATH)
    md_path, all_chapters = resolve_chapters(book_info, default_md)

    # Chapter / window selection
    if args.chapters == "all":
        chapters_to_run = sorted(all_chapters.keys())
    else:
        chapters_to_run = [int(c) for c in args.chapters.split()]
    selected = {c: all_chapters[c] for c in chapters_to_run if c in all_chapters}

    from prompts import build_qac_prompt, build_critique_prompt, build_verify_prompt
    from prompts import QAC_SYSTEM, CRITIQUE_SYSTEM, VERIFY_SYSTEM

    window_plan = select_windows(selected, args.windows, win_size)
    total_windows = sum(len(v) for v in window_plan.values())

    print(f"\nFetching pricing from OpenRouter API ...")
    pricing: dict[str, dict] = {}
    for model in [gen_model, ver1, ver2]:
        p = fetch_model_pricing(model, api_key)
        if p:
            pricing[model] = p
            print(
                f"  {model}: "
                f"${p['input_per_1m']:.2f}/1M in  "
                f"${p['output_per_1m']:.2f}/1M out"
            )
        else:
            pricing[model] = {"input_per_1m": 0.0, "output_per_1m": 0.0}
            print(f"  {model}: pricing unavailable (defaulting to $0)")

    # Sample a random window
    print(f"\nSampling a random window to measure prompt sizes ...")
    full_md, offsets = load_markdown(md_path)

    flat_windows = [
        (chap, ws, we)
        for chap, windows in window_plan.items()
        for ws, we in windows
    ]
    sample_chap, sample_ws, sample_we = random.choice(flat_windows)
    sample_text = extract_pages(full_md, offsets, sample_ws, sample_we)
    print(f"  Sampled C{sample_chap:02d} pages {sample_ws}-{sample_we}")

    # Measure prompt token estimates (chars / 4)
    gen_prompt   = build_qac_prompt(sample_chap, sample_ws, sample_we, sample_text)
    gen_in_tok   = len(gen_prompt) // 4
    gen_out_tok  = cfg.get("max_tokens_generation", 16000)

    # Dummy QAC for critique / verify prompt sizing
    dummy_qac = {
        "difficulty": "medium",
        "question":   "What is a transaction and what properties must it satisfy?",
        "mock_answer": "A transaction is a unit of work. It must be atomic, consistent, isolated, and durable.",
        "rubric":      ["Must define transaction", "Must mention ACID"],
        "gold_chunks": [
            "A transaction is a unit of program execution that accesses and possibly updates various data items.",
            "The transaction must satisfy the ACID properties: atomicity, consistency, isolation, and durability.",
        ],
        "chunk_relationships": {"composites": [], "substitutes": []},
    }
    crit_in_tok  = len(build_critique_prompt(dummy_qac)) // 4
    crit_out_tok = cfg.get("max_tokens_critique", 2000)
    ref_in_tok   = crit_in_tok + 200   # critique output feeds back in
    ref_out_tok  = cfg.get("max_tokens_refine", 4000)
    ver_in_tok   = len(build_verify_prompt(dummy_qac)) // 4
    ver_out_tok  = cfg.get("max_tokens_verify", 1500)

    # Assumptions
    qacs_per_window = 8   # typical midpoint

    def cost(model: str, in_tok: int, out_tok: int) -> float:
        p = pricing.get(model, {"input_per_1m": 0, "output_per_1m": 0})
        return (in_tok * p["input_per_1m"] + out_tok * p["output_per_1m"]) / 1_000_000

    cost_gen        = cost(gen_model, gen_in_tok, gen_out_tok)
    cost_critique   = cost(gen_model, crit_in_tok, crit_out_tok) * qacs_per_window
    cost_refine     = cost(gen_model, ref_in_tok,  ref_out_tok)  * qacs_per_window
    cost_verify_1   = cost(ver1,      ver_in_tok,  ver_out_tok)  * qacs_per_window
    cost_verify_2   = cost(ver2,      ver_in_tok,  ver_out_tok)  * qacs_per_window
    cost_per_window = cost_gen + cost_critique + cost_refine + cost_verify_1 + cost_verify_2
    total_cost      = cost_per_window * total_windows

    print(f"\n{'='*60}")
    print(f"COST ESTIMATE  ({total_windows} windows, ~{qacs_per_window} QACs/window)")
    print(f"{'='*60}")
    print(f"  Per window breakdown:")
    print(f"    Generation  ({gen_model}): ${cost_gen:.4f}")
    print(f"    Critique    ({gen_model}): ${cost_critique:.4f}")
    print(f"    Refine      ({gen_model}): ${cost_refine:.4f}")
    print(f"    Verify 1    ({ver1}):      ${cost_verify_1:.4f}")
    print(f"    Verify 2    ({ver2}):      ${cost_verify_2:.4f}")
    print(f"    ─────────────────────────────────────────")
    print(f"    Total per window:           ${cost_per_window:.4f}")
    print()
    print(f"  Total for {total_windows} windows:")
    print(f"    Estimate:    ${total_cost:.2f}")
    print(f"    Range:       ${total_cost * 0.7:.2f} – ${total_cost * 1.5:.2f}")
    print(f"    (range reflects ±30%/+50% variation in content density)")
    print(f"{'='*60}")
    print(
        "\nNote: 'estimate' assumes every QAC goes through all refinement steps.\n"
        "In practice ~50-60% of QACs require no refinement and skip Steps 3-5,\n"
        "so the actual cost is likely closer to the lower bound of the range."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python src/llm_benchmark_generation/main.py",
        description="QAC benchmark generation pipeline for database systems textbook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────────────────────────────────────
    gen_p = sub.add_parser(
        "generate",
        help="Generate and refine QAC pairs for one or more chapters.",
    )
    gen_p.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG})",
    )
    gen_p.add_argument(
        "--chapters", default="all",
        help='Chapters to process: "all" or space-separated numbers e.g. "1 2 3"',
    )
    gen_p.add_argument(
        "--windows", default="first",
        help=(
            'Windows to run: "first", "all", or 1-based indices e.g. "1 2"  '
            "(default: first)"
        ),
    )
    gen_p.add_argument(
        "--output", default=None,
        help="Override output filename (without directory path)",
    )
    gen_p.add_argument(
        "--no-interactive", action="store_true",
        help="Skip the already-done prompt and use non_interactive_default from config",
    )
    gen_p.add_argument(
        "--dry-run", action="store_true",
        help="Print the execution plan without making any API calls",
    )

    # ── estimate-cost ─────────────────────────────────────────────────────────
    est_p = sub.add_parser(
        "estimate-cost",
        help="Estimate API cost for a given config and chapter/window selection.",
    )
    est_p.add_argument("--config",   default=DEFAULT_CONFIG)
    est_p.add_argument("--chapters", default="all")
    est_p.add_argument(
        "--windows", default="all",
        help='Windows to include in estimate: "first", "all", or 1-based indices',
    )

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "estimate-cost":
        cmd_estimate_cost(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()