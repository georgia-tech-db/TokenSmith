"""
src/benchmark_eval/runner.py

TokenSmith interface for benchmark evaluation.

Loads TokenSmith's artifacts once, then runs each QAC question through
get_answer() in test mode (no console rendering, no chat history) and
returns the answer plus full retrieved chunk metadata for downstream
metric and judge evaluation.

Key design decisions
--------------------
- enable_history is always forced to False (each benchmark question is independent)
- system_prompt_mode is read from cfg.system_prompt_mode (whatever is in config.yaml)
  unless overridden by an AB test parameter
- is_test_mode=True makes get_answer() return (answer_str, chunks_info, hyde_query)
  without any console rendering
- Results are written to a JSONL file after each question for crash safety
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Optional

# ── Make sure the project root is importable ──────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import RAGConfig
from src.instrumentation.logging import get_logger
from src.main import get_answer
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    load_artifacts,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# These params require re-indexing or re-embedding and cannot be varied at
# query time. AB testing will reject them with a clear error message.
INVALID_AB_PARAMS = frozenset({
    "chunk_size_in_chars",
    "chunk_overlap",
    "chunk_mode",
    "embed_model",
    "embedding_model_context_window",
    "do_llm_chunk_reorg",
    "do_llm_coref_res",
    "use_indexed_chunks",
    "extracted_index_path",
    "page_to_chunk_map_path",
    "chunk_config",
})


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tokensmith_artifacts(cfg: RAGConfig, index_prefix: str = "textbook_index") -> Dict:
    """
    Load FAISS, BM25, chunks, sources, and metadata from the artifacts directory.
    Returns the artifacts dict expected by get_answer().
    Raises RuntimeError if loading fails (run index mode first).
    """
    try:
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
            artifacts_dir, index_prefix
        )
        retrievers = [
            FAISSRetriever(faiss_idx, cfg.embed_model),
            BM25Retriever(bm25_idx),
        ]
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(
                IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
            )
        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k),
        )
        return {
            "chunks": chunks,
            "sources": sources,
            "meta": meta,
            "retrievers": retrievers,
            "ranker": ranker,
        }
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TokenSmith artifacts: {exc}\n"
            "Run 'python main.py index' first."
        ) from exc


def make_args(cfg: RAGConfig) -> Namespace:
    """
    Fabricate the args Namespace that get_answer() expects.

    - system_prompt_mode is set to "" so it falls back to cfg.system_prompt_mode
      (the behaviour is: args.system_prompt_mode or cfg.system_prompt_mode)
    - double_prompt respects cfg.use_double_prompt
    - enable_history is always forced False for benchmark runs
    """
    return Namespace(
        system_prompt_mode="",       # falls back to cfg.system_prompt_mode
        double_prompt=False,          # overridden by cfg.use_double_prompt if set
    )


# ─────────────────────────────────────────────────────────────────────────────
# QAC file loading
# ─────────────────────────────────────────────────────────────────────────────

def load_verified_qacs(path: pathlib.Path) -> List[Dict]:
    """Load all QAC records from a verified JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Verified QAC file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Skipping malformed JSONL on line {lineno}: {exc}")
    # Only include approved or edited_approved records
    approved = [
        r for r in records
        if r.get("annotation_status") in ("approved", "edited_approved")
    ]
    skipped = len(records) - len(approved)
    if skipped:
        print(f"  [INFO] Skipped {skipped} non-approved records "
              f"(only 'approved'/'edited_approved' are used for benchmarking)")
    return approved


# ─────────────────────────────────────────────────────────────────────────────
# Single QAC runner
# ─────────────────────────────────────────────────────────────────────────────

def run_qac(
    qac:       Dict,
    cfg:       RAGConfig,
    artifacts: Dict,
    logger:    Any,
) -> Dict:
    """
    Run one QAC question through TokenSmith and collect the full result.

    Returns a dict containing:
        question          : str
        ts_answer         : str — TokenSmith's generated answer
        retrieved_chunks  : list[dict] — chunk_id, content, rank, scores
        hyde_query        : str | None — the HyDE-expanded query if used
        qac               : dict — the original QAC record
        error             : str | None — set if get_answer() raised
    """
    # Force history off for all benchmark runs
    cfg.enable_history = False

    question = qac.get("question", "")
    args     = make_args(cfg)

    try:
        result = get_answer(
            question=question,
            cfg=cfg,
            args=args,
            logger=logger,
            console=None,          # no console rendering in benchmark mode
            artifacts=artifacts,
            is_test_mode=True,     # returns (ans, chunks_info, hyde_query)
        )

        # get_answer returns different types depending on is_test_mode
        if isinstance(result, tuple):
            ts_answer, chunks_info, hyde_query = result
        else:
            # Fallback — shouldn't happen in test mode
            ts_answer  = result
            chunks_info = []
            hyde_query  = None

        return {
            "question":         question,
            "ts_answer":        ts_answer or "",
            "retrieved_chunks": chunks_info or [],
            "hyde_query":       hyde_query,
            "qac":              qac,
            "error":            None,
        }

    except Exception as exc:
        import traceback
        return {
            "question":         question,
            "ts_answer":        "",
            "retrieved_chunks": [],
            "hyde_query":       None,
            "qac":              qac,
            "error":            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark run
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    qacs:          List[Dict],
    cfg:           RAGConfig,
    artifacts:     Dict,
    output_jsonl:  pathlib.Path,
    run_label:     str = "benchmark",
    resume:        bool = True,
) -> List[Dict]:
    """
    Run all QACs through TokenSmith and collect raw results.

    Writes one JSONL line per QAC immediately after each run for crash safety.
    If resume=True, skips QACs whose record_id already exists in output_jsonl.

    Returns the full list of result dicts.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger()

    # Load existing results for crash recovery
    already_done: set[str] = set()
    existing: List[Dict]   = []
    if resume and output_jsonl.exists():
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        rid = rec.get("qac", {}).get("record_id", "")
                        if rid:
                            already_done.add(rid)
                            existing.append(rec)
                    except json.JSONDecodeError:
                        pass
        if already_done:
            print(f"  [RESUME] {len(already_done)} results already exist — skipping")

    total   = len(qacs)
    results = list(existing)

    print(f"\n[BENCHMARK] Running {total} questions  (label: {run_label})")

    with open(output_jsonl, "a", encoding="utf-8") as f:
        for i, qac in enumerate(qacs):
            rid = qac.get("record_id", f"q{i:04d}")
            if rid in already_done:
                continue

            diff = qac.get("difficulty", "?")
            print(
                f"  [{i+1:>3}/{total}] [{diff.upper():6}] "
                f"{qac.get('question', '')[:60]} ...",
                end=" ", flush=True,
            )

            t0     = time.time()
            result = run_qac(qac, cfg, artifacts, logger)
            elapsed = time.time() - t0

            n_chunks = len(result["retrieved_chunks"])
            error    = result.get("error")
            print(
                f"{'ERROR' if error else 'OK'}  "
                f"[{n_chunks} chunks | {elapsed:.1f}s]"
            )
            if error:
                print(f"         ERROR: {error[:120]}")

            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n[BENCHMARK] Done. {len(results)} results written to {output_jsonl}")
    return results