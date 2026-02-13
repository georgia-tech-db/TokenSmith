"""
CLI entry point for the LlamaRetriever pipeline.

Uses an iterative evidence-curation agent instead of single-shot generation.

Usage:
    python -m src.llamaretriever index
    python -m src.llamaretriever index --rebuild
    python -m src.llamaretriever chat
    python -m src.llamaretriever query "What is normalization?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from .config import LlamaIndexConfig

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."


# ── Argument parsing ─────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LlamaRetriever Pipeline")
    parser.add_argument("mode", choices=["index", "chat", "query"])
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--gen-model", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--gpu-layers", type=int, default=None)
    return parser.parse_args()


_CONFIG_PATH = "config/config.yaml"


def build_config(args: argparse.Namespace) -> LlamaIndexConfig:
    cfg = LlamaIndexConfig.from_yaml(_CONFIG_PATH)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.gen_model:
        cfg.gen_model = args.gen_model
    if args.embed_model:
        cfg.embed_model = args.embed_model
    if args.no_rerank:
        cfg.use_reranker = False
    if args.gpu_layers is not None:
        cfg.n_gpu_layers = args.gpu_layers
    return cfg


# ── Index mode ───────────────────────────────────────────────────────────


def run_index_mode(cfg: LlamaIndexConfig, rebuild: bool) -> None:
    from llama_index.core import Settings
    from .models import build_llm, build_embed_model
    from .indexer import get_or_build_index, build_index

    Settings.llm = build_llm(cfg)
    Settings.embed_model = build_embed_model(cfg)
    Settings.chunk_size = cfg.chunk_size
    Settings.chunk_overlap = cfg.chunk_overlap

    if rebuild:
        build_index(cfg)
    else:
        get_or_build_index(cfg)
    print("\nIndexing complete.")


# ── Core query function ──────────────────────────────────────────────────


def get_answer(
    question: str,
    cfg: LlamaIndexConfig,
    artifacts: Dict,
    logger: Optional["QueryLogger"] = None,
) -> str:
    """
    Run a single query through the evidence-curation agent.

    Flow: retrieve -> sentence split -> LLM curates evidence -> LLM synthesizes cited answer.
    """
    from .agent import run_agent

    t0 = time.time()
    result = run_agent(
        question=question,
        retriever=artifacts["retriever"],
        reranker=artifacts["reranker"],
        llm=artifacts["llm"],
        max_curate_steps=cfg.max_curate_steps,
    )
    total_time = time.time() - t0

    # ── Print references ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVIDENCE  ({} references, {} LLM calls, {:.1f}s)".format(
        len(result.references), result.total_llm_calls, total_time,
    ))
    print("=" * 60)
    for ref in result.references:
        print(f"\n  [{ref.id}] § {ref.section}")
        print(f"     {ref.passage}")

    # ── Print answer ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ANSWER")
    print("=" * 60)
    print(f"\n{result.answer}\n")
    print("=" * 60 + "\n")

    # ── Log ───────────────────────────────────────────────────────────────
    if logger:
        logger.log_query(
            question=question,
            answer=result.answer,
            references=[
                {"id": r.id, "passage": r.passage, "section": r.section, "source": r.source}
                for r in result.references
            ],
            iterations=result.iterations,
            total_llm_calls=result.total_llm_calls,
            total_time_s=total_time,
        )

    return result.answer


# ── Initialization helper ────────────────────────────────────────────────


def init_artifacts(cfg: LlamaIndexConfig) -> tuple:
    """Load models + index + retriever + reranker. Fail fast on any error."""
    from llama_index.core import Settings
    from .models import build_llm, build_embed_model
    from .indexer import load_index
    from .retriever import build_retriever, build_reranker
    from .logger import QueryLogger

    llm = build_llm(cfg)
    embed_model = build_embed_model(cfg)
    Settings.llm = llm
    Settings.embed_model = embed_model

    index = load_index(cfg)
    retriever = build_retriever(index, cfg)
    reranker = build_reranker(cfg)

    artifacts = {"llm": llm, "retriever": retriever, "reranker": reranker}
    logger = QueryLogger(cfg)

    return artifacts, logger


# ── Chat session ─────────────────────────────────────────────────────────


def run_chat_session(cfg: LlamaIndexConfig) -> None:
    """Load artifacts once, then run the interactive chat loop."""
    print("Initializing LlamaIndex pipeline...")
    artifacts, logger = init_artifacts(cfg)
    print("Initialization complete. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            q = input("Ask > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            get_answer(q, cfg, artifacts, logger)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            raise


# ── Single query mode ────────────────────────────────────────────────────


def run_query_mode(cfg: LlamaIndexConfig, question: str) -> None:
    """Run a single query and exit."""
    artifacts, logger = init_artifacts(cfg)
    get_answer(question, cfg, artifacts, logger)


# ── Entry point ──────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    if args.mode == "index":
        run_index_mode(cfg, rebuild=args.rebuild)
    elif args.mode == "chat":
        run_chat_session(cfg)
    elif args.mode == "query":
        if not args.question:
            print("Error: 'query' mode requires a question.", file=sys.stderr)
            sys.exit(1)
        run_query_mode(cfg, args.question)


if __name__ == "__main__":
    main()
