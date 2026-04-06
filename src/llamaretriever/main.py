"""
CLI entry point for the BookRAG pipeline.

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
from typing import Dict, Optional

from .config import LlamaIndexConfig


# ── Argument parsing ─────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BookRAG Pipeline")
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

    from .indexer import build_index, get_or_build_index
    from .models import build_embed_model, build_index_llm, build_llm

    Settings.llm = build_llm(cfg)
    Settings.embed_model = build_embed_model(cfg)
    Settings.chunk_size = cfg.chunk_size
    Settings.chunk_overlap = cfg.chunk_overlap

    index_llm = build_index_llm(cfg)
    if index_llm is not None:
        print(f"Index-time LLM: {cfg.index_llm_provider} / {cfg.index_llm_model}")

    if rebuild:
        build_index(cfg, index_llm=index_llm)
    else:
        get_or_build_index(cfg, index_llm=index_llm)
    print("\nIndexing complete.")


# ── Core query function ──────────────────────────────────────────────────


def get_answer(
    question: str,
    cfg: LlamaIndexConfig,
    artifacts: Dict,
    logger: Optional["QueryLogger"] = None,
    return_references: bool = False,
):
    """
    Run a single query through the BookRAG pipeline.

    Flow: classify → section-select → leaf-retrieve → synthesize.
    """
    from .agent import run_bookrag

    t0 = time.time()
    result = run_bookrag(
        question=question,
        llm=artifacts["llm"],
        section_retriever=artifacts["section_retriever"],
        leaf_index=artifacts["leaf_index"],
        tree=artifacts["tree"],
        reranker=artifacts["reranker"],
        cfg=cfg,
    )
    total_time = time.time() - t0

    # ── Print selected sections ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BOOKRAG  (type={}, {} sections, {} refs, {} LLM calls, {:.1f}s)".format(
        result.query_type,
        len(result.selected_sections),
        len(result.references),
        result.total_llm_calls,
        total_time,
    ))
    print("=" * 60)

    if result.selected_sections:
        tree = artifacts["tree"]
        print("  Selected sections:")
        for sid in result.selected_sections:
            sec = tree.sections.get(sid)
            if sec:
                print(f"    [{sid}] {' > '.join(sec.header_path)}")

    for ref in result.references:
        print(f"\n  [{ref.id}] ({ref.source})")
        print(f"     Path : {ref.header_path}")
        print(f"     {ref.passage[:200]}{'...' if len(ref.passage) > 200 else ''}")

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
                {
                    "id": r.id,
                    "passage": r.passage,
                    "section_id": r.section_id,
                    "chapter": r.chapter,
                    "section": r.section,
                    "subsection": r.subsection,
                    "header_path": r.header_path,
                    "source": r.source,
                }
                for r in result.references
            ],
            iterations=result.iterations,
            total_llm_calls=result.total_llm_calls,
            total_time_s=total_time,
            query_type=result.query_type,
            selected_sections=result.selected_sections,
        )

    if return_references:
        return result.answer, result.references
    return result.answer


# ── Initialization helper ────────────────────────────────────────────────


def init_artifacts(cfg: LlamaIndexConfig) -> tuple:
    """Load models + indices + tree + retrievers + reranker."""
    from llama_index.core import Settings

    from .indexer import load_index
    from .logger import QueryLogger
    from .models import build_embed_model, build_llm
    from .retriever import build_reranker, build_section_retriever

    llm = build_llm(cfg)
    embed_model = build_embed_model(cfg)
    Settings.llm = llm
    Settings.embed_model = embed_model

    leaf_index, section_index, tree = load_index(cfg)
    section_retriever = build_section_retriever(section_index, cfg)
    reranker = build_reranker(cfg)

    artifacts = {
        "llm": llm,
        "leaf_index": leaf_index,
        "section_retriever": section_retriever,
        "tree": tree,
        "reranker": reranker,
    }
    logger = QueryLogger(cfg)

    return artifacts, logger


# ── Chat session ─────────────────────────────────────────────────────────


def run_chat_session(cfg: LlamaIndexConfig) -> None:
    print("Initializing BookRAG pipeline...")
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
