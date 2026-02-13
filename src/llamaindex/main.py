"""
CLI entry point for the LlamaIndex RAG pipeline.

Mirrors the flow of src/main.py:
  1. Parse args -> load config
  2. Index mode : load docs -> chunk -> embed -> persist
  3. Chat mode  : load artifacts once -> chat loop (retrieve -> rerank -> generate)

Usage:
    python -m src.llamaindex index
    python -m src.llamaindex index --rebuild
    python -m src.llamaindex chat
    python -m src.llamaindex query "What is normalization?"
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional

from .config import LlamaIndexConfig

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."


# ── Argument parsing ─────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LlamaIndex RAG Pipeline")
    parser.add_argument("mode", choices=["index", "chat", "query"])
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--gen-model", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--gpu-layers", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LlamaIndexConfig:
    cfg = LlamaIndexConfig.from_yaml(args.config) if args.config else LlamaIndexConfig()
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
    Run a single query through the pipeline.  Mirrors src/main.py:get_answer().

    Flow: retrieve (vector + BM25 via RRF) -> rerank -> generate (stream).
    Logs question + chunks + answer to JSON.
    """
    from llama_index.core import QueryBundle
    from llama_index.core.response_synthesizers import get_response_synthesizer

    retriever = artifacts["retriever"]
    reranker = artifacts["reranker"]
    llm = artifacts["llm"]

    # Step 1: Retrieve (RRF fusion of vector + BM25)
    t0 = time.time()
    query_bundle = QueryBundle(query_str=question)
    nodes = retriever.retrieve(query_bundle)

    # Step 2: Rerank with cross-encoder
    if reranker is not None:
        nodes = reranker.postprocess_nodes(nodes, query_bundle)

    retrieval_time = time.time() - t0

    if not nodes:
        print(f"\n{ANSWER_NOT_FOUND}\n")
        return ANSWER_NOT_FOUND

    # Collect chunk info for logging
    chunks_for_log = []
    for rank, node in enumerate(nodes, 1):
        chunks_for_log.append({
            "rank": rank,
            "score": round(float(node.score), 4) if node.score is not None else None,
            "text": node.text,
            "metadata": {k: str(v) for k, v in node.metadata.items()},
            "char_len": len(node.text),
        })

    # Step 3: Generate (streaming)
    t1 = time.time()
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="compact",
        streaming=True,
    )
    streaming_response = synthesizer.synthesize(question, nodes)

    print("\n" + "=" * 60)
    print("  ANSWER")
    print("=" * 60 + "\n")

    answer_text = ""
    for token in streaming_response.response_gen:
        print(token, end="", flush=True)
        answer_text += token
    print("\n\n" + "=" * 60 + "\n")

    generation_time = time.time() - t1

    # Log to JSON
    if logger:
        logger.log_query(
            question=question,
            answer=answer_text,
            chunks=chunks_for_log,
            retrieval_time_s=retrieval_time,
            generation_time_s=generation_time,
        )

    return answer_text


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
