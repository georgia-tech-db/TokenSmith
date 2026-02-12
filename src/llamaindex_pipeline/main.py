"""
CLI entry point for the LlamaIndex RAG pipeline.

Usage:
    python -m src.llamaindex.main index              # build index
    python -m src.llamaindex.main index --rebuild     # force rebuild
    python -m src.llamaindex.main chat                # interactive chat
    python -m src.llamaindex.main query "What is X?"  # single query
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LlamaIndex RAG Pipeline for TokenSmith"
    )
    parser.add_argument(
        "mode",
        choices=["index", "chat", "query"],
        help="'index' to build, 'chat' for interactive, 'query' for single question",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="question string (only for 'query' mode)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to YAML config file (optional, uses defaults otherwise)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force rebuild the index even if one exists",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="override data directory",
    )
    parser.add_argument(
        "--gen-model",
        default=None,
        help="override path to generation model (.gguf)",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="override HuggingFace embedding model name",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="disable cross-encoder reranking",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="number of GPU layers for llama.cpp (-1 = all)",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace):
    """Build config from YAML + CLI overrides."""
    from .config import LlamaIndexConfig

    if args.config:
        cfg = LlamaIndexConfig.from_yaml(args.config)
    else:
        cfg = LlamaIndexConfig()

    # Apply CLI overrides
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.gen_model:
        cfg.gen_model_path = args.gen_model
    if args.embed_model:
        cfg.embed_model_name = args.embed_model
    if args.no_rerank:
        cfg.use_reranker = False
    if args.gpu_layers is not None:
        cfg.n_gpu_layers = args.gpu_layers

    return cfg


def run_index(args: argparse.Namespace) -> None:
    """Build the document index."""
    cfg = _build_config(args)
    from .pipeline import RAGPipeline

    pipe = RAGPipeline(config=cfg)
    pipe.index(force_rebuild=args.rebuild)
    print("\nIndexing complete.")


def _print_chunks(chunks: list, label: str = "Retrieved Chunks") -> None:
    """Pretty-print the chunks used for answering."""
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {label}  ({len(chunks)} chunk(s))")
    print(sep)
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "")
        print(f"\n  [{i}] score={score_str}")
        if meta:
            # Show the most useful metadata keys
            for key in ("Header_2", "Header_1", "file_name", "section"):
                if key in meta:
                    print(f"      {key}: {meta[key]}")
        print(f"      text ({len(text)} chars):")
        # Print the chunk text indented
        preview = text
        for line in preview.splitlines():
            print(f"        {line}")
    print(f"\n{sep}\n")


def run_query(args: argparse.Namespace) -> None:
    """Run a single query."""
    if not args.question:
        print("Error: 'query' mode requires a question argument.", file=sys.stderr)
        sys.exit(1)

    cfg = _build_config(args)
    from .pipeline import RAGPipeline

    pipe = RAGPipeline(config=cfg)
    pipe.load()

    result = pipe.query_verbose(args.question)

    # Print the chunks that were used
    _print_chunks(result["chunks"])

    # Print the answer
    print(f"Answer:\n{result['answer']}\n")


def run_chat(args: argparse.Namespace) -> None:
    """Interactive chat loop."""
    cfg = _build_config(args)
    from .pipeline import RAGPipeline

    pipe = RAGPipeline(config=cfg)
    pipe.load()

    print("\nLlamaIndex RAG Pipeline — Interactive Chat")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            question = input("Ask > ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            try:
                source_nodes, response_gen = pipe.stream_query(question)

                # Print the chunks used for context
                chunks = [
                    {
                        "score": n.score,
                        "text": n.text,
                        "metadata": n.metadata,
                    }
                    for n in source_nodes
                ]
                _print_chunks(chunks)

                # Stream the answer
                print("Answer:")
                for token in response_gen:
                    print(token, end="", flush=True)
                print("\n")
            except Exception:
                # Fall back to verbose (non-streaming) query
                result = pipe.query_verbose(question)
                _print_chunks(result["chunks"])
                print(f"Answer:\n{result['answer']}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main() -> None:
    args = parse_args()

    if args.mode == "index":
        run_index(args)
    elif args.mode == "query":
        run_query(args)
    elif args.mode == "chat":
        run_chat(args)


if __name__ == "__main__":
    main()
