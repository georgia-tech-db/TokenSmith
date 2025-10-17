import argparse
import pathlib
import sys
from typing import Dict, Optional

from src.config import QueryPlanConfig
from src.generator import answer
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Welcome to TokenSmith!"
    )

    # Required arguments
    parser.add_argument(
        "mode",
        choices=["index", "chat"],
        help="operation mode: 'index' to build index, 'chat' to query"
    )

    # Common arguments
    parser.add_argument(
        "--config",
        help="path to custom config file (uses default if not specified)"
    )
    parser.add_argument(
        "--pdf_dir",
        default="data/chapters/",
        help="directory containing PDF files (default: %(default)s)"
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="prefix for generated index files (default: %(default)s)"
    )
    parser.add_argument(
        "--model_path",
        help="path to generation model (uses config default if not specified)"
    )

    # Indexing-specific arguments
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument(
        "--pdf_range",
        metavar="START-END",
        help="specific range of PDFs to index (e.g., '27-33')"
    )
    indexing_group.add_argument(
        "--keep_tables",
        action="store_true",
        help="include tables in the index"
    )
    indexing_group.add_argument(
        "--visualize",
        action="store_true",
        help="generate visualizations during indexing"
    )

    return parser.parse_args()


def find_fallback_config() -> Optional[QueryPlanConfig]:
    """
    Looks for a fallback config file in standard locations.

    Returns:
        A QueryPlanConfig object if a file is found, otherwise None.
    """
    search_paths = [
        pathlib.Path("~/.config/tokensmith/config.yaml").expanduser(),
        pathlib.Path("~/.config/tokensmith/config.yml").expanduser(),
        pathlib.Path("config/config.yaml"),
    ]
    for path in search_paths:
        if path.exists():
            return QueryPlanConfig.from_yaml(path)
    return None


def run_index_mode(args: argparse.Namespace, cfg: QueryPlanConfig):
    """Handles the logic for building the index."""

    # Robust range filtering
    try:
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end + 1)] # Inclusive range
            print(f"Indexing PDFs in range: {start}-{end}")
        else:
            pdf_paths = None
    except ValueError:
        print(f"ERROR: Invalid format for --pdf_range. Expected 'start-end', but got '{args.pdf_range}'.")
        sys.exit(1)
    
    strategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)

    build_index(
        markdown_file="data/book_without_image.md",
        cfg=cfg,
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
    )


def run_chat_session(args: argparse.Namespace, cfg: QueryPlanConfig):
    """
    Initializes artifacts and runs the main interactive chat loop.
    """
    logger = get_logger()
    # planner = HeuristicQueryPlanner(cfg)

    # Load artifacts, initialize retrievers and rankers once before the loop.
    print("Welcome to Tokensmith! Initializing chat...")
    try:
        # Disabled till we fix the core pipeline
        # cfg = planner.plan(q)
        artifacts_dir = cfg.make_artifacts_directory()
        faiss_index, bm25_index, chunks, sources = load_artifacts(artifacts_dir=artifacts_dir, index_prefix=args.index_prefix)

        retrievers = [
            FAISSRetriever(faiss_index, cfg.embed_model),
            BM25Retriever(bm25_index)
        ]
        ranker = EnsembleRanker(
            ensemble_method =cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k)
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize chat artifacts: {e}")
        print("Please ensure you have run 'index' mode first.")
        sys.exit(1)

    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            logger.log_query_start(q)

            # Step 1: Retrieval
            pool_n = max(cfg.pool_size, cfg.top_k + 10)
            raw_scores: Dict[str, Dict[int, float]] = {}
            for retriever in retrievers:
                raw_scores[retriever.name] = retriever.get_scores(q, pool_n, chunks)
            # TODO: Fix retrieval logging.

            # Step 2: Ranking
            ordered = ranker.rank(raw_scores=raw_scores)
            topk_idxs = apply_seg_filter(cfg, chunks, ordered)
            logger.log_chunks_used(topk_idxs, chunks, sources)

            ranked_chunks = [chunks[i] for i in topk_idxs]

            # Step 3: Final Re-ranking
            # Disabled till we fix the core pipeline
            # ranked_chunks = rerank(q, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.top_k)

            # Step 4: Generation
            model_path = args.model_path or cfg.model_path
            ans = answer(q, ranked_chunks, model_path, max_tokens=cfg.max_gen_tokens)

            print("\n=================== START OF ANSWER ===================")
            print(ans.strip() if ans and ans.strip() else "(No output from model)")
            print("\n==================== END OF ANSWER ====================")
            logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": model_path})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logger.log_error(str(e))
            break

    # TODO: Fix completion logging.
    # logger.log_query_complete()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Robust config loading
    if args.config:
        cfg = QueryPlanConfig.from_yaml(args.config)
    else:
        cfg = find_fallback_config()

    if cfg is None:
        raise FileNotFoundError(
            "No config file provided and no fallback found at config/ or ~/.config/tokensmith/"
        )

    init_logger(cfg)

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()
