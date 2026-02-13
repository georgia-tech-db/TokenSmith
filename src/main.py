# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional

from rich.live import Live

from src.config import RAGConfig
from src.generator import answer, dedupe_generated_text
from src.index_builder import build_index
from src.instrumentation.logging import get_logger, RunLogger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import filter_retrieved_chunks, BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts
from src.query_enhancement import generate_hypothetical_document
from src.ranking.reranker import rerank
from rich.console import Console
from rich.markdown import Markdown

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."

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
    parser.add_argument(
        "--system_prompt_mode",
        choices=["baseline", "tutor", "concise", "detailed"],
        default="baseline",
        help="system prompt mode (choices: baseline, tutor, concise, detailed)"
    )
    
    # Indexing-specific arguments
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument(
        "--keep_tables",
        action="store_true",
        help="include tables in the index"
    )
    indexing_group.add_argument(
        "--multiproc_indexing",
        action="store_true",
        help="use multiprocessing for index building"
    )
    indexing_group.add_argument(
        "--embed_with_headings",
        action="store_true",
        help="embed sections with headings"
    )

    return parser.parse_args()


def run_index_mode(args: argparse.Namespace, cfg: RAGConfig):
    """Handles the logic for building the index."""

    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory()

    # TODO: Add logic to chooes which markdown files to index. For now, we are simply indexing the first one.
    data_dir = pathlib.Path("data")
    md_files = sorted(data_dir.glob("*.md"))

    if len(md_files) == 0:
        print("ERROR: No markdown files found in data/. Run extraction first.", file=sys.stderr)
        sys.exit(1)

    build_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
    )

def use_indexed_chunks(question: str, chunks: list, logger: "RunLogger") -> list:
    """
    Retrieve chunks from the indexed chunks based on simple keyword matching.
    """
    with open('index/sections/textbook_index_page_to_chunk_map.json', 'r') as f:
            page_to_chunk_map = json.load(f)
    with open('data/extracted_index.json', 'r') as f:
        extracted_index = json.load(f)

    keywords = get_keywords(question)
    chunk_ids = set()
    ranked_chunks = []

    print(f"Extracted keywords for indexed chunk retrieval: {keywords}")

    chunk_ids = {
        chunk_id
        for word in keywords
        if word in extracted_index
        for page_no in extracted_index[word]
        for chunk_id in page_to_chunk_map.get(str(page_no), [])
    }
            
    for cid in chunk_ids:
        ranked_chunks.append(chunks[cid])

    print(f"Chunks retrieved using indexed chunks: {len(ranked_chunks)}")
    return ranked_chunks

def get_answer(
    question: str,
    cfg: RAGConfig,
    args: argparse.Namespace,
    logger: "RunLogger",
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False
) -> str:
    """
    Run a single query through the pipeline.
    """    
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]
    
    logger.log_query_start(question)
    
    # Step 1: Get chunks (golden, retrieved, or none)
    chunks_info = None
    hyde_query = None
    if golden_chunks and cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks
    elif cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
    elif cfg.use_indexed_chunks:
        # Use chunks from the textbook index
        ranked_chunks = use_indexed_chunks(question, chunks, logger)
    else:
        # Step 0: Query Enhancement (HyDE)
        retrieval_query = question
        if cfg.use_hyde:
            model_path = cfg.gen_model
            hypothetical_doc = generate_hypothetical_document(
                question, model_path, max_tokens=cfg.hyde_max_tokens
            )
            retrieval_query = hypothetical_doc
            hyde_query = hypothetical_doc
            # print(f"🔍 HyDE query: {hypothetical_doc}")
        
        # Step 1: Retrieval
        pool_n = max(cfg.num_candidates, cfg.top_k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)
        # TODO: Fix retrieval logging.
        
        # Step 2: Ranking
        ordered = ranker.rank(raw_scores=raw_scores)
        topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered)
        logger.log_chunks_used(topk_idxs, chunks, sources)
        
        ranked_chunks = [chunks[i] for i in topk_idxs]
        
        # Capture chunk info if in test mode
        if is_test_mode:
            # Compute individual ranker ranks
            faiss_scores = raw_scores.get("faiss", {})
            bm25_scores = raw_scores.get("bm25", {})
            index_scores = raw_scores.get("index_keywords", {})
            
            faiss_ranked = sorted(faiss_scores.keys(), key=lambda i: faiss_scores[i], reverse=True)
            bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)
            index_ranked = sorted(index_scores.keys(), key=lambda i: index_scores[i], reverse=True)
            
            faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
            bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
            index_ranks = {idx: rank + 1 for rank, idx in enumerate(index_ranked)}
            
            chunks_info = []
            for rank, idx in enumerate(topk_idxs, 1):
                chunks_info.append({
                    "rank": rank,
                    "chunk_id": idx,
                    "content": chunks[idx],
                    "faiss_score": faiss_scores.get(idx, 0),
                    "faiss_rank": faiss_ranks.get(idx, 0),
                    "bm25_score": bm25_scores.get(idx, 0),
                    "bm25_rank": bm25_ranks.get(idx, 0),
                    "index_score": index_scores.get(idx, 0),
                    "index_rank": index_ranks.get(idx, 0),
                })

        # Step 3: Final re-ranking
        ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k)

    # If no chunks found, return answer not found message
    if ranked_chunks == []:
        console.print("\n"+ANSWER_NOT_FOUND+"\n")
        return ANSWER_NOT_FOUND

    # Step 4: Generation
    model_path = cfg.gen_model
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode

    stream_iter = answer(
        question,
        ranked_chunks,
        model_path,
        max_tokens=cfg.max_gen_tokens,
        system_prompt_mode=system_prompt,
    )

    if is_test_mode:
        # We do not render MD in the test mode
        ans = ""
        for delta in stream_iter:
            ans += delta
        ans = dedupe_generated_text(ans)
        return ans, chunks_info, hyde_query
    else:
        # Accumulate the full text while rendering incremental Markdown chunks
        ans = render_streaming_ans(console, stream_iter)
    return ans


def render_streaming_ans(console, stream_iter):
    if not console:
        raise ValueError("Console must be non null for rendering.")
    ans = ""
    is_first = True
    with Live(console=console, refresh_per_second=5) as live:
        for delta in stream_iter:
            if is_first:
                # we need to do this to ensure this marker comes after warning noise in Macs.
                console.print("\n[bold cyan]==================== START OF ANSWER ===================[/bold cyan]\n")
                is_first = False
            ans += delta
            live.update(Markdown(ans))
    ans = dedupe_generated_text(ans)
    live.update(Markdown(ans))
    console.print("\n[bold cyan]===================== END OF ANSWER ====================[/bold cyan]\n")
    return ans

def get_keywords(question: str) -> list:
    """
    Simple keyword extraction from the question.
    """
    stopwords = set([
        "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in", 
        "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what"
    ])
    words = question.lower().split()
    keywords = [word.strip('.,!?()[]') for word in words if word not in stopwords]
    return keywords

def run_chat_session(args: argparse.Namespace, cfg: RAGConfig):
    """
    Initializes artifacts and runs the main interactive chat loop.
    """
    logger = get_logger()
    console = Console()

    # planner = HeuristicQueryPlanner(cfg)

    # Load artifacts, initialize retrievers and rankers once before the loop.
    print("Welcome to Tokensmith! Initializing chat...")
    try:
        # Disabled till we fix the core pipeline
        # cfg = planner.plan(q)
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_index, bm25_index, chunks, sources, meta = load_artifacts(
            artifacts_dir=artifacts_dir, 
            index_prefix=args.index_prefix
        )

        retrievers = [
            FAISSRetriever(faiss_index, cfg.embed_model),
            BM25Retriever(bm25_index)
        ]
        
        # Add index keyword retriever if weight > 0
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(
                IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
            )
        
        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k)
        )
        
        # Package artifacts for reuse
        artifacts = {
            "chunks": chunks,
            "sources": sources,
            "retrievers": retrievers,
            "ranker": ranker,
            "meta": meta,
        }
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

            # Use the single query function. get_answer also renders the streaming markdown.
            ans = get_answer(q, cfg, args, logger, console, artifacts=artifacts)
            logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": cfg.gen_model})

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

    # Config loading
    config_path = pathlib.Path("config/config.yaml")
    cfg = None
    if config_path.exists():
        cfg = RAGConfig.from_yaml(config_path)

    if cfg is None:
        raise FileNotFoundError(
            "No config file provided at config/config.yaml."
        )

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()
