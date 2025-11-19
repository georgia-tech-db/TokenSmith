import argparse
import json
import hashlib
import pathlib
import sys
import time
from typing import Dict, Optional, Any, List

import numpy as np

from src.config import QueryPlanConfig
from src.generator import answer
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger, RunLogger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts
from src.query_enhancement import generate_hypothetical_document
from src.embedder import SentenceTransformer

_SEMANTIC_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_SEMANTIC_CACHE_THRESHOLD = 0.85
_SEMANTIC_CACHE_MAX_ENTRIES = 50
_QUESTION_EMBEDDERS: Dict[str, SentenceTransformer] = {}


def _normalize_question(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def _make_cache_config_key(
    cfg: QueryPlanConfig,
    args: argparse.Namespace,
    golden_chunks: Optional[list],
) -> str:
    payload = {
        "model_path": args.model_path or cfg.model_path,
        "embed_model": cfg.embed_model,
        "top_k": cfg.top_k,
        "pool_size": cfg.pool_size,
        "system_prompt_mode": args.system_prompt_mode or cfg.system_prompt_mode,
        "ensemble_method": cfg.ensemble_method,
        "ranker_weights": cfg.ranker_weights,
        "use_hyde": cfg.use_hyde,
        "use_indexed_chunks": cfg.use_indexed_chunks,
        "disable_chunks": cfg.disable_chunks,
        "use_golden_chunks": bool(golden_chunks and cfg.use_golden_chunks),
        "index_prefix": getattr(args, "index_prefix", None),
    }
    if golden_chunks and cfg.use_golden_chunks:
        sig = hashlib.sha256("||".join(golden_chunks).encode("utf-8")).hexdigest()
        payload["golden_signature"] = sig
    return json.dumps(payload, sort_keys=True)


def _semantic_cache_lookup(config_key: str, query_embedding: np.ndarray):
    entries = _SEMANTIC_CACHE.get(config_key) or []
    if not entries or query_embedding is None:
        return None
    best_entry = None
    best_score = -1.0
    for entry in entries:
        cached_vec = entry.get("embedding")
        if cached_vec is None:
            continue
        sim = float(np.dot(cached_vec, query_embedding))
        if sim > best_score:
            best_score = sim
            best_entry = entry
    if best_entry and best_score >= _SEMANTIC_CACHE_THRESHOLD:
        return best_entry["payload"]
    return None


def _semantic_cache_store(
    config_key: str,
    normalized_question: str,
    question_embedding: Optional[np.ndarray],
    payload: Dict[str, Any],
) -> None:
    if question_embedding is None:
        return
    entries = _SEMANTIC_CACHE.setdefault(config_key, [])
    entries.append(
        {
            "question": normalized_question,
            "embedding": question_embedding.astype(np.float32),
            "payload": payload,
        }
    )
    if len(entries) > _SEMANTIC_CACHE_MAX_ENTRIES:
        entries.pop(0)


def _get_question_embedder(
    retrievers: List[Any], cfg: QueryPlanConfig
) -> Optional[SentenceTransformer]:
    for retriever in retrievers or []:
        if isinstance(retriever, FAISSRetriever):
            return retriever.embedder
    model_path = cfg.embed_model
    if not model_path:
        return None
    embedder = _QUESTION_EMBEDDERS.get(model_path)
    if embedder is None:
        embedder = SentenceTransformer(model_path)
        _QUESTION_EMBEDDERS[model_path] = embedder
    return embedder


def _compute_question_embedding(
    question: str,
    retrievers: List[Any],
    cfg: QueryPlanConfig,
) -> Optional[np.ndarray]:
    embedder = _get_question_embedder(retrievers, cfg)
    if not embedder:
        return None
    vec = embedder.encode(
        [question],
        batch_size=1,
        normalize=True,
        show_progress_bar=False,
    )
    if vec.size == 0:
        return None
    return vec[0]

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
    
    artifacts_dir = cfg.make_artifacts_directory()

    build_index(
        markdown_file="data/book_with_pages.md",
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
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
    cfg: QueryPlanConfig,
    args: argparse.Namespace,
    logger: "RunLogger",
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
    normalized_question = _normalize_question(question)
    config_cache_key = _make_cache_config_key(cfg, args, golden_chunks)
    stage_timings: Dict[str, float] = {}
    question_embedding: Optional[np.ndarray] = None

    semantic_hit = None
    if _SEMANTIC_CACHE.get(config_cache_key):
        question_embedding = _compute_question_embedding(
            normalized_question, retrievers, cfg
        )
        semantic_hit = _semantic_cache_lookup(config_cache_key, question_embedding)

    if semantic_hit:
        chunk_indices = semantic_hit.get("chunk_indices", [])
        if chunk_indices and not cfg.disable_chunks and not cfg.use_indexed_chunks:
            logger.log_chunks_used(chunk_indices, chunks, sources)
        stage_timings["semantic_cache_hit_seconds"] = 0.0
        logger.log_stage_timings(stage_timings)
        ans = semantic_hit.get("answer", "")
        if is_test_mode:
            return ans, semantic_hit.get("chunks_info"), semantic_hit.get("hyde_query")
        return ans
    
    # Step 1: Get chunks (golden, retrieved, or none)
    chunks_info = None
    hyde_query = None
    chunk_indices: list[int] = []
    if golden_chunks and cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks
    elif cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
    elif cfg.use_indexed_chunks:
        # Use chunks from the textbook index
        lookup_start = time.perf_counter()
        ranked_chunks = use_indexed_chunks(question, chunks, logger)
        stage_timings["indexed_chunk_lookup_seconds"] = time.perf_counter() - lookup_start
    else:
        # Step 0: Query Enhancement (HyDE)
        retrieval_query = question
        if cfg.use_hyde:
            model_path = args.model_path or cfg.model_path
            hyde_start = time.perf_counter()
            hypothetical_doc = generate_hypothetical_document(
                question, model_path, max_tokens=cfg.hyde_max_tokens
            )
            stage_timings["hyde_seconds"] = time.perf_counter() - hyde_start
            retrieval_query = hypothetical_doc
            hyde_query = hypothetical_doc
            # print(f"🔍 HyDE query: {hypothetical_doc}")
        
        # Step 1: Retrieval
        pool_n = max(cfg.pool_size, cfg.top_k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        retrieval_start = time.perf_counter()
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)
        stage_timings["retrieval_seconds"] = time.perf_counter() - retrieval_start
        # TODO: Fix retrieval logging.
        
        # Step 2: Ranking
        ranking_start = time.perf_counter()
        ordered = ranker.rank(raw_scores=raw_scores)
        topk_idxs = apply_seg_filter(cfg, chunks, ordered)
        stage_timings["ranking_seconds"] = time.perf_counter() - ranking_start
        logger.log_chunks_used(topk_idxs, chunks, sources)
        chunk_indices = topk_idxs
        
        ranked_chunks = [chunks[i] for i in topk_idxs]
        
        # Capture chunk info if in test mode
        if is_test_mode:
            # Compute individual ranker ranks
            faiss_scores = raw_scores.get("faiss", {})
            bm25_scores = raw_scores.get("bm25", {})
            
            faiss_ranked = sorted(faiss_scores.keys(), key=lambda i: faiss_scores[i], reverse=True)  # Higher score = better
            bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)  # Higher score = better
            
            faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
            bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
            
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
                })
        
        # Step 3: Final Re-ranking (if enabled)
        # Disabled till we fix the core pipeline
        # ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.top_k)
    
    # Step 4: Generation
    model_path = args.model_path or cfg.model_path
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode
    generation_start = time.perf_counter()
    ans = answer(
        question, 
        ranked_chunks, 
        model_path, 
        max_tokens=cfg.max_gen_tokens, 
        system_prompt_mode=system_prompt
    )
    stage_timings["generation_seconds"] = time.perf_counter() - generation_start
    logger.log_stage_timings(stage_timings)
    
    cache_payload = {
        "answer": ans,
        "chunks_info": chunks_info,
        "hyde_query": hyde_query,
        "chunk_indices": chunk_indices,
    }
    if question_embedding is None:
        question_embedding = _compute_question_embedding(
            normalized_question, retrievers, cfg
        )
    _semantic_cache_store(
        config_cache_key,
        normalized_question,
        question_embedding,
        cache_payload,
    )

    if is_test_mode:
        return ans, chunks_info, hyde_query
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
        faiss_index, bm25_index, chunks, sources = load_artifacts(
            artifacts_dir=artifacts_dir, 
            index_prefix=args.index_prefix
        )

        retrievers = [
            FAISSRetriever(faiss_index, cfg.embed_model),
            BM25Retriever(bm25_index)
        ]
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
            "ranker": ranker
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

            # Use the single query function
            ans = get_answer(q, cfg, args, logger=logger,artifacts=artifacts)

            print("\n=================== START OF ANSWER ===================")
            print(ans.strip() if ans and ans.strip() else "(No output from model)")
            print("\n==================== END OF ANSWER ====================")
            logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": args.model_path or cfg.model_path})

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
        cfg = QueryPlanConfig.from_yaml(config_path)

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
