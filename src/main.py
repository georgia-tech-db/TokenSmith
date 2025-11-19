import argparse
import json
import pathlib
import sys
import yaml
from typing import Dict, Optional, List, Tuple, Union

from src.config import QueryPlanConfig
from src.generator import answer
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger, RunLogger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts
from src.query_enhancement import generate_hypothetical_document
from src.planning.heuristics import HeuristicQueryPlanner


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
    
    # Load raw config for indexing settings
    config_path = pathlib.Path("config/config.yaml")
    indexing_config = None
    if config_path.exists():
        raw_config = yaml.safe_load(open(config_path))
        indexing_config = raw_config.get("indexing", {})

    build_index(
        markdown_file="data/book_with_pages.md",
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
        indexing_config=indexing_config,
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
    is_test_mode: bool = False,
    conversation_history: Optional[List[Dict]] = None
) -> Union[str, Tuple[str, List[Dict]], Tuple[str, Dict, str]]:
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
    topk_idxs = []  # Initialize for metadata extraction
    if golden_chunks and cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks[:5]  # Limit to top 5
        # For golden chunks, we don't have indices, so metadata will be empty
    elif cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
    elif cfg.use_indexed_chunks:
        # Use chunks from the textbook index
        ranked_chunks = use_indexed_chunks(question, chunks, logger)
        # Limit to top 5 for citations
        ranked_chunks = ranked_chunks[:5]
        # For indexed chunks, we don't have indices easily, so metadata will be empty
    else:
        # Step 0: Query Enhancement (HyDE)
        retrieval_query = question
        if cfg.use_hyde:
            model_path = args.model_path or cfg.model_path
            hypothetical_doc = generate_hypothetical_document(
                question, model_path, max_tokens=cfg.hyde_max_tokens
            )
            retrieval_query = hypothetical_doc
            hyde_query = hypothetical_doc
            # print(f"HyDE query: {hypothetical_doc}")
        
        # Step 1: Retrieval
        pool_n = max(cfg.pool_size, cfg.top_k + 10)
        
        # Check if contextual retrieval is enabled (load from config file)
        config_path = pathlib.Path("config/config.yaml")
        contextual_config = None
        use_contextual = False
        if config_path.exists():
            raw_config = yaml.safe_load(open(config_path))
            contextual_config = raw_config.get("contextual_retrieval", {})
            use_contextual = contextual_config.get("enabled", False)
        
        if use_contextual and artifacts.get("metadata"):
            # Use contextual retrieval with expansion and cross-reference boosting
            from src.retriever_contextual import ContextualRetriever, CrossReferenceBooster
            
            contextual_retriever = ContextualRetriever(
                base_retrievers=retrievers,
                metadata=artifacts["metadata"],
                expansion_window=contextual_config.get('expansion_window', 2),
                decay_factor=contextual_config.get('decay_factor', 0.5)
            )
            
            # Get contextual scores
            contextual_scores = contextual_retriever.get_scores_with_context(
                retrieval_query, pool_n, chunks
            )
            
            # Apply cross-reference boosting
            artifacts_dir = cfg.make_artifacts_directory()
            page_to_chunk_path = artifacts_dir / f"{args.index_prefix}_page_to_chunk_map.json"
            booster = CrossReferenceBooster(
                index_path="data/extracted_index.json",
                page_to_chunk_path=str(page_to_chunk_path),
                boost_factor=contextual_config.get('cross_ref_boost', 1.3)
            )
            boosted_scores = booster.boost_scores(retrieval_query, contextual_scores)
            
            # Rank and select top-k
            ordered = sorted(boosted_scores.keys(), 
                            key=lambda i: boosted_scores[i], 
                            reverse=True)
            topk_idxs = apply_seg_filter(cfg, chunks, ordered)
            logger.log_chunks_used(topk_idxs, chunks, sources)
            
            # Get top 5 chunks with their metadata
            top_5_idxs = topk_idxs[:5]
            ranked_chunks = [chunks[i] for i in top_5_idxs]
            topk_idxs = top_5_idxs
        else:
            # Standard retrieval
            raw_scores: Dict[str, Dict[int, float]] = {}
            for retriever in retrievers:
                raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)
            # TODO: Fix retrieval logging.
            
            # Step 2: Ranking
            ordered = ranker.rank(raw_scores=raw_scores)
            topk_idxs = apply_seg_filter(cfg, chunks, ordered)
            logger.log_chunks_used(topk_idxs, chunks, sources)
            
            # Get top 5 chunks with their metadata
            top_5_idxs = topk_idxs[:5]
            ranked_chunks = [chunks[i] for i in top_5_idxs]
            # Store top_5_idxs for metadata extraction later
            topk_idxs = top_5_idxs
        
        # Capture chunk info if in test mode
        if is_test_mode:
            # Compute individual ranker ranks
            # raw_scores may not exist if contextual retrieval was used
            if 'raw_scores' in locals():
                faiss_scores = raw_scores.get("faiss", {})
                bm25_scores = raw_scores.get("bm25", {})
            else:
                faiss_scores = {}
                bm25_scores = {}
            
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
    
    # Extract metadata for retrieved chunks (top 5)
    chunk_metadata = []
    if artifacts.get("metadata") and len(artifacts["metadata"]) > 0:
        metadata = artifacts["metadata"]
        
        # If we have topk_idxs (from normal retrieval), use those
        if topk_idxs:
            for idx in topk_idxs[:5]:
                if idx < len(metadata) and metadata[idx]:
                    meta = metadata[idx]
                    # Only add if we have at least page_number or section info
                    if meta.get('page_number') or meta.get('section') or meta.get('chapter'):
                        chunk_metadata.append({
                            'page_number': meta.get('page_number'),
                            'chapter': meta.get('chapter', 0),
                            'section': meta.get('section', 'Unknown'),
                            'section_hierarchy': meta.get('section_hierarchy', {})
                        })
        # Otherwise, try to find metadata by matching chunk content (for indexed_chunks or golden_chunks)
        elif ranked_chunks:
            # Try to match chunks to metadata by content
            for chunk in ranked_chunks[:5]:
                # Find matching metadata by content preview or section
                for idx, meta in enumerate(metadata):
                    if meta and (meta.get('text_preview') and chunk[:50] in meta.get('text_preview', '')):
                        if meta.get('page_number') or meta.get('section') or meta.get('chapter'):
                            chunk_metadata.append({
                                'page_number': meta.get('page_number'),
                                'chapter': meta.get('chapter', 0),
                                'section': meta.get('section', 'Unknown'),
                                'section_hierarchy': meta.get('section_hierarchy', {})
                            })
                            break
                if len(chunk_metadata) >= 5:
                    break
    
    # Step 4: Generation
    model_path = args.model_path or cfg.model_path
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode
    ans = answer(
        question, 
        ranked_chunks, 
        model_path, 
        max_tokens=cfg.max_gen_tokens, 
        system_prompt_mode=system_prompt,
        chunk_metadata=chunk_metadata if chunk_metadata else None,
        conversation_history=conversation_history
    )
    
    if is_test_mode:
        return ans, chunks_info, hyde_query
    
    # Return answer with metadata for citation display
    return ans, chunk_metadata if chunk_metadata else []

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
    
    # Check if query planner is enabled
    config_path = pathlib.Path("config/config.yaml")
    use_query_planner = False
    if config_path.exists():
        raw_config = yaml.safe_load(open(config_path))
        use_query_planner = raw_config.get("use_query_planner", True)  # Default to True
    
    planner = None
    if use_query_planner:
        planner = HeuristicQueryPlanner(cfg)
        print("Query planner enabled - adaptive retrieval based on query type")

    # Load artifacts, initialize retrievers and rankers once before the loop.
    print("Welcome to Tokensmith! Initializing chat...")
    try:
        artifacts_dir = cfg.make_artifacts_directory()
        faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
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
            "ranker": ranker,
            "metadata": metadata
        }
        
        # Verify metadata is loaded
        if metadata and len(metadata) > 0:
            # Check if metadata has actual content (not just empty dicts)
            sample_meta = metadata[0] if metadata else {}
            if sample_meta and (sample_meta.get('page_number') or sample_meta.get('section')):
                print(f"[OK] Metadata loaded: {len(metadata)} chunks with citation information")
            else:
                print("[WARNING] Metadata file exists but appears empty. Rebuild index to enable citations.")
                print("   Run: make run-index")
        else:
            print("[WARNING] No metadata found. Citations will not be available.")
            print("   To enable citations, rebuild the index: make run-index")
    except Exception as e:
        print(f"ERROR: Failed to initialize chat artifacts: {e}")
        print("Please ensure you have run 'index' mode first.")
        sys.exit(1)

    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Initialize conversation history
    conversation_history = []
    # Load conversation config
    config_path = pathlib.Path("config/config.yaml")
    max_history = 5  # Default: 5 turns = 10 messages
    if config_path.exists():
        raw_config = yaml.safe_load(open(config_path))
        conv_config = raw_config.get("conversation", {})
        if conv_config.get("enabled", True):
            max_history = conv_config.get("max_history", 5)
        else:
            conversation_history = None  # Disable if not enabled
    
    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if q.lower() in {"clear", "/clear", "/reset"}:
                conversation_history = []
                print("Conversation history cleared.")
                continue

            # Use query planner if enabled
            query_cfg = cfg
            if planner:
                query_cfg = planner.plan(q)
                # Recreate ranker with new weights if they changed
                if query_cfg.ranker_weights != cfg.ranker_weights:
                    artifacts["ranker"] = EnsembleRanker(
                        ensemble_method=query_cfg.ensemble_method,
                        weights=query_cfg.ranker_weights,
                        rrf_k=int(query_cfg.rrf_k)
                    )

            # Use the single query function
            result = get_answer(q, query_cfg, args, logger=logger, artifacts=artifacts, conversation_history=conversation_history)
            
            # Handle return value (answer, metadata) or just answer for backward compatibility
            if isinstance(result, tuple):
                ans, chunk_metadata = result
            else:
                ans = result
                chunk_metadata = []

            print("\n=================== START OF ANSWER ===================")
            print(ans.strip() if ans and ans.strip() else "(No output from model)")
            print("\n==================== END OF ANSWER ====================")
            
            # Display citations if available
            if chunk_metadata:
                print("\nREFERENCES:")
                for i, meta in enumerate(chunk_metadata, 1):
                    citation_parts = []
                    if meta.get('page_number'):
                        citation_parts.append(f"Page {meta['page_number']}")
                    if meta.get('chapter', 0) > 0:
                        citation_parts.append(f"Chapter {meta['chapter']}")
                    section_hierarchy = meta.get('section_hierarchy', {})
                    if section_hierarchy.get('section', 0) > 0:
                        section_str = f"{section_hierarchy['section']}"
                        if section_hierarchy.get('subsection', 0) > 0:
                            section_str += f".{section_hierarchy['subsection']}"
                        citation_parts.append(f"Section {section_str}")
                    
                    if citation_parts:
                        print(f"  [{i}] {', '.join(citation_parts)}")
                    section = meta.get('section', '')
                    if section and len(section) > 0 and section != 'Unknown':
                        # Show section name (truncated if too long)
                        section_display = section[:60] + "..." if len(section) > 60 else section
                        print(f"      {section_display}")
            else:
                # Debug: show why citations aren't available
                if not artifacts.get("metadata"):
                    print("\n[WARNING] No citations: Metadata not loaded. Rebuild index with: make run-index")
                elif not topk_idxs:
                    print("\n[WARNING] No citations: No chunks retrieved.")
                else:
                    print(f"\n[WARNING] No citations: Metadata exists but couldn't extract citation info for top chunks.")
                    print(f"   Debug: topk_idxs={topk_idxs[:3] if topk_idxs else None}, metadata_len={len(artifacts.get('metadata', []))}")
            
            logger.log_generation(ans, {"max_tokens": query_cfg.max_gen_tokens, "model_path": args.model_path or cfg.model_path})
            
            # Update conversation history (limit to 5 turns = 10 messages)
            if conversation_history is not None:
                conversation_history.append({"role": "user", "content": q})
                conversation_history.append({"role": "assistant", "content": ans})
                # Trim to last 5 turns (10 messages)
                if len(conversation_history) > max_history * 2:
                    conversation_history = conversation_history[-(max_history * 2):]

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
