# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional, List, Tuple, Union, Any

from rich.live import Live
from rich.console import Console
from rich.markdown import Markdown

from src.config import RAGConfig
from src.generator import answer, double_answer, dedupe_generated_text
from src.index_builder import build_index
from src.instrumentation.logging import get_logger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.query_enhancement import generate_hypothetical_document, contextualize_query
from src.retriever import (
    filter_retrieved_chunks, 
    BM25Retriever, 
    FAISSRetriever, 
    IndexKeywordRetriever, 
    get_page_numbers, 
    load_artifacts
)
from src.ranking.reranker import rerank

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Welcome to TokenSmith!")
    parser.add_argument("mode", choices=["index", "chat"], help="operation mode")
    parser.add_argument("--pdf_dir", default="data/chapters/", help="directory containing PDF files")
    parser.add_argument("--index_prefix", default="textbook_index", help="prefix for generated index files")
    parser.add_argument("--model_path", help="path to generation model")
    parser.add_argument("--system_prompt_mode", choices=["baseline", "tutor", "concise", "detailed"], default="baseline")
    
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument("--keep_tables", action="store_true")
    indexing_group.add_argument("--multiproc_indexing", action="store_true")
    indexing_group.add_argument("--embed_with_headings", action="store_true")
    parser.add_argument(
        "--double_prompt",
        action="store_true",
        help="enable double prompting for higher quality answers"
    )

    return parser.parse_args()

def run_index_mode(args: argparse.Namespace, cfg: RAGConfig):
    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory()

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    build_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        embedding_model_context_window=cfg.embedding_model_context_window,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
    )
def use_indexed_chunks(question: str, chunks: list) -> list:
    # Logic for keyword matching from textbook index
    try:
        with open('index/sections/textbook_index_page_to_chunk_map.json', 'r') as f:
            page_to_chunk_map = json.load(f)
        with open('data/extracted_index.json', 'r') as f:
            extracted_index = json.load(f)
    except FileNotFoundError:
        return []

    keywords = get_keywords(question)
    chunk_ids = {
        chunk_id
        for word in keywords
        if word in extracted_index
        for page_no in extracted_index[word]
        for chunk_id in page_to_chunk_map.get(str(page_no), [])
    }
    return [chunks[cid] for cid in chunk_ids], list(chunk_ids)

def get_answer(
    question: str,
    cfg: RAGConfig,
    args: argparse.Namespace,
    logger: Any,
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False,
    additional_log_info: Optional[Dict[str, Any]] = None
) -> Union[str, Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """Run a single query through the pipeline."""

    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]

    ranked_chunks: List[str] = []
    topk_idxs: List[int] = []
    scores: List[float] = []
    chunk_diagnostics: Dict[int, Dict[str, Any]] = {}
    hyde_query = None
    chunks_info = None

    # Step 1: Get chunks
    if golden_chunks and cfg.use_golden_chunks:
        ranked_chunks = golden_chunks

    elif cfg.disable_chunks:
        ranked_chunks = []

    elif cfg.use_indexed_chunks:
        ranked_chunks, topk_idxs = use_indexed_chunks(question, chunks)

    else:
        retrieval_query = question
        if cfg.use_hyde:
            retrieval_query = generate_hypothetical_document(
                question, cfg.gen_model, max_tokens=cfg.hyde_max_tokens
            )
            hyde_query = retrieval_query

        pool_n = max(cfg.num_candidates, cfg.top_k + 10)

        # Step 1a: Per-retriever raw scores
        raw_scores: Dict[str, Dict[int, float]] = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(
                retrieval_query, pool_n, chunks
            )

        # Step 2: Fusion ranking
        ordered, scores = ranker.rank(raw_scores=raw_scores)

        # Post-fusion rank lookup: chunk_idx -> 1-based rank in fused ordering
        post_fusion_ranks = {idx: rank + 1 for rank, idx in enumerate(ordered)}

        # Step 3: Filter to top-k
        topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered)

        # Step 4: Build per-retriever rank lookups for diagnostics
        faiss_scores = raw_scores.get("faiss", {})
        bm25_scores  = raw_scores.get("bm25", {})

        faiss_ranked = sorted(faiss_scores, key=lambda i: faiss_scores[i], reverse=True)
        bm25_ranked  = sorted(bm25_scores,  key=lambda i: bm25_scores[i],  reverse=True)

        faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
        bm25_ranks  = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}

        # Build diagnostics dict for each top-k chunk (pre-reranking)
        for idx in topk_idxs:
            chunk_diagnostics[idx] = {
                "faiss_score":         faiss_scores.get(idx, None),
                "faiss_rank":          faiss_ranks.get(idx, None),
                "bm25_score":          bm25_scores.get(idx, None),
                "bm25_rank":           bm25_ranks.get(idx, None),
                "post_fusion_rank":    post_fusion_ranks.get(idx, None),
                # Filled in after reranking below
                "post_reranking_rank": None,
                "cross_encoder_score": None,
            }

        # Step 5: Reranking — pass (idx, text) pairs so indices survive
        indexed_chunks = [(idx, chunks[idx]) for idx in topk_idxs]
        reranked = rerank(
            question,
            indexed_chunks,
            mode=cfg.rerank_mode,
            top_n=cfg.rerank_top_k,
        )
        # reranked is List[Tuple[int, str, float]]: (idx, text, cross_encoder_score)

        # Rebuild topk_idxs and ranked_chunks in reranked order
        topk_idxs    = [idx for idx, _, _ in reranked]
        ranked_chunks = [(text, ce_score) for _, text, ce_score in reranked]

        # Fill in post-reranking rank and cross-encoder score
        for rerank_rank, (idx, _, ce_score) in enumerate(reranked, start=1):
            if idx in chunk_diagnostics:
                chunk_diagnostics[idx]["post_reranking_rank"] = rerank_rank
                chunk_diagnostics[idx]["cross_encoder_score"] = ce_score

        # For test mode
        if is_test_mode:
            chunks_info = []
            for rerank_rank, (idx, text, ce_score) in enumerate(reranked, start=1):
                diag = chunk_diagnostics.get(idx, {})
                chunks_info.append({
                    "rank":                rerank_rank,
                    "chunk_id":            idx,
                    "content":             text,
                    "faiss_score":         diag.get("faiss_score"),
                    "faiss_rank":          diag.get("faiss_rank"),
                    "bm25_score":          diag.get("bm25_score"),
                    "bm25_rank":           diag.get("bm25_rank"),
                    "post_fusion_rank":    diag.get("post_fusion_rank"),
                    "post_reranking_rank": rerank_rank,
                    "cross_encoder_score": ce_score,
                })

    if not ranked_chunks and not cfg.disable_chunks:
        if console:
            console.print(f"\n{ANSWER_NOT_FOUND}\n")
        return ANSWER_NOT_FOUND

    # Step 6: Generation
    model_path = cfg.gen_model
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode
    use_double = getattr(args, "double_prompt", False) or cfg.use_double_prompt

    if use_double:
        stream_iter = double_answer(
            question, ranked_chunks, model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )
    else:
        stream_iter = answer(
            question, ranked_chunks, model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )

    if is_test_mode:
        ans = ""
        for delta in stream_iter:
            ans += delta
        ans = dedupe_generated_text(ans)
        return ans, chunks_info, hyde_query

    # Step 7: Render + log
    ans = render_streaming_ans(console, stream_iter)

    meta = artifacts.get("meta", [])
    page_nums = get_page_numbers(topk_idxs, meta)

    # Build aligned lists for the logger (post-reranking order)
    top_chunks_text = [chunks[i] for i in topk_idxs]
    top_sources     = [sources[i] for i in topk_idxs]
    top_scores      = [
        scores[ordered.index(i)] if i in ordered else 0.0
        for i in topk_idxs
    ]

    logger.save_chat_log(
        query=question,
        config_state=cfg.get_config_state(),
        ordered_scores=top_scores,
        chat_request_params={
            "system_prompt": system_prompt,
            "max_tokens": cfg.max_gen_tokens,
        },
        top_idxs=topk_idxs,
        chunks=top_chunks_text,
        sources=top_sources,
        page_map=page_nums,
        full_response=ans,
        top_k=len(topk_idxs),
        chunk_diagnostics=chunk_diagnostics,
        additional_log_info=additional_log_info,
    )

    return ans


def render_streaming_ans(console, stream_iter):
    ans = ""
    is_first = True
    with Live(console=console, refresh_per_second=8) as live:
        for delta in stream_iter:
            if is_first:
                console.print("\n[bold cyan]=== START OF ANSWER ===[/bold cyan]\n")
                is_first = False
            ans += delta
            live.update(Markdown(ans))
    ans = dedupe_generated_text(ans)
    live.update(Markdown(ans))
    console.print("\n[bold cyan]=== END OF ANSWER ===[/bold cyan]\n")
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
    logger = get_logger()
    console = Console()

    print("Initializing TokenSmith Chat...")
    try:
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, args.index_prefix)
        print(f"Loaded {len(chunks)} chunks and {len(sources)} sources from artifacts.")
        retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path))
        
        ranker = EnsembleRanker(ensemble_method=cfg.ensemble_method, weights=cfg.ranker_weights, rrf_k=int(cfg.rrf_k))
        print("Loaded retrievers and initialized ranker.")
        artifacts = {"chunks": chunks, "sources": sources, "retrievers": retrievers, "ranker": ranker, "meta": meta}
    except Exception as e:
        print(f"ERROR: {e}. Run 'index' mode first.")
        sys.exit(1)

    chat_history = []
    additional_log_info = {}
    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        print("CHAT HISTORY:", chat_history)  # Debug print to trace chat history
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            effective_q = q
            if cfg.enable_history and chat_history:
                try:
                    effective_q = contextualize_query(q, chat_history, cfg.gen_model)
                    additional_log_info["is_contextualizing_query"] = True
                    additional_log_info["contextualized_query"] = effective_q
                    additional_log_info["original_query"] = q
                    additional_log_info["chat_history"] = chat_history
                    print(f"Contextualized Query: {effective_q}")  # Debug print to trace contextualization
                except Exception as e:
                    print(f"Warning: Failed to contextualize query: {e}. Using original query.")
                    effective_q = q
            
            # Use the single query function. get_answer also renders the streaming markdown and takes care of logging, so we need not do anything else here.
            ans = get_answer(effective_q, cfg, args, logger, console, artifacts=artifacts, additional_log_info=additional_log_info)

            # Update Chat history (make it atomic for user + assistant turn)
            try:
                user_turn      = {"role": "user", "content": q}
                assistant_turn = {"role": "assistant", "content": ans}
                chat_history  += [user_turn, assistant_turn]
            except Exception as e:
                print(f"Warning: Failed to update chat history: {e}")
                # We can continue without chat history, so we do not break the loop here.

            # Trim chat history to avoid exceeding context window
            if len(chat_history) > cfg.max_history_turns * 2:
                chat_history = chat_history[-cfg.max_history_turns * 2:]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break



def main():
    args = parse_args()
    config_path = pathlib.Path("config/config.yaml")
    if not config_path.exists(): raise FileNotFoundError("config/config.yaml not found.")
    cfg = RAGConfig.from_yaml(config_path)
    print(f"Loaded configuration from {config_path.resolve()}.")
    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)

if __name__ == "__main__":
    main()
