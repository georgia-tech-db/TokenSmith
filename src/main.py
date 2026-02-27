# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional, List, Any

from rich.live import Live
from rich.console import Console
from rich.markdown import Markdown

from src.config import RAGConfig
from src.generator import answer, dedupe_generated_text
from src.index_builder import build_index
from src.instrumentation.logging import get_logger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import (
    filter_retrieved_chunks, 
    BM25Retriever, 
    FAISSRetriever, 
    IndexKeywordRetriever, 
    get_page_numbers, 
    load_artifacts
)
from src.query_enhancement import generate_hypothetical_document
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
    artifacts: Dict
) -> str:
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]
    meta = artifacts["meta"]

    topk_idxs = []
    ordered_scores = []
    
    # 1. Retrieval & Ranking Logic
    if cfg.disable_chunks:
        ranked_chunks = []
    elif cfg.use_indexed_chunks:
        ranked_chunks, topk_idxs = use_indexed_chunks(question, chunks)
    else:
        retrieval_query = question
        if cfg.use_hyde:
            retrieval_query = generate_hypothetical_document(question, cfg.gen_model, max_tokens=cfg.hyde_max_tokens)
        
        pool_n = max(cfg.num_candidates, cfg.top_k + 10)
        raw_scores = {ret.name: ret.get_scores(retrieval_query, pool_n, chunks) for ret in retrievers}
        
        # Rank and filter
        topk_idxs, ordered_scores = ranker.rank(raw_scores=raw_scores)
        topk_idxs = filter_retrieved_chunks(cfg, chunks, topk_idxs)
        ranked_chunks = [chunks[i] for i in topk_idxs]
        
        # Final Rerank
        ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k)

    if not ranked_chunks and not cfg.disable_chunks:
        console.print(f"\n{ANSWER_NOT_FOUND}\n")
        return ANSWER_NOT_FOUND

    # 2. Generation
    stream_iter = answer(
        question, ranked_chunks, cfg.gen_model,
        max_tokens=cfg.max_gen_tokens,
        system_prompt_mode=args.system_prompt_mode or cfg.system_prompt_mode,
    )

    full_ans = render_streaming_ans(console, stream_iter)

    # 3. Unified Logging (Matches API Server)
    try:
        page_nums = get_page_numbers(topk_idxs, meta)
        log_chunks = [chunks[i] for i in topk_idxs]
        log_sources = [sources[i] for i in topk_idxs]

        logger.save_chat_log(
            query=question,
            chat_request_params={
                "mode": "cli_chat",
                "system_prompt": args.system_prompt_mode
            },
            ordered_scores=ordered_scores,
            config_state=cfg.get_config_state(),
            top_idxs=topk_idxs,
            chunks=log_chunks,
            sources=log_sources,
            page_map=page_nums,
            full_response=full_ans,
            top_k=cfg.top_k
        )
    except Exception as e:
        print(f"Logging failed: {e}")

    return full_ans

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

    while True:
        try:
            q = input("\nAsk > ").strip()
            if q.lower() in {"exit", "quit"}: break
            if not q: continue
            get_answer(q, cfg, args, logger, console, artifacts)
        except KeyboardInterrupt: break
        except Exception as e:
            print(f"Error: {e}")

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