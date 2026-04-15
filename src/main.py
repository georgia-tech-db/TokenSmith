# noinspection PyUnresolvedReferences

import argparse
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
from src.query_enhancement import contextualize_query
from src.retrieval_pipeline import build_runtime_retrievers, execute_retrieval_plan, trace_to_dict
from src.retriever import (
    get_page_numbers,
    load_artifact_bundle,
)
from src.ranking.reranker import rerank

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for index or chat mode."""
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
    """Build chunk and section indexes from the source document."""
    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory()

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    source_file: Optional[pathlib.Path] = md_files[0] if md_files else None
    if source_file is None:
        sections_json = data_dir / "extracted_sections.json"
        if sections_json.exists():
            source_file = sections_json
            print(f"No markdown export found. Falling back to extracted sections JSON: {source_file}")
        else:
            print("ERROR: No markdown files or extracted_sections.json found in data/.", file=sys.stderr)
            sys.exit(1)

    build_index(
        markdown_file=str(source_file),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
    )

def _run_adaptive_retrieval(
    question: str,
    cfg: RAGConfig,
    bundle: Any,
    runtime_retrievers: Dict,
    meta: List,
    chunks: List[str],
    history: Optional[List[Dict[str, str]]],
    is_test_mode: bool,
) -> Tuple[List[str], List[int], Optional[List[Dict[str, Any]]], Any, List]:
    """Run retrieval through the adaptive pipeline with query routing and page-aware reranking.

    Returns (ranked_chunks, topk_idxs, chunks_info, retrieval_trace, scores).
    """
    ranked_chunks, topk_idxs, retrieval_trace = execute_retrieval_plan(
        query=question,
        cfg=cfg,
        bundle=bundle,
        retrievers=runtime_retrievers,
        history=history,
    )
    scores = retrieval_trace.fused_chunk_scores
    chunks_info = None

    if is_test_mode:
        score_lookup = {
            name: {
                idx: {"score": score, "rank": rank + 1}
                for rank, (idx, score) in enumerate(
                    sorted(values.items(), key=lambda item: item[1], reverse=True)
                )
            }
            for name, values in retrieval_trace.chunk_scores.items()
        }
        chunks_info = [
            {
                "rank": rank,
                "chunk_id": idx,
                "content": chunks[idx],
                "page_numbers": meta[idx].get("page_numbers", []),
                "section_path": meta[idx].get("section_path"),
                "faiss_score": score_lookup.get("faiss", {}).get(idx, {}).get("score", 0),
                "faiss_rank": score_lookup.get("faiss", {}).get(idx, {}).get("rank", 0),
                "bm25_score": score_lookup.get("bm25", {}).get(idx, {}).get("score", 0),
                "bm25_rank": score_lookup.get("bm25", {}).get(idx, {}).get("rank", 0),
                "index_score": score_lookup.get("index_keywords", {}).get(idx, {}).get("score", 0),
                "index_rank": score_lookup.get("index_keywords", {}).get(idx, {}).get("rank", 0),
            }
            for rank, idx in enumerate(topk_idxs, 1)
        ]

    return ranked_chunks, topk_idxs, chunks_info, retrieval_trace, scores


def _run_legacy_retrieval(
    question: str,
    cfg: RAGConfig,
    artifacts: Dict,
    chunks: List[str],
    is_test_mode: bool,
) -> Tuple[List[str], List[int], Optional[List[Dict[str, Any]]], List]:
    """Run retrieval through the legacy ensemble ranker path without adaptive routing.

    Returns (ranked_chunks, topk_idxs, chunks_info, scores).
    """
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]
    pool_n = max(cfg.num_candidates, cfg.top_k + 10)
    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(question, pool_n, chunks)
    ordered, scores = ranker.rank(raw_scores=raw_scores)
    topk_idxs = [int(chunk_id) for chunk_id in ordered[: cfg.top_k]]
    ranked_chunks = [chunks[idx] for idx in topk_idxs]
    chunks_info = None

    if is_test_mode:
        faiss_scores = raw_scores.get("faiss", {})
        bm25_scores = raw_scores.get("bm25", {})
        index_scores = raw_scores.get("index_keywords", {})
        faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(sorted(faiss_scores, key=faiss_scores.get, reverse=True))}
        bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(sorted(bm25_scores, key=bm25_scores.get, reverse=True))}
        index_ranks = {idx: rank + 1 for rank, idx in enumerate(sorted(index_scores, key=index_scores.get, reverse=True))}
        chunks_info = [
            {
                "rank": rank,
                "chunk_id": idx,
                "content": chunks[idx],
                "faiss_score": faiss_scores.get(idx, 0),
                "faiss_rank": faiss_ranks.get(idx, 0),
                "bm25_score": bm25_scores.get(idx, 0),
                "bm25_rank": bm25_ranks.get(idx, 0),
                "index_score": index_scores.get(idx, 0),
                "index_rank": index_ranks.get(idx, 0),
            }
            for rank, idx in enumerate(topk_idxs, 1)
        ]

    ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k)
    return ranked_chunks, topk_idxs, chunks_info, scores


def get_answer(
    question: str,
    cfg: RAGConfig,
    args: argparse.Namespace,
    logger: Any,
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False,
    additional_log_info: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Union[str, Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """Run a single query through the retrieval and generation pipeline.

    Selects between the adaptive retrieval pipeline (when an ArtifactBundle is
    available) and the legacy ensemble-ranker path, then generates an answer.

    Args:
        question: User query text.
        cfg: Active RAG configuration.
        args: CLI argument namespace (controls generation model and prompt mode).
        logger: RunLogger instance for persisting chat logs.
        console: Rich console for streaming output (None in test mode).
        artifacts: Dict containing loaded chunks, sources, metadata, and retrievers.
        golden_chunks: Optional pre-selected chunks to bypass retrieval.
        is_test_mode: When True, returns structured output instead of streaming.
        additional_log_info: Extra metadata to include in the chat log.
        history: Conversation history for follow-up query rewriting.

    Returns:
        In test mode: (answer_text, chunks_info, retrieval_trace_dict).
        In interactive mode: the generated answer string.
    """
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    meta = artifacts.get("meta", [])
    bundle = artifacts.get("bundle")
    runtime_retrievers = artifacts.get("runtime_retrievers")

    ranked_chunks: List[str] = []
    topk_idxs: List[int] = []
    scores: List = []
    chunks_info = None
    retrieval_trace = None

    if golden_chunks and cfg.use_golden_chunks:
        ranked_chunks = golden_chunks
    elif cfg.disable_chunks:
        ranked_chunks = []
    elif bundle and runtime_retrievers:
        ranked_chunks, topk_idxs, chunks_info, retrieval_trace, scores = _run_adaptive_retrieval(
            question, cfg, bundle, runtime_retrievers, meta, chunks, history, is_test_mode,
        )
    else:
        ranked_chunks, topk_idxs, chunks_info, scores = _run_legacy_retrieval(
            question, cfg, artifacts, chunks, is_test_mode,
        )

    if not ranked_chunks and not cfg.disable_chunks:
        if console:
            console.print(f"\n{ANSWER_NOT_FOUND}\n")
        return ANSWER_NOT_FOUND

    # Step 4: Generation
    model_path = cfg.gen_model
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode

    use_double = getattr(args, "double_prompt", False) or cfg.use_double_prompt

    if use_double:
        stream_iter = double_answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )
    else:
        stream_iter = answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )

    if is_test_mode:
        ans = ""
        for delta in stream_iter:
            ans += delta
        ans = dedupe_generated_text(ans)
        if retrieval_trace:
            return ans, chunks_info, trace_to_dict(retrieval_trace)
        return ans, chunks_info, None
    else:
        ans = render_streaming_ans(console, stream_iter)

        page_nums = get_page_numbers(topk_idxs, meta)
        log_chunks = [chunks[idx] for idx in topk_idxs]
        log_sources = [sources[idx] for idx in topk_idxs]
        merged_log_info = dict(additional_log_info or {})
        if retrieval_trace:
            merged_log_info["retrieval_trace"] = trace_to_dict(retrieval_trace)
        logger.save_chat_log(
            query=question,
            config_state=cfg.get_config_state(),
            ordered_scores=scores[:len(topk_idxs)] if scores else [],
            chat_request_params={
                "system_prompt": system_prompt,
                "max_tokens": cfg.max_gen_tokens
            },
            top_idxs=topk_idxs,
            chunks=log_chunks,
            sources=log_sources,
            page_map=page_nums,
            full_response=ans,
            top_k=len(topk_idxs),
            additional_log_info=merged_log_info
        )
        return ans

def render_streaming_ans(console, stream_iter):
    """Stream and render a markdown answer in the terminal via Rich Live."""
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

def run_chat_session(args: argparse.Namespace, cfg: RAGConfig):
    """Run an interactive chat loop, loading artifacts and streaming answers."""
    logger = get_logger()
    console = Console()

    print("Initializing TokenSmith Chat...")
    try:
        artifacts_dir = cfg.get_artifacts_directory()
        bundle = load_artifact_bundle(artifacts_dir, args.index_prefix)
        print(f"Loaded {len(bundle.chunks)} chunks and {len(bundle.sources)} sources from artifacts.")
        runtime_retrievers = build_runtime_retrievers(bundle, cfg)
        legacy_ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k),
        )
        print("Loaded retrievers and initialized ranker.")
        artifacts = {
            "bundle": bundle,
            "runtime_retrievers": runtime_retrievers,
            "chunks": bundle.chunks,
            "sources": bundle.sources,
            "meta": bundle.metadata,
            "retrievers": runtime_retrievers["chunk"],
            "ranker": legacy_ranker,
        }
    except Exception as e:
        print(f"ERROR: {e}. Run 'index' mode first.")
        sys.exit(1)

    chat_history = []
    additional_log_info = {}
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
            
            effective_q = q
            if cfg.enable_history and chat_history:
                try:
                    effective_q = contextualize_query(q, chat_history, cfg.gen_model)
                    additional_log_info["is_contextualizing_query"] = True
                    additional_log_info["contextualized_query"] = effective_q
                    additional_log_info["original_query"] = q
                    additional_log_info["chat_history"] = chat_history
                except Exception as e:
                    print(f"Warning: Failed to contextualize query: {e}. Using original query.")
                    effective_q = q
            
            # Use the single query function. get_answer also renders the streaming markdown and takes care of logging, so we need not do anything else here.
            ans = get_answer(
                effective_q,
                cfg,
                args,
                logger,
                console,
                artifacts=artifacts,
                additional_log_info=additional_log_info,
                history=chat_history,
            )

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
    if not config_path.exists():
        raise FileNotFoundError("config/config.yaml not found.")
    cfg = RAGConfig.from_yaml(config_path)
    print(f"Loaded configuration from {config_path.resolve()}.")
    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)

if __name__ == "__main__":
    main()
