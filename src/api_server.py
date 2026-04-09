"""
FastAPI server for TokenSmith chat functionality.
Provides REST API endpoints for the React frontend.
"""

import sys
import pathlib
import re, json
import traceback
from uuid import uuid4
from copy import deepcopy
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Tuple
import traceback
import os

_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import RAGConfig
from src.generator import answer
from src.feedback_store import (
    init_feedback_db,
    save_answer,
    save_feedback,
    get_answer_question,
    update_user_topic_state,
)
from src.instrumentation.logging import get_logger
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    filter_retrieved_chunks,
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    get_page_numbers,
    load_artifacts,
)
from src.ranking.reranker import rerank
from src.user_feedback_model import TopicExtractor, estimate_difficulty

INDEX_PREFIX = "textbook_index"

_artifacts: Optional[Dict[str, List[str]]] = None
_retrievers: Optional[List] = None
_ranker: Optional[EnsembleRanker] = None
_config: Optional[RAGConfig] = None
_logger = None
_topic_extractor: Optional[TopicExtractor] = None


class SourceItem(BaseModel):
    page: int
    text: str

    class Config:
        frozen = True


class ChatRequest(BaseModel):
    query: str
    enable_chunks: Optional[bool] = None
    prompt_type: Optional[str] = None
    max_chunks: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    answer_id: str
    vote: int
    reason: Optional[str] = None
    session_id: str


class FeedbackResponse(BaseModel):
    ok: bool
    message: str


class ChatResponse(BaseModel):
    answer_id: str
    session_id: str
    answer: str
    sources: List[SourceItem]
    chunks_used: List[int]
    chunks_by_page: Dict[int, List[str]]
    query: str


def _resolve_config_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def _ensure_initialized():
    if not all([_config, _artifacts, _retrievers, _ranker]):
        raise HTTPException(
            status_code=503,
            detail="Artifacts not loaded. Please run indexing first.",
        )


def _retrieve_and_rank(
    query: str,
    top_k: Optional[int] = None,
) -> Tuple[List[int], List[float], Dict[str, Dict[int, float]]]:
    """
    Run all retrievers and fuse scores.

    Returns:
        ordered_ids:    Chunk indices sorted by fused score, truncated to top_k.
        ordered_scores: Corresponding fused scores.
        raw_scores:     Per-retriever score dicts — used for diagnostics.
    """
    chunks = _artifacts["chunks"]
    effective_top_k = top_k if top_k is not None else _config.top_k
    pool_n = max(_config.num_candidates, effective_top_k + 10)

    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in _retrievers:
        raw_scores[retriever.name] = retriever.get_scores(query, pool_n, chunks)

    ordered_ids, ordered_scores = _ranker.rank(raw_scores=raw_scores)

    limit = effective_top_k
    ordered_ids    = ordered_ids[:limit]
    ordered_scores = ordered_scores[:limit]

    return ordered_ids, ordered_scores, raw_scores


def _build_chunk_diagnostics(
    topk_idxs: List[int],
    ordered_ids: List[int],
    ordered_scores: List[float],
    raw_scores: Dict[str, Dict[int, float]],
) -> Dict[int, Dict[str, Any]]:
    """
    Build per-chunk diagnostic dict (pre-reranking).
    post_reranking_rank and cross_encoder_score are filled in after reranking.
    """
    faiss_scores = raw_scores.get("faiss", {})
    bm25_scores  = raw_scores.get("bm25", {})

    faiss_ranked = sorted(faiss_scores, key=lambda i: faiss_scores[i], reverse=True)
    bm25_ranked  = sorted(bm25_scores,  key=lambda i: bm25_scores[i],  reverse=True)

    faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
    bm25_ranks  = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}

    # post_fusion_rank: rank within the full fused ordering (not just top-k)
    post_fusion_ranks = {idx: rank + 1 for rank, idx in enumerate(ordered_ids)}

    diagnostics: Dict[int, Dict[str, Any]] = {}
    for idx in topk_idxs:
        diagnostics[idx] = {
            "faiss_score":         faiss_scores.get(idx, None),
            "faiss_rank":          faiss_ranks.get(idx, None),
            "bm25_score":          bm25_scores.get(idx, None),
            "bm25_rank":           bm25_ranks.get(idx, None),
            "post_fusion_rank":    post_fusion_ranks.get(idx, None),
            "post_reranking_rank": None,   # filled after reranking
            "cross_encoder_score": None,   # filled after reranking
        }
    return diagnostics


def _apply_reranking(
    query: str,
    topk_idxs: List[int],
    chunk_diagnostics: Dict[int, Dict[str, Any]],
) -> Tuple[List[int], List[Any]]:
    """
    Run cross-encoder reranking and fill diagnostics in-place.

    Returns:
        reranked_idxs:   Chunk indices in reranked order.
        reranked_chunks: List of (text, ce_score) tuples for the generator.
    """
    chunks = _artifacts["chunks"]
    indexed_chunks = [(idx, chunks[idx]) for idx in topk_idxs]

    reranked = rerank(
        query,
        indexed_chunks,
        mode=_config.rerank_mode,
        top_n=_config.rerank_top_k,
    )
    # reranked: List[Tuple[int, str, float]]

    reranked_idxs   = [idx for idx, _, _ in reranked]
    reranked_chunks = [(text, ce_score) for _, text, ce_score in reranked]

    for rerank_pos, (idx, _, ce_score) in enumerate(reranked, start=1):
        if idx in chunk_diagnostics:
            chunk_diagnostics[idx]["post_reranking_rank"] = rerank_pos
            chunk_diagnostics[idx]["cross_encoder_score"] = ce_score

    return reranked_idxs, reranked_chunks


def _create_log(
    chunks: List[str],
    sources: List[str],
    topk_idxs: List[int],
    ordered_ranked_scores: List[float],
    page_nums: Dict,
    full_response_accumulator: List[str],
    request: ChatRequest,
    enable_chunks: bool,
    prompt_type: str,
    max_chunks: int,
    temperature: float,
    chunk_diagnostics: Optional[Dict[int, Dict[str, Any]]] = None,
) -> bool:
    try:
        # Align everything to the actual number of reranked chunks,
        # NOT max_chunks — reranking may have reduced the count to rerank_top_k.
        n = len(topk_idxs)
        log_chunks  = [chunks[i]  for i in topk_idxs]
        log_sources = [sources[i] for i in topk_idxs]

        # Trim fusion scores to match reranked count.
        # ordered_ranked_scores comes from fusion (pre-reranking) so may be longer.
        log_scores = ordered_ranked_scores[:n]

        _logger.save_chat_log(
            query=request.query,
            config_state=_config.get_config_state(),
            ordered_scores=log_scores,
            chat_request_params={
                "enable_chunks": {"provided": request.enable_chunks, "used": enable_chunks},
                "prompt_type":   {"provided": request.prompt_type,   "used": prompt_type},
                "max_chunks":    {"provided": request.max_chunks,     "used": max_chunks},
                "temperature":   {"provided": request.temperature,    "used": temperature},
            },
            top_idxs=topk_idxs,
            chunks=log_chunks,
            sources=log_sources,
            page_map=page_nums,
            full_response="".join(full_response_accumulator),
            top_k=n,
            chunk_diagnostics=chunk_diagnostics,
        )
        return True

    except Exception as log_exc:
        print(f"Logging error: {log_exc}")
        traceback.print_exc()
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifacts, _retrievers, _ranker, _config, _logger, _topic_extractor

    config_path = _resolve_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")

    _config  = RAGConfig.from_yaml(config_path)
    _logger  = get_logger()

    try:
        artifacts_dir = _config.get_artifacts_directory()
        faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
            artifacts_dir=artifacts_dir,
            index_prefix=INDEX_PREFIX,
        )

        _artifacts = {"chunks": chunks, "sources": sources, "meta": metadata}

        _retrievers = [
            FAISSRetriever(faiss_index, _config.embed_model),
            BM25Retriever(bm25_index),
        ]
        if _config.ranker_weights.get("index_keywords", 0) > 0:
            _retrievers.append(
                IndexKeywordRetriever(
                    _config.extracted_index_path,
                    _config.page_to_chunk_map_path,
                )
            )

        _ranker = EnsembleRanker(
            ensemble_method=_config.ensemble_method,
            weights=_config.ranker_weights,
            rrf_k=int(_config.rrf_k),
        )

        init_feedback_db()
        _topic_extractor = (
            TopicExtractor(
                extracted_index_path=_config.extracted_index_path,
                page_to_chunk_map_path=_config.page_to_chunk_map_path,
            )
            if _config.enable_topic_extraction
            else None
        )
        print("TokenSmith API initialized successfully")

    except Exception as exc:
        print(f"Warning: Could not load artifacts: {exc}")
        print("   Run indexing first or check your configuration")

    yield
    print("Shutting down TokenSmith API...")


app = FastAPI(
    title="TokenSmith API",
    description="REST API for TokenSmith RAG chat functionality",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:3001",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "TokenSmith API is running"}


@app.post("/api/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    if request.vote not in (1, -1):
        raise HTTPException(status_code=400, detail="vote must be 1 or -1")

    save_feedback(
        answer_id=request.answer_id,
        session_id=request.session_id,
        vote=request.vote,
        reason=request.reason,
    )

    question = get_answer_question(request.answer_id)
    if question and _topic_extractor:
        topics = _topic_extractor.extract_topics(question)
        base_difficulty = estimate_difficulty(question)
        difficulty = "hard" if request.vote == -1 else base_difficulty
        delta = -0.2 if request.vote == -1 else 0.1
        for topic in topics:
            update_user_topic_state(
                session_id=request.session_id,
                topic=topic,
                difficulty=difficulty,
                delta_confidence=delta,
                evidence={
                    "type": "feedback",
                    "answer_id": request.answer_id,
                    "vote": request.vote,
                    "reason": request.reason,
                },
            )
        return FeedbackResponse(ok=True, message="Feedback stored.")
    if not question:
        return FeedbackResponse(ok=True, message="Feedback stored; unknown answer_id.")
    return FeedbackResponse(ok=True, message="Feedback stored; topic extractor disabled.")


@app.post("/api/test-chat")
async def test_chat(request: ChatRequest):
    """Test endpoint — retrieval only, no generation."""
    print(f"Test chat request: {request.query}")
    try:
        _ensure_initialized()
    except HTTPException as exc:
        return {"error": exc.detail, "status": "error"}

    if not request.query.strip():
        return {"error": "Query cannot be empty", "status": "error"}

    enable_chunks = (
        request.enable_chunks
        if request.enable_chunks is not None
        else not _config.disable_chunks
    )
    max_chunks = (
        request.top_k
        if request.top_k is not None
        else (request.max_chunks if request.max_chunks is not None else _config.top_k)
    )

    if not enable_chunks:
        return {"error": "Chunk retrieval disabled.", "status": "error"}

    try:
        topk_idxs, ordered_ranked_scores, raw_scores = _retrieve_and_rank(
            request.query, top_k=max_chunks
        )
        topk_idxs = [int(i) for i in topk_idxs]
        ranked_chunks = [_artifacts["chunks"][i] for i in topk_idxs[:max_chunks]]

        return {
            "status": "success",
            "query": request.query,
            "chunks_found": len(ranked_chunks),
            "top_chunks": ranked_chunks[:3],
            "raw_scores": ordered_ranked_scores,
            "top_idxs": topk_idxs,
            "message": "Retrieval and ranking successful, generation skipped",
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "status": "error"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    _ensure_initialized()
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    enable_chunks = (
        request.enable_chunks
        if request.enable_chunks is not None
        else not _config.disable_chunks
    )
    disable_chunks = not enable_chunks
    prompt_type  = request.prompt_type  if request.prompt_type  is not None else _config.system_prompt_mode
    max_chunks   = request.top_k        if request.top_k        is not None else (request.max_chunks if request.max_chunks is not None else _config.top_k)
    temperature  = request.temperature  if request.temperature  is not None else 0.7

    chunks  = _artifacts["chunks"]
    sources = _artifacts["sources"]

    # --- Retrieval, ranking, reranking ---
    chunk_diagnostics: Dict[int, Dict[str, Any]] = {}
    ordered_ranked_scores: List[float] = []

    if disable_chunks:
        ranked_chunks, topk_idxs = [], []
    else:
        topk_idxs, ordered_ranked_scores, raw_scores = _retrieve_and_rank(
            request.query, top_k=max_chunks
        )
        topk_idxs = [int(i) for i in topk_idxs]

        chunk_diagnostics = _build_chunk_diagnostics(
            topk_idxs, topk_idxs, ordered_ranked_scores, raw_scores
        )

        topk_idxs, ranked_chunks = _apply_reranking(
            request.query, topk_idxs, chunk_diagnostics
        )

    answer_id  = str(uuid4())
    session_id = request.session_id or str(uuid4())

    async def event_generator():
        full_response_accumulator = []
        try:
            page_nums = get_page_numbers(topk_idxs, _artifacts["meta"])
            sources_used: set  = set()
            chunks_by_page: Dict[int, List[str]] = {}

            for i in topk_idxs[:max_chunks]:
                pages = page_nums.get(i, [1]) or [1]
                for page in pages:
                    chunks_by_page.setdefault(page, []).append(chunks[i])
                    sources_used.add(SourceItem(page=page, text=sources[i]))

            yield f"data: {json.dumps({'type': 'sources', 'content': [s.dict() for s in sources_used]})}\n\n"
            yield f"data: {json.dumps({'type': 'chunks_by_page', 'content': chunks_by_page})}\n\n"

            for delta in answer(
                request.query,
                ranked_chunks,
                _config.gen_model,
                _config.max_gen_tokens,
                system_prompt_mode=prompt_type,
                temperature=temperature,
            ):
                if delta:
                    full_response_accumulator.append(delta)
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

            if _logger:
                success_log = _create_log(
                    chunks,
                    sources,
                    topk_idxs,
                    ordered_ranked_scores,
                    page_nums,
                    full_response_accumulator,
                    request,
                    enable_chunks,
                    prompt_type,
                    max_chunks,
                    temperature,
                    chunk_diagnostics=chunk_diagnostics,
                )
                if not success_log:
                    print("Logging failed for this request.")

            retrieval_info = {
                "chunks_used": [int(i) for i in topk_idxs[:max_chunks]],
                "page_numbers": page_nums,
                "index_prefix": INDEX_PREFIX,
            }
            save_answer(
                answer_id=answer_id,
                session_id=session_id,
                question=request.query,
                answer="".join(full_response_accumulator),
                retrieval_info=retrieval_info,
                model=_config.gen_model,
                prompt_mode=prompt_type,
            )

            if _topic_extractor:
                topics = _topic_extractor.extract_topics(request.query)
                difficulty = estimate_difficulty(request.query)
                for topic in topics:
                    update_user_topic_state(
                        session_id=session_id,
                        topic=topic,
                        difficulty=difficulty,
                        delta_confidence=0.0,
                        evidence={
                            "type": "question",
                            "question": request.query,
                            "answer_id": answer_id,
                        },
                    )

            yield f"data: {json.dumps({'type': 'done', 'answer_id': answer_id, 'session_id': session_id, 'sources': [s.dict() for s in sources_used]})}\n\n"

        except Exception as e:
            print(f"Backend error: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint (non-streaming)."""
    _ensure_initialized()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    enable_chunks = (
        request.enable_chunks
        if request.enable_chunks is not None
        else not _config.disable_chunks
    )
    disable_chunks = not enable_chunks
    prompt_type  = request.prompt_type  if request.prompt_type  is not None else _config.system_prompt_mode
    max_chunks   = request.top_k        if request.top_k        is not None else (request.max_chunks if request.max_chunks is not None else _config.top_k)
    temperature  = request.temperature  if request.temperature  is not None else 0.7

    chunks  = _artifacts["chunks"]
    sources = _artifacts["sources"]

    chunk_diagnostics: Dict[int, Dict[str, Any]] = {}
    ordered_ranked_scores: List[float] = []

    try:
        # --- Retrieval, ranking, reranking ---
        if disable_chunks:
            ranked_chunks, topk_idxs = [], []
        else:
            retrieval_result = _retrieve_and_rank(request.query, top_k=max_chunks)

            if (
                not retrieval_result
                or not isinstance(retrieval_result, (list, tuple))
                or len(retrieval_result) != 3
            ):
                topk_idxs, ordered_ranked_scores, raw_scores = [], [], {}
            else:
                topk_idxs, ordered_ranked_scores, raw_scores = retrieval_result

            topk_idxs = [int(i) for i in topk_idxs]

            chunk_diagnostics = _build_chunk_diagnostics(
                topk_idxs, topk_idxs, ordered_ranked_scores, raw_scores
            )

            topk_idxs, ranked_chunks = _apply_reranking(
                request.query, topk_idxs, chunk_diagnostics
            )

        if not _config.gen_model:
            raise HTTPException(status_code=500, detail="Model path not configured.")

        # --- Generation ---
        try:
            answer_text = "".join(
                answer(
                    request.query,
                    ranked_chunks,
                    _config.gen_model,
                    _config.max_gen_tokens,
                    system_prompt_mode=prompt_type,
                    temperature=temperature,
                )
            )
        except Exception as gen_error:
            print(f"Generation failed: {gen_error}")
            answer_text = "I'm sorry, but I couldn't generate a response due to an internal error."

        # --- Page/source metadata ---
        page_nums = get_page_numbers(topk_idxs, _artifacts["meta"]) or {}
        sources_used: set = set()
        chunks_by_page: Dict[int, List[str]] = {}

        for i in topk_idxs[:max_chunks]:
            pages = page_nums.get(i, [1])
            if isinstance(pages, list):
                for page in pages:
                    sources_used.add(SourceItem(page=int(page), text=sources[i]))
                    chunks_by_page.setdefault(int(page), []).append(chunks[i])
            elif isinstance(pages, int):
                sources_used.add(SourceItem(page=int(pages), text=sources[i]))
                chunks_by_page.setdefault(int(pages), []).append(chunks[i])

        # --- Logging ---
        if _logger:
            success_log = _create_log(
                chunks,
                sources,
                topk_idxs,
                ordered_ranked_scores,
                page_nums,
                [answer_text],
                request,
                enable_chunks,
                prompt_type,
                max_chunks,
                temperature,
                chunk_diagnostics=chunk_diagnostics,
            )
            if not success_log:
                print("Logging failed for this request.")

        answer_id  = str(uuid4())
        session_id = request.session_id or str(uuid4())
        retrieval_info = {
            "chunks_used": topk_idxs[:max_chunks],
            "page_numbers": get_page_numbers(topk_idxs, _artifacts["meta"]),
            "index_prefix": INDEX_PREFIX,
        }
        save_answer(
            answer_id=answer_id,
            session_id=session_id,
            question=request.query,
            answer=answer_text,
            retrieval_info=retrieval_info,
            model=_config.gen_model,
            prompt_mode=prompt_type,
        )

        if _topic_extractor:
            topics = _topic_extractor.extract_topics(request.query)
            difficulty = estimate_difficulty(request.query)
            for topic in topics:
                update_user_topic_state(
                    session_id=session_id,
                    topic=topic,
                    difficulty=difficulty,
                    delta_confidence=0.0,
                    evidence={
                        "type": "question",
                        "question": request.query,
                        "answer_id": answer_id,
                    },
                )

        return ChatResponse(
            answer_id=answer_id,
            session_id=session_id,
            answer=answer_text.strip() or "No response generated",
            sources=list(sources_used),
            chunks_used=topk_idxs,
            chunks_by_page=chunks_by_page,
            query=request.query,
        )

    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)