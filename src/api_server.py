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
from typing import Dict, List, Optional
import traceback
import os

# Add project root to Python path to allow imports when run directly
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
from src.retriever import filter_retrieved_chunks, BM25Retriever, FAISSRetriever, IndexKeywordRetriever, get_page_numbers, load_artifacts
from src.user_feedback_model import TopicExtractor, estimate_difficulty

# Constants
INDEX_PREFIX = "textbook_index"


# Global state populated during app lifespan
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
        frozen = True  # Makes the model hashable so it can be used in sets


class ChatRequest(BaseModel):
    query: str
    enable_chunks: Optional[bool] = None
    prompt_type: Optional[str] = None  # Maps to system_prompt_mode
    max_chunks: Optional[int] = None  # Maps to top_k for retrieval
    temperature: Optional[float] = None
    top_k: Optional[int] = None  # Alternative name for max_chunks, takes precedence if both provided
    session_id: Optional[str] = None
    gen_model: Optional[str] = None  # Optional generator model override (must exist under models/generators/)


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
    """Return the absolute path to the API config."""
    return pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def _ensure_initialized():
    if not all([_config, _artifacts, _retrievers, _ranker]):
        raise HTTPException(
            status_code=503,
            detail="Artifacts not loaded. Please run indexing first."
        )

def _get_available_gen_models() -> List[str]:
    """Return generator model paths derived from the local models/generators/ directory."""
    models_dir = _project_root / "models" / "generators"
    if not models_dir.exists():
        return []
    return sorted(
        str(pathlib.Path("models") / "generators" / p.name)
        for p in models_dir.iterdir()
        if p.is_file() and p.suffix == ".gguf"
    )

def _resolve_gen_model(requested_model: Optional[str]) -> str:
    """Pick generator model, validating membership if an override is provided."""
    if not _config or not _config.gen_model:
        raise HTTPException(status_code=500, detail="Model path not configured.")
    if not requested_model:
        return _config.gen_model
    available = _get_available_gen_models()
    if requested_model not in available:
        raise HTTPException(status_code=400, detail="Unknown generator model.")
    return requested_model

def _create_log(chunks , sources , topk_idxs, ordered_ranked_scores, page_nums, full_response_accumulator, request,
                 enable_chunks, prompt_type, max_chunks, temperature):
    try:
        # Capture the actual strings used for the log file
        log_chunks = [chunks[i] for i in topk_idxs[:max_chunks]]
        log_sources = [sources[i] for i in topk_idxs[:max_chunks]]
        
        # Just Logging
        _logger.save_chat_log(
            query=request.query,
            config_state=_config.get_config_state(),
            ordered_scores=ordered_ranked_scores,
            chat_request_params={
                "enable_chunks": {
                    "provided": request.enable_chunks,
                    "used": enable_chunks
                },
                "prompt_type": {
                    "provided": request.prompt_type,
                    "used": prompt_type
                },
                "max_chunks": {
                    "provided": request.max_chunks,
                    "used": max_chunks
                },
                "temperature": {
                    "provided": request.temperature,
                    "used": temperature
                }
            },
            top_idxs=topk_idxs[:max_chunks],
            chunks=log_chunks,
            sources=log_sources,
            page_map=page_nums,
            full_response="".join(full_response_accumulator),
            top_k=max_chunks
        )

        return True

    except Exception as log_exc:
        return False

def _retrieve_and_rank(query: str, top_k: Optional[int] = None):
    chunks = _artifacts["chunks"]
    effective_top_k = top_k if top_k is not None else _config.top_k
    pool_n = max(_config.num_candidates, effective_top_k + 10)
    raw_scores: Dict[str, Dict[int, float]] = {}

    for retriever in _retrievers:
        raw_scores[retriever.name] = retriever.get_scores(query, pool_n, chunks)

    ordered_ids, ordered_scores = _ranker.rank(raw_scores=raw_scores)

    if top_k is not None:
        ordered_ids = ordered_ids[:top_k]
        ordered_scores = ordered_scores[:top_k]
    else:
        ordered_ids = ordered_ids[:_config.top_k]
        ordered_scores = ordered_scores[:_config.top_k]

    return ordered_ids, ordered_scores

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize artifacts on startup."""
    global _artifacts, _retrievers, _ranker, _config, _logger, _topic_extractor

    config_path = _resolve_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")

    _config = RAGConfig.from_yaml(config_path)    
    _logger = get_logger()

    try:
        artifacts_dir = _config.get_artifacts_directory()
        faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
            artifacts_dir=artifacts_dir,
            index_prefix=INDEX_PREFIX
        )

        _artifacts = {
            "chunks": chunks,
            "sources": sources,
            "meta": metadata,
        }

        _retrievers = [
            FAISSRetriever(faiss_index, _config.embed_model),
            BM25Retriever(bm25_index),
        ]
        
        # Add index keyword retriever if weight > 0
        if _config.ranker_weights.get("index_keywords", 0) > 0:
            _retrievers.append(
                IndexKeywordRetriever(_config.extracted_index_path, _config.page_to_chunk_map_path)
            )

        _ranker = EnsembleRanker(
            ensemble_method=_config.ensemble_method,
            weights=_config.ranker_weights,
            rrf_k=int(_config.rrf_k),
        )

        init_feedback_db()
        if _config.enable_topic_extraction:
            _topic_extractor = TopicExtractor(
                extracted_index_path=_config.extracted_index_path,
                page_to_chunk_map_path=_config.page_to_chunk_map_path,
            )
        else:
            _topic_extractor = None

        print("TokenSmith API initialized successfully")
    except Exception as exc:
        print(f"Warning: Could not load artifacts: {exc}")
        print("   Run indexing first or check your configuration")

    yield

    print("🔄 Shutting down TokenSmith API...")


# Create FastAPI app
app = FastAPI(
    title="TokenSmith API",
    description="REST API for TokenSmith RAG chat functionality",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3001",  # Alternative React dev server
        "http://localhost:8080",  # Alternative dev server
        "http://127.0.0.1:3000",  # Alternative localhost format
        "http://127.0.0.1:5173",  # Alternative localhost format
        "http://127.0.0.1:3001",  # Alternative localhost format
        "http://127.0.0.1:8080",  # Alternative localhost format
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "TokenSmith API is running"}


@app.post("/api/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """Store user feedback on an answer."""
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

@app.get("/api/models/generators")
async def list_generator_models():
    """List available generator models from the local models/ directory."""
    _ensure_initialized()
    return {
        "available": _get_available_gen_models(),
        "default": _config.gen_model if _config else None,
    }

@app.post("/api/test-chat")
async def test_chat(request: ChatRequest):
    """Test chat endpoint that bypasses generation to isolate issues."""
    print(f"Test chat request: {request.query}")

    try:
        _ensure_initialized()
    except HTTPException as exc:
        return {"error": exc.detail, "status": "error"}

    if not request.query.strip():
        return {"error": "Query cannot be empty", "status": "error"}

    # Parameter handling (aligned with /api/chat)
    enable_chunks = (
        request.enable_chunks
        if request.enable_chunks is not None
        else not _config.disable_chunks
    )
    disable_chunks = not enable_chunks
    max_chunks = (
        request.top_k
        if request.top_k is not None
        else (request.max_chunks if request.max_chunks is not None else _config.top_k)
    )

    if disable_chunks:
        return {
            "error": "Chunk retrieval disabled; enable chunks to test retrieval.",
            "status": "error",
        }

    try:
        # ✅ Correct order (matches /api/chat)
        topk_idxs, ordered_ranked_scores = _retrieve_and_rank(
            request.query, top_k=max_chunks
        )

        # Ensure safe types
        topk_idxs = [int(i) for i in (topk_idxs or [])]
        ordered_ranked_scores = ordered_ranked_scores or {}

        ranked_chunks = [
            _artifacts["chunks"][i] for i in topk_idxs[:max_chunks]
        ]

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
        print(f"Test chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "error"}



@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    import json
    _ensure_initialized()
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    enable_chunks = request.enable_chunks if request.enable_chunks is not None else not _config.disable_chunks
    disable_chunks = not enable_chunks
    prompt_type = request.prompt_type if request.prompt_type is not None else _config.system_prompt_mode
    max_chunks = request.top_k if request.top_k is not None else (request.max_chunks if request.max_chunks is not None else _config.top_k)
    temperature = request.temperature if request.temperature is not None else 0.7
    
    chunks = _artifacts["chunks"]
    sources = _artifacts["sources"]
    
    if disable_chunks:
        ranked_chunks, topk_idxs = [], []
    else:
        topk_idxs, ordered_ranked_scores = _retrieve_and_rank(request.query, top_k=max_chunks)
        topk_idxs = [int(i) for i in topk_idxs]
        ranked_chunks = [chunks[i] for i in topk_idxs[:max_chunks]]
    
    model_path = _resolve_gen_model(request.gen_model)

    answer_id = str(uuid4())
    session_id = request.session_id or str(uuid4())
    
    async def event_generator():
        full_response_accumulator = []
        try:
            page_nums = get_page_numbers(topk_idxs, _artifacts["meta"])
            sources_used = set()
            chunks_by_page: Dict[int, List[str]] = {}
            for i in topk_idxs[:max_chunks]:
                source_text = sources[i]
                pages = page_nums.get(i, [1]) or [1]

                print(f"[DEBUG] i={i} pages={pages!r} page_nums_has_key={i in page_nums}", flush=True)

                for page in pages:
                    chunks_by_page.setdefault(page, []).append(chunks[i])
                    sources_used.add(SourceItem(page=page, text=source_text))
            
            yield f"data: {json.dumps({'type': 'sources', 'content': [s.dict() for s in sources_used]})}\n\n"
            yield f"data: {json.dumps({'type': 'chunks_by_page', 'content': chunks_by_page})}\n\n"

            # Stream generation token by token
            for delta in answer(request.query, ranked_chunks, model_path,
                              _config.max_gen_tokens, system_prompt_mode=prompt_type, temperature=temperature):
                if delta:
                    full_response_accumulator.append(delta) # Capture for log
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
            
            if _logger:
                success_log = _create_log(chunks , sources , topk_idxs, ordered_ranked_scores, page_nums, full_response_accumulator, request,
                            enable_chunks, prompt_type, max_chunks, temperature)
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

            # Include sources in the final done message for completeness
            yield f"data: {json.dumps({'type': 'done', 'answer_id': answer_id, 'session_id': session_id, 'sources': [s.dict() for s in sources_used]})}\n\n"
        except Exception as e:
            # Using print here so you can see crashes in the terminal while debugging
            print(f"Backend error: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint (Non-streaming)."""
    _ensure_initialized()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Parameter setup (Syncing with stream logic)
    enable_chunks = (
        request.enable_chunks
        if request.enable_chunks is not None
        else not _config.disable_chunks
    )
    disable_chunks = not enable_chunks
    prompt_type = (
        request.prompt_type
        if request.prompt_type is not None
        else _config.system_prompt_mode
    )
    max_chunks = (
        request.top_k
        if request.top_k is not None
        else (
            request.max_chunks
            if request.max_chunks is not None
            else _config.top_k
        )
    )
    temperature = request.temperature if request.temperature is not None else 0.7

    chunks = _artifacts["chunks"]
    sources = _artifacts["sources"]

    try:
        # 2. Retrieval & Ranking (SAFE against mocked None return)
        if disable_chunks:
            ranked_chunks, topk_idxs, ordered_ranked_scores = [], [], {}
        else:
            retrieval_result = _retrieve_and_rank(
                request.query, top_k=max_chunks
            )

            # 🔒 Safe unpacking for unit tests where ranker is mocked
            if (
                not retrieval_result
                or not isinstance(retrieval_result, (list, tuple))
                or len(retrieval_result) != 2
            ):
                topk_idxs, ordered_ranked_scores = [], {}
            else:
                topk_idxs, ordered_ranked_scores = retrieval_result

            topk_idxs = [int(i) for i in (topk_idxs or [])]
            ordered_ranked_scores = ordered_ranked_scores or {}

            ranked_chunks = [chunks[i] for i in topk_idxs[:max_chunks]]

        model_path = _resolve_gen_model(request.gen_model)

        # 3. Full Generation
        try:
            answer_text = "".join(
                answer(
                    request.query,
                    ranked_chunks,
                    model_path,
                    _config.max_gen_tokens,
                    system_prompt_mode=prompt_type,
                    temperature=temperature,
                )
            )
        except Exception as gen_error:
            print(f"Generation failed: {str(gen_error)}")
            answer_text = (
                "I'm sorry, but I couldn't generate a response due to an internal error."
            )

        # 4. Post-processing (Metadata & Pages)
        page_nums = get_page_numbers(topk_idxs, _artifacts["meta"]) or {}

        sources_used = set()
        chunks_by_page: Dict[int, List[str]] = {}

        for i in topk_idxs[:max_chunks]:
            source_text = sources[i]
            pages = page_nums.get(i, [1])

            if isinstance(pages, list):
                for page in pages:
                    sources_used.add(SourceItem(page=int(page), text=source_text))
                    chunks_by_page.setdefault(int(page), []).append(chunks[i])
            elif isinstance(pages, int):
                sources_used.add(SourceItem(page=int(pages), text=source_text))
                chunks_by_page.setdefault(int(pages), []).append(chunks[i])
            else: # Error case
                print(f"Unexpected page number format for chunk index {i}: {pages}")


        # 5. Logging
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
            )
            if not success_log:
                print("Logging failed for this request.")


        answer_id = str(uuid4())
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
            answer=answer_text.strip() if answer_text and answer_text.strip() else "No response generated",
            sources=list(sources_used),
            chunks_used=topk_idxs,
            chunks_by_page=chunks_by_page,
            query=request.query,
        )

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
