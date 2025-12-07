"""
FastAPI server for TokenSmith chat functionality.
Provides REST API endpoints for the React frontend.
"""

import pathlib
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import QueryPlanConfig
from src.generator import answer
from src.instrumentation.logging import init_logger, get_logger
from src.ranking.ranker import EnsembleRanker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, IndexKeywordRetriever, get_page_numbers, load_artifacts


# Constants
INDEX_PREFIX = "textbook_index"


# Global state populated during app lifespan
_artifacts: Optional[Dict[str, List[str]]] = None
_retrievers: Optional[List] = None
_ranker: Optional[EnsembleRanker] = None
_config: Optional[QueryPlanConfig] = None
_logger = None


class SourceItem(BaseModel):
    page: int
    text: str


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    chunks_used: List[int]
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


def _retrieve_and_rank(query: str):
    chunks = _artifacts["chunks"]
    pool_n = max(_config.pool_size, _config.top_k + 10)
    raw_scores: Dict[str, Dict[int, float]] = {}

    for retriever in _retrievers:
        raw_scores[retriever.name] = retriever.get_scores(query, pool_n, chunks)

    ordered = _ranker.rank(raw_scores=raw_scores)
    topk_idxs = apply_seg_filter(_config, chunks, ordered)
    return raw_scores, topk_idxs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize artifacts on startup."""
    global _artifacts, _retrievers, _ranker, _config, _logger

    config_path = _resolve_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")

    _config = QueryPlanConfig.from_yaml(config_path)
    init_logger(_config)
    _logger = get_logger()

    try:
        artifacts_dir = _config.make_artifacts_directory()
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

        print("✅ TokenSmith API initialized successfully")
    except Exception as exc:
        print(f"⚠️  Warning: Could not load artifacts: {exc}")
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


@app.post("/api/test-chat")
async def test_chat(request: ChatRequest):
    """Test chat endpoint that bypasses generation to isolate issues."""
    print(f"🔍 Test chat request: {request.query}")
    
    try:
        _ensure_initialized()
    except HTTPException as exc:
        return {"error": exc.detail, "status": "error"}
    
    if not request.query.strip():
        return {"error": "Query cannot be empty", "status": "error"}
    
    if _config.disable_chunks:
        return {
            "error": "Chunk retrieval disabled in configuration; enable chunks to test retrieval.",
            "status": "error",
        }

    try:
        raw_scores, topk_idxs = _retrieve_and_rank(request.query)
        ranked_chunks = [_artifacts["chunks"][i] for i in topk_idxs]
        
        return {
            "status": "success",
            "query": request.query,
            "chunks_found": len(ranked_chunks),
            "top_chunks": ranked_chunks[:3],  # First 3 chunks
            "raw_scores": raw_scores,
            "message": "Retrieval and ranking successful, generation skipped"
        }
        
    except Exception as e:
        print(f"Test chat error: {str(e)}")
        return {"error": str(e), "status": "error"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    import json
    _ensure_initialized()
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    chunks = _artifacts["chunks"]
    sources = _artifacts["sources"]
    
    if _logger:
        _logger.log_query_start(request.query)
    
    if _config.disable_chunks:
        ranked_chunks, topk_idxs = [], []
    else:
        _, topk_idxs = _retrieve_and_rank(request.query)
        ranked_chunks = [chunks[i] for i in topk_idxs]
        if _logger:
            _logger.log_chunks_used(topk_idxs, chunks, sources)
    
    if not _config.model_path:
        raise HTTPException(status_code=500, detail="Model path not configured.")

    
    async def event_generator():
        try:
            # First send the references (page/text pairs)
            page_nums = get_page_numbers(topk_idxs, _artifacts["meta"])
            sources_used = []
            for i in topk_idxs:
                source_text = chunks[i]
                page = page_nums[i] if i in page_nums else 1
                sources_used.append(SourceItem(page=page, text=source_text))

            yield f"data: {json.dumps({'type': 'sources', 'content': [s.dict() for s in sources_used]})}\n\n"

            for delta in answer(request.query, ranked_chunks, _config.model_path,
                              _config.max_gen_tokens, system_prompt_mode=_config.system_prompt_mode):
                if delta:
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            if _logger:
                _logger.log_error(e, context="streaming")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    print(f"🔍 Received chat request: {request.query}")  # Debug logging
    
    _ensure_initialized()
    
    if not request.query.strip():
        print("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        chunks = _artifacts["chunks"]
        sources = _artifacts["sources"]

        if _logger:
            _logger.log_query_start(request.query)
        else:
            print("Logger not available, skipping query logging")
        
        if _config.disable_chunks:
            print("Chunk usage disabled in configuration; skipping retrieval.")
            ranked_chunks: List[str] = []
            topk_idxs: List[int] = []
        else:
            _, topk_idxs = _retrieve_and_rank(request.query)
            ranked_chunks = [chunks[i] for i in topk_idxs]
            if _logger:
                _logger.log_chunks_used(topk_idxs, chunks, sources)

        max_tokens = _config.max_gen_tokens
        model_path = _config.model_path

        if not model_path:
            raise HTTPException(status_code=500, detail="Model path not configured.")

        try:
            answer_text = "".join(answer(
                request.query,
                ranked_chunks,
                model_path,
                max_tokens,
                system_prompt_mode=_config.system_prompt_mode,
            ))
        except Exception as gen_error:
            print(f"Generation failed: {str(gen_error)}")
            if _logger:
                _logger.log_error(gen_error, context="generation")
            answer_text = (
                "I’m sorry, but I couldn’t generate a response due to an internal error. "
                "Please try again or check the server logs for more details."
            )

        sources_used = []
        for i in topk_idxs:
            source_text = sources[i]
            # Extract page number from source text if available, otherwise use 1
            page = 1
            if "page" in source_text.lower():
                try:
                    # Try to extract page number from source text
                    page_match = re.search(r'page\s*(\d+)', source_text.lower())
                    if page_match:
                        page = int(page_match.group(1))
                except (ValueError, AttributeError):
                    page = 1
            
            sources_used.append(SourceItem(page=page, text=source_text))
        
        if _logger:
            _logger.log_generation(
                answer_text, 
                {"max_tokens": max_tokens, "model_path": model_path}
            )
        
        return ChatResponse(
            answer=answer_text.strip() if answer_text and answer_text.strip() else "No response generated",
            sources=sources_used,
            chunks_used=topk_idxs,
            query=request.query
        )
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        if _logger:
            _logger.log_error(e)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
