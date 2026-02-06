"""
FastAPI server for TokenSmith chat functionality.
Provides REST API endpoints for the React frontend.
"""

import sys
import pathlib
import re
import json
from copy import deepcopy
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

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
from src.instrumentation.logging import init_logger, get_logger
from src.ranking.ranker import EnsembleRanker
from src.retriever import filter_retrieved_chunks, BM25Retriever, FAISSRetriever, IndexKeywordRetriever, get_page_numbers, load_artifacts


# Constants
INDEX_PREFIX = "textbook_index"


# Global state populated during app lifespan
_artifacts: Optional[Dict[str, List[str]]] = None
_retrievers: Optional[List] = None
_ranker: Optional[EnsembleRanker] = None
_config: Optional[RAGConfig] = None
_logger = None
INDEX_METADATA: Optional[Dict] = None


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


class ChatResponse(BaseModel):
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


def _retrieve_and_rank(query: str, top_k: Optional[int] = None):
    chunks = _artifacts["chunks"]
    effective_top_k = top_k if top_k is not None else _config.top_k
    pool_n = max(_config.num_candidates, effective_top_k + 10)
    raw_scores: Dict[str, Dict[int, float]] = {}

    for retriever in _retrievers:
        raw_scores[retriever.name] = retriever.get_scores(query, pool_n, chunks)

    ordered = _ranker.rank(raw_scores=raw_scores)
    # Create a temporary config with the effective top_k for filtering
    temp_config = _config
    if top_k is not None:
        temp_config = deepcopy(_config)
        temp_config.top_k = effective_top_k
    topk_idxs = filter_retrieved_chunks(temp_config, chunks, ordered)
    return raw_scores, topk_idxs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize artifacts on startup."""
    global _artifacts, _retrievers, _ranker, _config, _logger, INDEX_METADATA

    config_path = _resolve_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")

    _config = RAGConfig.from_yaml(config_path)
    init_logger(_config)
    _logger = get_logger()

    try:
        artifacts_dir = _config.get_artifacts_directory()
        
        # Load index metadata manifest
        meta_path = artifacts_dir / f"{INDEX_PREFIX}_index_info.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                INDEX_METADATA = json.load(f)
        else:
            INDEX_METADATA = {"status": "not_indexed", "indexed_chapters": []}

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


@app.get("/api/index/status")
def get_index_status():
    """
    Returns metadata about the currently loaded index, 
    including which chapters are indexed.
    """
    return INDEX_METADATA


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
    
    # Get request parameters with config defaults as fallback
    enable_chunks = request.enable_chunks if request.enable_chunks is not None else not _config.disable_chunks
    disable_chunks = not enable_chunks
    max_chunks = request.top_k if request.top_k is not None else (request.max_chunks if request.max_chunks is not None else _config.top_k)
    
    if disable_chunks:
        return {
            "error": "Chunk retrieval disabled; enable chunks to test retrieval.",
            "status": "error",
        }

    try:
        raw_scores, topk_idxs = _retrieve_and_rank(request.query, top_k=max_chunks)
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
    
    # Get request parameters with config defaults as fallback
    enable_chunks = request.enable_chunks if request.enable_chunks is not None else not _config.disable_chunks
    disable_chunks = not enable_chunks
    prompt_type = request.prompt_type if request.prompt_type is not None else _config.system_prompt_mode
    max_chunks = request.top_k if request.top_k is not None else (request.max_chunks if request.max_chunks is not None else _config.top_k)
    temperature = request.temperature if request.temperature is not None else _config.temperature
    
    chunks = _artifacts["chunks"]
    sources = _artifacts["sources"]
    
    if _logger:
        _logger.log_query_start(request.query)
    
    if disable_chunks:
        ranked_chunks, topk_idxs = [], []
    else:
        _, topk_idxs = _retrieve_and_rank(request.query, top_k=max_chunks)
        ranked_chunks = [chunks[i] for i in topk_idxs[:max_chunks]]
        if _logger:
            _logger.log_chunks_used(topk_idxs, chunks, sources)
    
    if not _config.gen_model:
        raise HTTPException(status_code=500, detail="Model path not configured.")

    
    async def event_generator():
        try:
            # First send the references (page/text pairs) and chunks by page
            page_nums = get_page_numbers(topk_idxs, _artifacts["meta"])
            sources_used = set()
            chunks_by_page: Dict[int, List[str]] = {}
            for i in topk_idxs[:max_chunks]:
                source_text = sources[i]
                page = page_nums[i] if i in page_nums else 1
                sources_used.add(SourceItem(page=page, text=source_text))
                chunks_by_page.setdefault(page, []).append(chunks[i])
            
            # Remove duplicates by converting to set of tuples, then back to SourceItem
            yield f"data: {json.dumps({'type': 'sources', 'content': [s.dict() for s in sources_used]})}\n\n"
            yield f"data: {json.dumps({'type': 'chunks_by_page', 'content': chunks_by_page})}\n\n"

            for delta in answer(request.query, ranked_chunks, _config.gen_model,
                              _config.max_gen_tokens, system_prompt_mode=prompt_type, temperature=temperature):
                if delta:
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
            
            # Include sources in the final done message for completeness
            yield f"data: {json.dumps({'type': 'done', 'sources': [s.dict() for s in sources_used]})}\n\n"
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
    
    # Get request parameters with config defaults as fallback
    enable_chunks = request.enable_chunks if request.enable_chunks is not None else not _config.disable_chunks
    disable_chunks = not enable_chunks
    prompt_type = request.prompt_type if request.prompt_type is not None else _config.system_prompt_mode
    max_chunks = request.top_k if request.top_k is not None else (request.max_chunks if request.max_chunks is not None else _config.top_k)
    temperature = request.temperature if request.temperature is not None else _config.temperature
    
    try:
        chunks = _artifacts["chunks"]
        sources = _artifacts["sources"]

        if _logger:
            _logger.log_query_start(request.query)
        else:
            print("Logger not available, skipping query logging")
        
        if disable_chunks:
            print("Chunk usage disabled; skipping retrieval.")
            ranked_chunks: List[str] = []
            topk_idxs: List[int] = []
        else:
            _, topk_idxs = _retrieve_and_rank(request.query, top_k=max_chunks)
            ranked_chunks = [chunks[i] for i in topk_idxs[:max_chunks]]
            if _logger:
                _logger.log_chunks_used(topk_idxs, chunks, sources)

        max_tokens = _config.max_gen_tokens
        model_path = _config.gen_model

        if not model_path:
            raise HTTPException(status_code=500, detail="Model path not configured.")

        try:
            answer_text = "".join(answer(
                request.query,
                ranked_chunks,
                model_path,
                max_tokens,
                system_prompt_mode=prompt_type,
                temperature=temperature,
            ))
        except Exception as gen_error:
            print(f"Generation failed: {str(gen_error)}")
            if _logger:
                _logger.log_error(gen_error, context="generation")
            answer_text = (
                "I'm sorry, but I couldn't generate a response due to an internal error. "
                "Please try again or check the server logs for more details."
            )

        sources_used = set()
        chunks_by_page: Dict[int, List[str]] = {}
        for i in topk_idxs[:max_chunks]:
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
            
            sources_used.add(SourceItem(page=page, text=source_text))
            chunks_by_page.setdefault(page, []).append(chunks[i])
        
        if _logger:
            _logger.log_generation(
                answer_text, 
                {"max_tokens": max_tokens, "model_path": model_path}
            )
        
        return ChatResponse(
            answer=answer_text.strip() if answer_text and answer_text.strip() else "No response generated",
            sources=sources_used,
            chunks_used=topk_idxs,
            chunks_by_page=chunks_by_page,
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
