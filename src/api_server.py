"""
FastAPI server for TokenSmith chat functionality.
Provides REST API endpoints for the React frontend.
"""

import asyncio
import pathlib
import re
from typing import Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import QueryPlanConfig
from src.generator import answer
from src.instrumentation.logging import init_logger, get_logger
from src.ranking.ranker import EnsembleRanker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts


# Global variables for loaded artifacts
_artifacts = None
_retrievers = None
_ranker = None
_config = None
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize artifacts on startup."""
    global _artifacts, _retrievers, _ranker, _config, _logger
    
    # Load configuration
    config_path = pathlib.Path("src/config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("No config file found at src/config/config.yaml")
    
    _config = QueryPlanConfig.from_yaml(config_path)
    _logger = init_logger(_config)
    
    # Load artifacts
    try:
        artifacts_dir = _config.make_artifacts_directory()
        faiss_index, bm25_index, chunks, sources = load_artifacts(
            artifacts_dir=artifacts_dir, 
            index_prefix="textbook_index"
        )
        
        _artifacts = {
            "faiss_index": faiss_index,
            "bm25_index": bm25_index,
            "chunks": chunks,
            "sources": sources
        }
        
        _retrievers = [
            FAISSRetriever(faiss_index, _config.embed_model),
            BM25Retriever(bm25_index)
        ]
        
        _ranker = EnsembleRanker(
            ensemble_method=_config.ensemble_method,
            weights=_config.ranker_weights,
            rrf_k=int(_config.rrf_k)
        )
        
        print("✅ TokenSmith API initialized successfully")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not load artifacts: {e}")
        print("   Run indexing first or check your configuration")
    
    yield
    
    # Cleanup on shutdown
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
    
    if not _artifacts or not _retrievers or not _ranker:
        return {"error": "Artifacts not loaded", "status": "error"}
    
    if not request.query.strip():
        return {"error": "Query cannot be empty", "status": "error"}
    
    try:
        # Just do retrieval and ranking, skip generation
        pool_n = max(_config.pool_size, _config.top_k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        
        for retriever in _retrievers:
            raw_scores[retriever.name] = retriever.get_scores(
                request.query, pool_n, _artifacts["chunks"]
            )
        
        # Step 2: Ranking
        ordered = _ranker.rank(raw_scores=raw_scores)
        topk_idxs = apply_seg_filter(_config, _artifacts["chunks"], ordered)
        
        ranked_chunks = [_artifacts["chunks"][i] for i in topk_idxs]
        
        return {
            "status": "success",
            "query": request.query,
            "chunks_found": len(ranked_chunks),
            "top_chunks": ranked_chunks[:3],  # First 3 chunks
            "message": "Retrieval and ranking successful, generation skipped"
        }
        
    except Exception as e:
        print(f"Test chat error: {str(e)}")
        return {"error": str(e), "status": "error"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    print(f"🔍 Received chat request: {request.query}")  # Debug logging
    
    if not _artifacts or not _retrievers or not _ranker:
        print("Artifacts not loaded")
        raise HTTPException(
            status_code=503, 
            detail="Artifacts not loaded. Please run indexing first."
        )
    
    if not request.query.strip():
        print("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Log query start (with error handling)
        if _logger:
            _logger.log_query_start(request.query)
        else:
            print(f"Logger not available, skipping query logging")
        
        # Step 1: Retrieval
        print(f"Starting retrieval step...")
        pool_n = max(_config.pool_size, _config.top_k + 10)
        print(f"Pool size: {pool_n}")
        raw_scores: Dict[str, Dict[int, float]] = {}
        
        print(f"Number of retrievers: {len(_retrievers)}")
        
        # TEMPORARY WORKAROUND: Only use BM25 retriever to avoid FAISS segfault
        print(f"Using BM25-only mode to avoid FAISS segfault...")
        for i, retriever in enumerate(_retrievers):
            if retriever.name == "bm25":
                print(f"Running BM25 retriever only: {retriever.name}")
                try:
                    scores = retriever.get_scores(
                        request.query, pool_n, _artifacts["chunks"]
                    )
                    raw_scores[retriever.name] = scores
                    print(f"BM25 retriever completed successfully")
                except Exception as retriever_error:
                    print(f"BM25 retriever failed: {str(retriever_error)}")
                    raise retriever_error
            else:
                print(f"Skipping {retriever.name} retriever to avoid segfault")
        
        print(f"Retrieval completed. Raw scores keys: {list(raw_scores.keys())}")
        
        # Step 2: Ranking
        print(f"Starting ranking step...")
        try:
            ordered = _ranker.rank(raw_scores=raw_scores)
            print(f"Ranking completed successfully")
        except Exception as ranking_error:
            print(f"Ranking failed: {str(ranking_error)}")
            raise ranking_error
        
        print(f"Starting segment filter...")
        try:
            topk_idxs = apply_seg_filter(_config, _artifacts["chunks"], ordered)
            print(f"Segment filter completed. Top K indices: {len(topk_idxs)}")
        except Exception as filter_error:
            print(f"Segment filter failed: {str(filter_error)}")
            raise filter_error
        
        if _logger:
            _logger.log_chunks_used(topk_idxs, _artifacts["chunks"], _artifacts["sources"])
        
        ranked_chunks = [_artifacts["chunks"][i] for i in topk_idxs]
        
        # Step 3: Generation
        max_tokens = _config.max_gen_tokens
        model_path = _config.model_path
        
        print(f"Starting generation with model: {model_path}")
        print(f"Max tokens: {max_tokens}")
        print(f"Number of chunks: {len(ranked_chunks)}")
        
        # Run generation synchronously (like main.py) to avoid threading issues
        try:
            print(f"About to call answer function...")
            print(f"Query: {request.query}")
            print(f"Chunks count: {len(ranked_chunks)}")
            print(f"Model path: {model_path}")
            print(f"Max tokens: {max_tokens}")
            
            # Test with a simple call first
            if len(ranked_chunks) > 0:
                print(f"First chunk preview: {ranked_chunks[0][:100]}...")
            
            print(f"Calling answer function directly (synchronous)...")
            # Call answer directly like main.py does - no thread pool
            answer_text = answer(
                request.query, 
                ranked_chunks, 
                model_path, 
                max_tokens
            )
            print(f"Generation completed successfully: {answer_text[:100]}...")
        except Exception as gen_error:
            print(f"Generation failed: {str(gen_error)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Return a fallback response instead of crashing
            answer_text = f"I apologize, but I encountered an error while generating a response. The query was: '{request.query}'. Please try again or check the server logs for more details."
        
        # Prepare sources for the chunks used - convert to new format
        sources_used = []
        for i in topk_idxs:
            source_text = _artifacts["sources"][i]
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
            _logger.log_error(str(e))
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
