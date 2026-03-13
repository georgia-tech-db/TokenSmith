# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import pathlib
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import RAGConfig
from src.generator import answer, dedupe_generated_text
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    filter_retrieved_chunks,
    load_artifacts,
)
from src.ranking.reranker import rerank


# --------------- Configuration ---------------

CONFIG_PATH = pathlib.Path("config/config.yaml")
INDEX_PREFIX = "textbook_index"

# --------------- Global state ---------------

_artifacts: Optional[Dict] = None
_config: Optional[RAGConfig] = None


# --------------- Pydantic models ---------------

class SchedulerRequest(BaseModel):
    queries: List[str]


class QueryResult(BaseModel):
    query: str
    answer: str


class SchedulerResponse(BaseModel):
    results: List[QueryResult]


# --------------- Pipeline helpers ---------------

def _retrieve_and_rank(query: str, top_k: Optional[int] = None):
    """Run retrieval and ranking for a single query."""
    chunks = _artifacts["chunks"]
    retrievers = _artifacts["retrievers"]
    ranker = _artifacts["ranker"]

    k = top_k or _config.top_k
    pool_n = max(_config.num_candidates, k + 10)

    raw_scores: Dict[str, Dict[int, float]] = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(query, pool_n, chunks)

    ordered, scores = ranker.rank(raw_scores=raw_scores)
    topk_idxs = filter_retrieved_chunks(_config, chunks, ordered)

    return topk_idxs, scores


def _run_query(query: str) -> str:
    """Run a single query through the full pipeline and return the answer text."""
    chunks = _artifacts["chunks"]

    # Retrieval & ranking
    topk_idxs, _ = _retrieve_and_rank(query)
    ranked_chunks = [chunks[i] for i in topk_idxs]

    # Re-ranking
    ranked_chunks = rerank(
        query, ranked_chunks, mode=_config.rerank_mode, top_n=_config.rerank_top_k
    )

    if not ranked_chunks:
        return "I'm sorry, but I don't have enough information to answer that question."

    # Generation
    stream_iter = answer(
        query,
        ranked_chunks,
        _config.gen_model,
        _config.max_gen_tokens,
        system_prompt_mode=_config.system_prompt_mode,
    )

    answer_text = ""
    for delta in stream_iter:
        answer_text += delta

    return dedupe_generated_text(answer_text).strip()


# --------------- App lifecycle ---------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts and models on startup."""
    global _artifacts, _config

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No config file found at {CONFIG_PATH}")

    _config = RAGConfig.from_yaml(CONFIG_PATH)

    try:
        artifacts_dir = _config.get_artifacts_directory()
        faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
            artifacts_dir=artifacts_dir, index_prefix=INDEX_PREFIX
        )

        retrievers = [
            FAISSRetriever(faiss_index, _config.embed_model),
            BM25Retriever(bm25_index),
        ]

        if _config.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(
                IndexKeywordRetriever(
                    _config.extracted_index_path, _config.page_to_chunk_map_path
                )
            )

        ranker = EnsembleRanker(
            ensemble_method=_config.ensemble_method,
            weights=_config.ranker_weights,
            rrf_k=int(_config.rrf_k),
        )

        _artifacts = {
            "chunks": chunks,
            "sources": sources,
            "meta": metadata,
            "retrievers": retrievers,
            "ranker": ranker,
        }

        print("Remote scheduler initialized successfully")
    except Exception as exc:
        print(f"Warning: Could not load artifacts: {exc}")
        print("   Run indexing first or check your configuration")

    yield

    print("Shutting down remote scheduler...")


# --------------- FastAPI app ---------------

app = FastAPI(
    title="TokenSmith Remote Scheduler",
    description="Receives queries from the local scheduler and runs them on the large model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/scheduler/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "initialized": _artifacts is not None,
    }


@app.post("/scheduler/run", response_model=SchedulerResponse)
async def run_queries(request: SchedulerRequest):
    """Run a batch of queries through the pipeline."""
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    results = []
    for query in request.queries:
        query = query.strip()
        if not query:
            continue
        try:
            answer_text = _run_query(query)
            results.append(QueryResult(query=query, answer=answer_text))
        except Exception as e:
            print(f"Error processing query '{query[:50]}...': {e}")
            results.append(
                QueryResult(query=query, answer=f"[error] {e}")
            )

    return SchedulerResponse(results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
