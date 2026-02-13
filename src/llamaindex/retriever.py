"""
Retriever and reranker factories.

Uses LlamaIndex built-in modules to match the original TokenSmith pipeline:
  - VectorIndexRetriever  (equivalent to FAISS)
  - BM25Retriever         (llama-index-retrievers-bm25)
  - QueryFusionRetriever  (reciprocal rank fusion)
  - SentenceTransformerRerank (cross-encoder/ms-marco-MiniLM-L6-v2)
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from .config import LlamaIndexConfig


def build_retriever(
    index: VectorStoreIndex,
    cfg: LlamaIndexConfig,
) -> QueryFusionRetriever:
    """
    Build a hybrid retriever: vector + BM25, fused with RRF.

    Equivalent to the original pipeline's:
      FAISSRetriever + BM25Retriever -> EnsembleRanker(method="rrf")
    """
    vector_retriever = index.as_retriever(similarity_top_k=cfg.num_candidates)

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=cfg.num_candidates,
    )

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=cfg.num_candidates,
        num_queries=1,                  # no query generation, just fuse
        mode="reciprocal_rerank",       # RRF
        use_async=False,
    )


def build_reranker(cfg: LlamaIndexConfig) -> SentenceTransformerRerank | None:
    """Build cross-encoder reranker (same model as original pipeline)."""
    if not cfg.use_reranker:
        return None
    return SentenceTransformerRerank(
        model=cfg.rerank_model,
        top_n=cfg.top_k,
    )
