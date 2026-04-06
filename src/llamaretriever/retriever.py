"""
Retriever and reranker factories for the BookRAG pipeline.

Provides hybrid (vector + BM25 + RRF) retrievers for both the
section index and the leaf index, plus the cross-encoder reranker.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever

from .config import LlamaIndexConfig


def _build_hybrid(index: VectorStoreIndex, top_k: int) -> QueryFusionRetriever:
    vector = index.as_retriever(similarity_top_k=top_k)
    bm25 = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=top_k,
    )
    return QueryFusionRetriever(
        retrievers=[vector, bm25],
        similarity_top_k=top_k,
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
    )


def build_leaf_retriever(
    index: VectorStoreIndex, cfg: LlamaIndexConfig,
) -> QueryFusionRetriever:
    return _build_hybrid(index, cfg.num_candidates)


def build_section_retriever(
    index: VectorStoreIndex, cfg: LlamaIndexConfig,
) -> QueryFusionRetriever:
    return _build_hybrid(index, cfg.section_top_k * 3)


def build_reranker(cfg: LlamaIndexConfig) -> SentenceTransformerRerank | None:
    if not cfg.use_reranker:
        return None
    return SentenceTransformerRerank(
        model=cfg.rerank_model,
        top_n=cfg.max_leaves,
    )
