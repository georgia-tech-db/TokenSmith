"""
Hybrid retriever combining vector search + keyword search with cross-encoder
reranking.

Retrieval strategy:
  1. VectorIndexRetriever  — semantic similarity via HuggingFace embeddings
  2. KeywordTableSimpleRetriever — BM25-style keyword matching
  3. Union results (OR mode) for maximum recall
  4. Cross-encoder reranking to precision-filter the top-k
"""

from __future__ import annotations

from typing import List

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore

from .config import LlamaIndexConfig


class HybridRetriever(BaseRetriever):
    """
    Combines vector similarity retrieval with keyword-based retrieval.

    Uses 'OR' mode: takes the union of both result sets for maximum recall,
    then relies on downstream reranking for precision.
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: BaseRetriever | None = None,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        if self._keyword_retriever is None:
            return vector_nodes

        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        # Merge: union by node_id, keep higher score when duplicated
        node_map: dict[str, NodeWithScore] = {}
        for n in vector_nodes:
            node_map[n.node.node_id] = n
        for n in keyword_nodes:
            nid = n.node.node_id
            if nid not in node_map or (n.score or 0) > (node_map[nid].score or 0):
                node_map[nid] = n

        # Sort by score descending
        merged = sorted(node_map.values(), key=lambda x: x.score or 0, reverse=True)
        return merged


def build_retriever(
    index: VectorStoreIndex,
    cfg: LlamaIndexConfig,
) -> HybridRetriever:
    """Build the hybrid retriever from an existing index."""
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=cfg.similarity_top_k,
    )

    # Keyword retriever from the same docstore
    keyword_retriever = None
    try:
        from llama_index.core import SimpleKeywordTableIndex

        # Build lightweight keyword index from the same nodes
        nodes = list(index.docstore.docs.values())
        if nodes:
            keyword_index = SimpleKeywordTableIndex(
                nodes,
                storage_context=index.storage_context,
                show_progress=False,
            )
            keyword_retriever = keyword_index.as_retriever(
                retriever_mode="simple",
                similarity_top_k=cfg.keyword_top_k,
            )
    except Exception:
        # Keyword retrieval is optional; fall back to vector-only
        print("Warning: keyword index unavailable, using vector-only retrieval.")

    return HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
    )


def build_reranker(cfg: LlamaIndexConfig) -> SentenceTransformerRerank | None:
    """Build a cross-encoder reranker if enabled."""
    if not cfg.use_reranker:
        return None
    return SentenceTransformerRerank(
        model=cfg.rerank_model,
        top_n=cfg.final_top_k,
    )
