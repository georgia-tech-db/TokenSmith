"""
retriever.py

Stores core retrieval logic using FAISS and BM25 scoring.
It also contains helpers for loading artifacts and filtering chunks.
"""

from __future__ import annotations

import pathlib
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict

import faiss
import numpy as np
from src.embedder import SentenceTransformer

from src.config import QueryPlanConfig
from src.index_builder import preprocess_for_bm25


# -------------------------- Embedder cache ------------------------------

_EMBED_CACHE: Dict[str, SentenceTransformer] = {}

def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBED_CACHE:
        # Use the cached embedding model to avoid reloading it on every call
        _EMBED_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBED_CACHE[model_name]


# -------------------------- Read artifacts -------------------------------

def load_artifacts(artifacts_dir: os.PathLike, index_prefix: str) -> Tuple[faiss.Index, List[str], List[str]]:
    """
    Loads:
      - FAISS index: {index_prefix}.faiss
      - chunks:      {index_prefix}_chunks.pkl
      - sources:     {index_prefix}_sources.pkl
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    faiss_index = faiss.read_index(str(artifacts_dir / f"{index_prefix}.faiss"))
    bm25_index  = pickle.load(open(artifacts_dir / f"{index_prefix}_bm25.pkl", "rb"))
    chunks      = pickle.load(open(artifacts_dir / f"{index_prefix}_chunks.pkl", "rb"))
    sources     = pickle.load(open(artifacts_dir / f"{index_prefix}_sources.pkl", "rb"))

    return faiss_index, bm25_index, chunks, sources


# -------------------------- Pretty previews -----------------------------

def _print_preview(chunks: List[str], n_preview: int = 100) -> None:
    for i, c in enumerate(chunks, 1):
        snippet = (c or "")[:n_preview].replace("\n", " ")
        print(f"[retriever] top{i:02d} → {len(c)} chars | {snippet!r}")


def _print_preview_idxs(
    chunks: List[str],
    srcs: List[str],
    tags: Optional[List[List[str]]],
    idxs: List[int],
    n_preview: int = 100,
) -> None:
    for rank, i in enumerate(idxs, 1):
        snippet = (chunks[i] or "")[:n_preview].replace("\n", " ")
        show_tags = (tags[i][:5] if tags else [])
        print(f"[retriever] top{rank:02d} | src={srcs[i]} | tags={show_tags} | {len(chunks[i])} chars | {snippet!r}")


# -------------------------- Filtering logic -----------------------------

def apply_seg_filter(cfg: QueryPlanConfig, chunks, ordered):
    seg_filter = cfg.seg_filter
    if seg_filter:
        keep = [i for i in ordered if seg_filter(chunks[i])]
        back = [i for i in ordered if i not in keep]
        topk_idxs = (keep + back)[:cfg.top_k]
    else:
        topk_idxs = ordered[:cfg.top_k]
    return topk_idxs


# -------------------------- Retrieval core ------------------------------

class Retriever(ABC):
    @abstractmethod
    def get_scores(self, query: str, pool_size: int, chunks: List[str]):
        """Retrieves the top 'pool_size' chunks cores for a given query."""
        pass


class FAISSRetriever(Retriever):
    name = "faiss"

    def __init__(self, index, embed_model: str):
        self.index = index
        self.embedder = _get_embedder(embed_model)

    def get_scores(self,
                query: str,
                pool_size: int,
                chunks: List[str]) -> Dict[int, float]:
        """
        Returns FAISS scores for top 'pool_size' keyed by global chunk index.
        """
        # FAISS expects a 2D array
        q_vec = self.embedder.encode([query]).astype("float32")
        
        # Safety check on vector dimensions
        if q_vec.shape[1] !=  self.index.d:
            raise ValueError(
                f"Embedding dim mismatch: index={ self.index.d} vs query={q_vec.shape[1]}"
            )

        # Perform the search
        distances, indices =  self.index.search(q_vec, pool_size)

        # Remove invalid indices and ensure they are within bounds
        cand_idxs = [i for i in indices[0] if 0 <= i < len(chunks)]

        # Create the distance dictionary, ensuring we only include valid candidates
        dists = {idx: float(dist) for idx, dist in zip(cand_idxs, distances[0][:len(cand_idxs)])}

        # Invert distance to score: 1 / (1 + distance). Adding 1 avoids division by zero.
        return {
            idx: 1.0 / (1.0 + dist)
            for idx, dist in dists.items()
        }


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index):
        self.index = index

    def get_scores(self,
                 query: str,
                 pool_size: int,
                 chunks: List[str]) -> Dict[int, float]:
        """
        Returns BM25 scores for top 'pool_size' keyed by global chunk index.
        """
        # Tokenize the query in the same way the index was built
        tokenized_query = preprocess_for_bm25(query)

        # Get scores for all documents in the corpus
        all_scores = self.index.get_scores(tokenized_query)

        # Find the indices of the top 'pool_size' scores
        num_candidates = min(pool_size, len(all_scores))
        top_k_indices = np.argpartition(-all_scores, kth=num_candidates-1)[:num_candidates]

        # Remove invalid indices and ensure they are within bounds
        top_k_indices = [i for i in top_k_indices if 0 <= i < len(chunks)]
        
        # Get the corresponding scores for the top indices
        top_scores = all_scores[top_k_indices]

        # Format the output as a dictionary of scores
        scores = {int(idx): float(score) for idx, score in zip(top_k_indices, top_scores)}

        return scores
    
# -------------------------- Cascade Retrieval core ------------------------------ #

class CascadeRetriever(Retriever):
    """ Allows for cascaded index searching"""

    def __init__(self, 
                 primary_retrievers: List[Retriever], 
                 secondary_retrievers: List[Retriever], 
                 primary_ranker, 
                 secondary_ranker, 
                 primary_chunks, 
                 secondary_chunks, 
                 threshold: float):

        # ----- Primary index retrieval ----- #
        self.primary_retrievers = primary_retrievers
        self.primary_ranker = primary_ranker
        self.primary_chunks = primary_chunks

        # ----- Secondary index retrieval ----- #
        self.secondary_retrievers = secondary_retrievers
        self.secondary_ranker = secondary_ranker
        self.secondary_chunks = secondary_chunks

        self.threshold = threshold

    def get_ranked_chunks(self, query: str, pool_size: int, top_k: int) -> Tuple[List[str], str, float]:
        """
        Look at the most recent source of truth retriever. If the threshold is not met, then look at the second most 
        recent source of truth
        Returns:
        - list of chunks
        - the source ('primary' or 'secondary)
        - the best score from the primary search.
        """
        # query
        primary_scores: Dict[str, Dict[int, float]] = {}
        for retriever in self.primary_retrievers:
            primary_scores[retriever.name] = retriever.get_scores(query, pool_size, self.primary_chunks)

        # Rank
        top_score = 0.0
        primary_topk_idxs = []
        if any(primary_scores.values()):
            primary_ordered = self.primary_ranker.rank(raw_scores=primary_scores)
            primary_topk_idxs = primary_ordered[:top_k]
            
            if primary_topk_idxs:
                top_chunk_idx = primary_topk_idxs[0]
                top_score = primary_scores.get("bm25", {}).get(top_chunk_idx, 0.0)

        #print(self.threshold)
        if top_score >= self.threshold:
            print("[CascadeRetriever] Using primary index.")
            return [self.primary_chunks[i] for i in primary_topk_idxs], "primary", top_score
        else:
            print("[CascadeRetriever] Primary index failed to meet threshold. Using secondary index.")
            # query second latest source of truth
            secondary_scores: Dict[str, Dict[int, float]] = {}
            for retriever in self.secondary_retrievers:
                secondary_scores[retriever.name] = retriever.get_scores(query, pool_size, self.secondary_chunks)
            
            secondary_ordered = self.secondary_ranker.rank(raw_scores=secondary_scores)
            secondary_topk_idxs = secondary_ordered[:top_k]
            
            return [self.secondary_chunks[i] for i in secondary_topk_idxs], "secondary", top_score

    def get_scores(self, query: str, pool_size: int, chunks: List[str]):
        return self.primary_retrievers[0].get_scores(query, pool_size, chunks)