"""
retriever.py

Stores core retrieval logic using FAISS and BM25 scoring.
It also contains helpers for loading artifacts and filtering chunks.
"""

from __future__ import annotations

import pathlib
import os
import pickle
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import QueryPlanConfig


# -------------------------- Embedder cache ------------------------------

_EMBED_CACHE: Dict[str, SentenceTransformer] = {}

def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBED_CACHE:
        # Use the cached embedding model to avoid reloading it on every call
        _EMBED_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
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


def load_summary_artifacts(artifacts_dir: os.PathLike, index_prefix: str) -> Tuple[Optional[faiss.Index], Optional[List[Dict]]]:
    """
    Loads summary artifacts if they exist:
      - FAISS summary index: {index_prefix}_summaries.faiss
      - summaries:          {index_prefix}_summaries.pkl
      
    Returns:
        Tuple of (summary_index, summaries) or (None, None) if not available
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    summary_index_path = artifacts_dir / f"{index_prefix}_summaries.faiss"
    summaries_path = artifacts_dir / f"{index_prefix}_summaries.pkl"
    
    if not (summary_index_path.exists() and summaries_path.exists()):
        return None, None
    
    try:
        summary_index = faiss.read_index(str(summary_index_path))
        summaries = pickle.load(open(summaries_path, "rb"))
        return summary_index, summaries
    except Exception as e:
        print(f"⚠️  Warning: Could not load summary artifacts: {e}")
        return None, None


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
        cleaned = re.sub(r"[^\w\s]", " ", query.lower())
        tokenized_query = cleaned.split()

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


class SummaryRetriever:
    """
    Retriever specifically for section summaries using FAISS.
    Returns top-k summaries based on semantic similarity.
    """
    name = "summaries"

    def __init__(self, summary_index: faiss.Index, summaries: List[Dict], embed_model: str):
        """
        Args:
            summary_index: FAISS index for summaries
            summaries: List of summary dicts with 'heading', 'summary', etc.
            embed_model: Path to embedding model
        """
        self.index = summary_index
        self.summaries = summaries
        self.embedder = _get_embedder(embed_model)

    def get_top_summaries(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Retrieve top-k most relevant summaries for a query.
        
        Args:
            query: User query
            top_k: Number of summaries to retrieve
            
        Returns:
            List of summary dicts
        """
        # Encode query
        q_vec = self.embedder.encode([query]).astype("float32")
        
        # Safety check
        if q_vec.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dim mismatch: index={self.index.d} vs query={q_vec.shape[1]}"
            )
        
        # Search for top summaries
        k = min(top_k, len(self.summaries))
        distances, indices = self.index.search(q_vec, k)
        
        # Get valid indices
        valid_idxs = [i for i in indices[0] if 0 <= i < len(self.summaries)]
        
        # Return corresponding summaries
        return [self.summaries[i] for i in valid_idxs]