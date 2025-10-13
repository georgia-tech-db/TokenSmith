from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


# typedef Candidate as base, we might change this into a class later
# Each candidate is identified by its global index into `chunks`
Candidate = int


class Ranker(ABC):
    name: str

    @abstractmethod
    def get_scores(self, *, raw_metrics: Dict[str, Any]) -> Dict[Candidate, float]:
        """Return raw, comparable scores for each candidate (higher is better)."""

class FaissSimilarityRanker(Ranker):
    name = "faiss"

    def get_scores(self, *, raw_metrics: Dict[str, Any]) -> Dict[Candidate, float]:
        """
        Converts L2 distance from FAISS to a similarity score.
        A smaller distance results in a higher similarity score.
        """
        faiss_distances = raw_metrics.get("faiss_distances", {})
        if not faiss_distances:
            return {}
        
        # Invert distance to score: 1 / (1 + distance). Adding 1 avoids division by zero.
        return {
            idx: 1.0 / (1.0 + dist)
            for idx, dist in faiss_distances.items()
        }

class BM25Ranker(Ranker):
    name = "bm25"

    def get_scores(self, *, raw_metrics: Dict[str, Any]) -> Dict[Candidate, float]:
        """
        Returns the raw BM25 scores. Higher is already better.
        """
        # BM25 scores are already relevance scores, so we can use them directly.
        return raw_metrics.get("bm25_scores", {})