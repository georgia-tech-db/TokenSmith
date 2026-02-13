"""
ranker.py

This module supports ranking strategies applied after chunk retrieval.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

# typedef Candidate as base, we might change this into a class later
# Each candidate is identified by its global index into `chunks`
Candidate = int

class EnsembleRanker:
    """
    Computes weighted reciprocal rank fusion (RRF) or weighted linear fusion of
    normalized retriever scores.
    ensemble_method should be one of 'linear' and 'rrf'.
    Weights must sum to 1. Example weights: {"faiss": 0.6, "bm25": 0.4}.
    """
    def __init__(self, ensemble_method: str, weights: Dict[str, float], rrf_k: int = 60):
        self.ensemble_method = ensemble_method.lower().strip()
        self.weights = {k: float(v) for k, v in weights.items()}
        self.rrf_k = int(rrf_k)

        # Validate that weights for the provided retrievers sum to 1.0
        active_weights = sum(self.weights.values())
        if active_weights != 1.0:
            raise ValueError(f"Weights for active retrivers must sum to 1.0. Current sum: {active_weights}")

    def rank(self, raw_scores: Dict[str, Dict[Candidate, float]]) -> List[int]:
        """
        Executes the rank fusion process on the provided raw scores.
        """
        # Collect scores from each active retriever
        per_retriever_scores: Dict[str, Dict[Candidate, float]] = {}
        for name in raw_scores:
            weight = self.weights.get(name, 0)
            if weight > 0:
                per_retriever_scores[name] = raw_scores[name]

        # Fuse scores using the specified method
        # Shahmeer: if there is only one retriever with weight > 0, just return its ordering, why call this function???
        if self.ensemble_method == "rrf":
            ordered_ids, ordered_scores = self._weighted_rrf_fuse(per_retriever_scores)
        elif self.ensemble_method == "linear":
            ordered_ids, ordered_scores = self._weighted_linear_fuse(per_retriever_scores)
        else:
            raise NotImplementedError(f"Ranking method '{self.ensemble_method}' is not implemented.")

        return ordered_ids, ordered_scores
    
    def _weighted_rrf_fuse(self, per_retriever_scores: Dict[str, Dict[Candidate, float]]) -> Tuple[List[int], List[float]]:
        """Performs Weighted Reciprocal Rank Fusion."""
        fused_scores = defaultdict(float)
        all_candidates = {cand for scores in per_retriever_scores.values() for cand in scores}

        # Convert scores to ranks
        per_retriever_ranks = {
            name: self.scores_to_ranks(scores)
            for name, scores in per_retriever_scores.items()
        }

        for cand in all_candidates:
            current_score = 0.0
            for name, ranks in per_retriever_ranks.items():
                if cand in ranks:
                    weight = self.weights.get(name, 0)
                    current_score += weight * (1.0 / (self.rrf_k + ranks[cand]))
            fused_scores[cand] = current_score

        # 1. Sort the items by score (value) in descending order
        # item[1] is the score, item[0] is the candidate ID
        sorted_items = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

        # 2. Unzip into two lists
        # We use int(cand) to ensure NumPy int64s don't crash the JSON logger
        sorted_ids = [int(cand) for cand, score in sorted_items]
        sorted_scores = [float(score) for cand, score in sorted_items]

        return sorted_ids, sorted_scores

    def _weighted_linear_fuse(self, per_retriever_scores: Dict[str, Dict[Candidate, float]]) -> Tuple[List[int], List[float]]:
        """Performs Weighted Linear Fusion."""
        fused_scores = defaultdict(float)

        # normalize vals per retriever 
        for name, scores in per_retriever_scores.items():
            normalized = self.normalize(scores)
            per_retriever_scores[name] = normalized

        all_candidates = {cand for scores in per_retriever_scores.values() for cand in scores}

        for cand in all_candidates:
            current_score = 0.0
            for name, scores in per_retriever_scores.items():
                if cand in scores:
                    weight = self.weights.get(name, 0)
                    current_score += weight * scores[cand]
            fused_scores[cand] = current_score

        # 1. Sort the items by score (value) in descending order
        sorted_items = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

        # 2. Unzip into two lists with clean types for the JSON logger
        sorted_ids = [int(cand) for cand, score in sorted_items]
        sorted_scores = [float(score) for cand, score in sorted_items]

        return sorted_ids, sorted_scores

    @staticmethod
    def scores_to_ranks(scores: Dict[Candidate, float]) -> Dict[Candidate, int]:
        """Turns a score dictionary into a 1-based rank dictionary."""
        if not scores:
            return {}
        sorted_candidates = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        return {idx: rank for rank, idx in enumerate(sorted_candidates, start=1)}

    @staticmethod
    def normalize(scores: Dict[Candidate, float]) -> Dict[Candidate, float]:
        """Maps arbitrary scores to [0,1] using min-max scaling."""
        if not scores:
            return {}
        vals = list(scores.values())
        min_val, max_val = min(vals), max(vals)
        if max_val <= min_val:
            return {i: 0.0 for i in scores}
        return {i: (v - min_val) / (max_val - min_val) for i, v in scores.items()}
