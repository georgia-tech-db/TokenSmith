from typing import Dict, List, Any
from .rankers import Ranker, Candidate


class EnsembleRanker:
    """
    Computes weighted linear fusion or weighted reciprocal fusion (RRF) of
    normalized ranker scores. ensemble_method should be one of 'linear' and
    'rrf'. Supported rankers: 'faiss', 'bm25'

    Example weights: {"faiss": 0.6, "bm25": 0.4}. Weights must sum to 1
    """
    def __init__(self, ensemble_method: str, rankers: List[Ranker], weights: Dict[str, float], rrf_k: int = 60):
        self.ensemble_method = ensemble_method.lower().strip()
        self.rankers = rankers
        self.weights = {k: float(v) for k, v in weights.items()}
        self.rrf_k = int(rrf_k)
        if sum(self.weights.values()) != 1.0:
            raise ValueError("Ranker weights must sum to 1.0")

    @staticmethod
    def _scores_to_ranks(scores: Dict[Candidate, float]) -> Dict[Candidate, int]:
        """
        Turns a score dictionary into a 1-based rank dictionary.
        """
        # Sort candidates their score in descending order.
        sorted_candidates = sorted(
            scores.keys(),
            key=lambda idx: scores.get(idx, -1e9),
            reverse=True
        )
        
        return {idx: rank for rank, idx in enumerate(sorted_candidates, start=1)}

    def _weighted_rrf_fuse(self, rank_dicts: List[Dict[int, int]], per_ranker_weight: List[float], all_candidates: List[int]) -> List[int]:
        """
        Weighted reciprocal rank fusion of multiple rank dictionaries.
        """
        fused_scores: Dict[int, float] = {}
        for cand in all_candidates:
            score = 0.0
            for rd in rank_dicts:
                if cand in rd:
                    score += (1.0 / (self.rrf_k + rd[cand])) * per_ranker_weight[rank_dicts.index(rd)]
            fused_scores[cand] = score
            
        return [c for c, _ in sorted(fused_scores.items(), key=lambda item: -item[1])]
    
    def _weighted_linear_fuse(self, per_ranker_scores: List[Dict[int, float]], per_ranker_weights: List[float]) -> List[int]:
        """
        Weighted linear fusion of multiple score dictionaries.
        """

        for i, raw in enumerate(per_ranker_scores):
            weight = self.weights.get(active[i].name)
            if weight <= 0.0:
                continue
            normalized_score = normalize(raw)
            for j in cand_idxs:
                combined[j] += weight * normalized_score.get(j, 0.0)

        ordered = [i for i, _ in sorted(combined.items(), key=lambda kv: kv[1], reverse=True)]

    def rank(self, *, raw_metrics: Dict[str, Any]) -> List[int]:
        """
        Executes the rank fusion process on the retrieved candidates.
        """
        all_candidates = set()
        per_ranker_scores: List[Dict[int, float]] = []
        per_ranker_weights: List[float] = []

        # Find active rankers
        active = [r for r in self.rankers if self.weights.get(r.name, 0.0) > 0.0]
        
        # Collect scores and weights from each active ranker
        for r in active:
            scores = r.get_scores(raw_metrics)
            cand_idxs = list(scores.keys())
            all_candidates.update(cand_idxs)
            per_ranker_scores.append(scores)
            per_ranker_weights.append(self.weights.get(r.name, 0.0))
            # log ranker scores
            try:
                from src.instrumentation.logging import get_logger
                get_logger().log_ranking_scores(r.name, scores, cand_idxs)
            except Exception:
                print(f"[WARNING] Logging failed for ranker {r.name}")

        all_candidates = list(all_candidates)

        # RRF and Exit
        if self.ensemble_method == "rrf":
            rank_dicts = [self._scores_to_ranks(scores) for scores in per_ranker_scores]
            ordered = self._rrf_fuse(rank_dicts, per_ranker_weights, all_candidates)
            try:
                from src.instrumentation.logging import get_logger
                get_logger().log_ensemble_result(ordered, self.ensemble_method, self.weights)
            except Exception:
                pass
            return ordered

        if self.ensemble_method != "linear":
            print(f'[WARNING] {self.ensemble_method} not implemented yet. Defaulting to linear.')

        # Linear weighted fusion
        ordered = _weighted_linear_fuse(per_ranker_scores, per_ranker_weights)

        try:
            from src.instrumentation.logging import get_logger
            get_logger().log_ensemble_result(ordered, self.ensemble_method, self.weights)
        except Exception:
            pass
        return ordered

def normalize(scores: Dict[Candidate, float]) -> Dict[Candidate, float]:
    """Map arbitrary scores to [0,1] (safe for ensemble)."""
    if not scores: return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo: return {i: 0.0 for i in scores}
    return {i: (v - lo) / (hi - lo) for i, v in scores.items()}
