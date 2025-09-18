from typing import Dict, List, Any
from .rankers import Ranker, Candidate


class EnsembleRanker:
    """
    Computes weighted linear fusion of normalized ranker scores
    or RRF (ensemble_method should be one of 'linear' and 'rrf').
    Supported rankers: 'faiss', 'bm25', 'tf-idf'

    Example weights: {"faiss": 0.6, "bm25": 0.25, "tf-idf": 0.15}
    Weights must sum to 1
    """
    def __init__(self, ensemble_method: str, rankers: List[Ranker], weights: Dict[str, float], rrf_k: int = 60):
        self.ensemble_method = ensemble_method.lower().strip()
        self.rankers = rankers
        self.weights = {k: float(v) for k, v in weights.items()}
        self.rrf_k = int(rrf_k)

    @staticmethod
    def _to_rank(score: Dict[int, float], order: List[int]) -> Dict[int, int]:
        """
        Turn a score dict into 1-based ranks. Ties broken by the candidate order.
        """
        pos = {idx: p for p, idx in enumerate(order)}
        sorted_idxs = sorted(order, key=lambda i: (-score.get(i, 0.0), pos[i]))
        return {idx: r for r, idx in enumerate(sorted_idxs, start=1)}

    def _rrf_fuse(self, rank_dicts: List[Dict[int, int]], order: List[int]) -> List[int]:
        fused: Dict[int, float] = {}
        for idx in order:
            s = 0.0
            for rd in rank_dicts:
                if idx in rd:
                    s += 1.0 / (self.rrf_k + rd[idx])
            fused[idx] = s
        return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])]

    def rank(self, *, query: str, chunks: List[str], cand_idxs: List[int], context: Dict[str, Any]) -> List[int]:
        # 1) prepare active rankers
        active = [r for r in self.rankers if self.weights.get(r.name, 0.0) > 0.0]
        for r in active:
            r.prepare(query=query, chunks=chunks, cand_idxs=cand_idxs, context=context)

        # 2) collect scores from each ranker
        per_ranker_scores: List[Dict[int, float]] = []
        for r in active:
            raw = r.score(query=query, chunks=chunks, cand_idxs=cand_idxs, context=context)
            per_ranker_scores.append(raw)
            # log ranker scores
            try:
                from src.instrumentation.logging import get_logger
                get_logger().log_ranking_scores(r.name, raw, cand_idxs)
            except Exception:
                print(f"[WARNING] Logging failed for ranker {r.name}")

        # 3) RRF
        if self.ensemble_method == "rrf":
            rank_dicts = [self._to_rank(scores, cand_idxs) for scores in per_ranker_scores]
            ordered = self._rrf_fuse(rank_dicts, cand_idxs)
            try:
                from src.instrumentation.logging import get_logger
                get_logger().log_ensemble_result(ordered, self.ensemble_method, self.weights)
            except Exception:
                pass
            return ordered

        if self.ensemble_method != "linear" and self.ensemble_method != "weighted":
            print(f'[WARNING] {self.ensemble_method} not implemented yet. Defaulting to linear.')

        # 3) Linear weighted fusion
        combined: Dict[int, float] = {i: 0.0 for i in cand_idxs}
        for i, raw in enumerate(per_ranker_scores):
            weight = self.weights.get(active[i].name)
            if weight <= 0.0:
                continue
            normalized_score = normalize(raw)
            for j in cand_idxs:
                combined[j] += weight * normalized_score.get(j, 0.0)

        ordered = [i for i, _ in sorted(combined.items(), key=lambda kv: kv[1], reverse=True)]
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
