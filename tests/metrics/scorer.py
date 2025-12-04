from typing import Dict, List, Any, Optional
from tests.metrics.registry import MetricRegistry


class SimilarityScorer:
    """Main scorer that uses selected metrics."""
    
    def __init__(self, enabled_metrics: Optional[List[str]] = None):
        self.registry = MetricRegistry()
        self.enabled_metrics = enabled_metrics or ["all"]
    
    def _get_active_metrics(self) -> Dict[str, Any]:
        """Get metrics that should be used for scoring."""
        available = self.registry.get_available_metrics()
        
        if "all" in self.enabled_metrics:
            return available
        
        active = {}
        for name in self.enabled_metrics:
            if name in available:
                active[name] = available[name]
            else:
                print(f"ERROR: Metric '{name}' not available")
        
        return active
    
    def calculate_scores(
        self,
        answer: str,
        expected: str,
        keywords: Optional[List[str]] = None,
        question: Optional[str] = None,
        ideal_retrieved_chunks: Optional[List[int]] = None,
        actual_retrieved_chunks: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Calculate scores using active metrics."""
        active_metrics = self._get_active_metrics()
        
        if not active_metrics:
            return {"error": "No metrics available"}
        
        scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        metric_kwargs = {
            "answer": answer,
            "expected": expected,
            "keywords": keywords,
            "question": question,
            "ideal_retrieved_chunks": ideal_retrieved_chunks,
            "actual_retrieved_chunks": actual_retrieved_chunks,
        }

        for name, metric in active_metrics.items():
            score = metric.calculate(**metric_kwargs)
            scores[f"{name}_similarity"] = score
            
            weight = metric.weight
            # Only include metrics with non-zero weight in final score
            if weight > 0:
                total_weighted_score += score * weight
                total_weight += weight
        
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        if keywords:
            answer_lower = answer.lower()
            keywords_matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
        else:
            keywords_matched = 0
        
        return {
            **scores,
            "final_score": final_score,
            "keywords_matched": keywords_matched,
            "active_metrics": list(active_metrics.keys())
        }