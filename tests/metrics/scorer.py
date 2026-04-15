from typing import Any, Dict, List, Optional
from tests.metrics.registry import MetricRegistry


class SimilarityScorer:
    """Main scorer that uses selected metrics."""
    
    def __init__(self, enabled_metrics: Optional[List[str]] = None):
        self.registry = MetricRegistry()
        self.enabled_metrics = enabled_metrics or ["all"]
    
    def _get_active_metrics(self) -> Dict[str, Any]:
        """Get metrics that should be used for scoring."""
        if "all" in self.enabled_metrics:
            return self.registry.get_available_metrics()

        active = {}
        for name in self.enabled_metrics:
            metric = self.registry.get_metric(name)
            if metric and metric.is_available():
                active[name] = metric
            else:
                print(f"ERROR: Metric '{name}' not available")

        return active
    
    def calculate_scores(
        self,
        answer: str,
        expected: str,
        keywords: Optional[List[str]] = None,
        question: Optional[str] = None,
        retrieval_gold: Optional[Dict[str, Any]] = None,
        actual_retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculate scores using active metrics."""
        active_metrics = self._get_active_metrics()
        
        if not active_metrics:
            return {"error": "No metrics available"}
        
        scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        answer_metric_names = []
        retrieval_metric_names = []
        
        for name, metric in active_metrics.items():
            if metric.metric_group == "retrieval":
                score = metric.calculate(
                    retrieval_gold=retrieval_gold,
                    actual_retrieved_chunks=actual_retrieved_chunks,
                )
                retrieval_metric_names.append(name)
            else:
                if name in ("llm_judge", "async_llm_judge") and question:
                    score = metric.calculate(answer, question, keywords)
                else:
                    score = metric.calculate(answer, expected, keywords)
                answer_metric_names.append(name)
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
            "active_metrics": list(active_metrics.keys()),
            "answer_metrics": answer_metric_names,
            "retrieval_metrics": retrieval_metric_names,
        }
