from typing import Dict, List, Optional
from tests.metrics.base import MetricBase


class MetricRegistry:
    """Registry for managing available metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, MetricBase] = {}
        self._auto_register()
    
    def _auto_register(self):
        """Automatically register all available metrics."""
        from tests.metrics import (
            SemanticSimilarityMetric,
            KeywordMatchMetric,
            NLIEntailmentMetric,
            AsyncLLMJudgeMetric
        )

        self.register(SemanticSimilarityMetric())
        self.register(KeywordMatchMetric())
        self.register(NLIEntailmentMetric())
        self.register(AsyncLLMJudgeMetric())

    def register(self, metric: MetricBase):
        """Register a new metric."""
        self._metrics[metric.name] = metric
        print(f"Registered metric: {metric}")
    
    def get_metric(self, name: str) -> Optional[MetricBase]:
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def get_available_metrics(self) -> Dict[str, MetricBase]:
        """Get all available metrics that can be used."""
        return {name: metric for name, metric in self._metrics.items() 
                if metric.is_available()}
    
    def get_all_metrics(self) -> Dict[str, MetricBase]:
        """Get all registered metrics (including unavailable ones)."""
        return self._metrics.copy()
    
    def list_metric_names(self) -> List[str]:
        """List all available metric names."""
        return list(self.get_available_metrics().keys())
    
    def list_all_metric_names(self) -> List[str]:
        """List all registered metric names (including unavailable)."""
        return list(self._metrics.keys())
