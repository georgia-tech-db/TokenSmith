from collections.abc import Callable
from typing import Dict, List, Optional
from tests.metrics.base import MetricBase


class MetricRegistry:
    """Registry for managing available metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, MetricBase] = {}
        self._factories: Dict[str, Callable[[], MetricBase]] = {}
        self._auto_register()
    
    def _auto_register(self):
        """Register metric factories without instantiating heavyweight models eagerly."""
        from tests.metrics import (
            SemanticSimilarityMetric,
            KeywordMatchMetric,
            NLIEntailmentMetric,
            AsyncLLMJudgeMetric,
            ChunkMAPAt10Metric,
            ChunkMRRAt10Metric,
            ChunkNDCGAt10Metric,
            ChunkRecallAt5Metric,
            ChunkRecallAt10Metric,
            DirectPageHitAt10Metric,
            PageHitAt5Metric,
            PageHitAt10Metric,
        )

        self.register_factory("semantic", SemanticSimilarityMetric)
        self.register_factory("keyword_match", KeywordMatchMetric)
        self.register_factory("nli", NLIEntailmentMetric)
        self.register_factory("async_llm_judge", AsyncLLMJudgeMetric)
        self.register_factory("chunk_ndcg_10", ChunkNDCGAt10Metric)
        self.register_factory("chunk_recall_5", ChunkRecallAt5Metric)
        self.register_factory("chunk_recall_10", ChunkRecallAt10Metric)
        self.register_factory("chunk_mrr_10", ChunkMRRAt10Metric)
        self.register_factory("chunk_map_10", ChunkMAPAt10Metric)
        self.register_factory("page_hit_5", PageHitAt5Metric)
        self.register_factory("page_hit_10", PageHitAt10Metric)
        self.register_factory("direct_page_hit_10", DirectPageHitAt10Metric)

    def register(self, metric: MetricBase):
        """Register a new metric."""
        self._metrics[metric.name] = metric

    def register_factory(self, name: str, factory: Callable[[], MetricBase]):
        """Register a lazy metric factory by name."""
        self._factories[name] = factory

    def _ensure_metric(self, name: str) -> Optional[MetricBase]:
        if name not in self._metrics and name in self._factories:
            self._metrics[name] = self._factories[name]()
        return self._metrics.get(name)
    
    def get_metric(self, name: str) -> Optional[MetricBase]:
        """Get a metric by name."""
        return self._ensure_metric(name)
    
    def get_available_metrics(self) -> Dict[str, MetricBase]:
        """Get all available metrics that can be used."""
        available = {}
        for name in self.list_all_metric_names():
            metric = self._ensure_metric(name)
            if metric and metric.is_available():
                available[name] = metric
        return available
    
    def get_all_metrics(self) -> Dict[str, MetricBase]:
        """Get all registered metrics (including unavailable ones)."""
        return {name: self._ensure_metric(name) for name in self.list_all_metric_names()}
    
    def list_metric_names(self) -> List[str]:
        """List all available metric names."""
        return list(self.get_available_metrics().keys())
    
    def list_all_metric_names(self) -> List[str]:
        """List all registered metric names (including unavailable)."""
        return list(dict.fromkeys([*self._factories.keys(), *self._metrics.keys()]))
