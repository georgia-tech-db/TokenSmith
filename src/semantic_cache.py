"""
semantic_cache.py

simplifications relative to the full QVCache design:
  - no FreshVamana mini-indexes; a single FAISS IndexFlatL2 is used (the
    cache is small enough that flat search is sub-millisecond).
  - eviction is per-entry LRU rather than per-mini-index.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import faiss
import numpy as np
from src.embedder import CachedEmbedder
from sklearn.decomposition import PCA

@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    answer: str
    ranked_chunks: List[str]
    metadata: Dict = field(default_factory=dict)

class SemanticCache:
    def __init__(self, capacity: int = 100, alpha: float = 0.9, deviation: float = 0.25, n_buckets: int = 8, d_reduced: int = 8, pca_min_samples: int = 10, embed_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"):
        self.capacity = capacity
        self.alpha = alpha
        self.deviation = deviation
        self.n_buckets = n_buckets
        self.d_reduced = d_reduced
        self.embed_model = CachedEmbedder(embed_model)

        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()
        self._ids: List[int] = []
        self._index = None
        self._embedding_dim = None
        self._next_id: int = 0

        # each region which is found using pca + the bucketing logic has a threshold
        self._thresholds: Dict[str, float] = {}

        # PCA for dimensionality reduction for region specific thresholding
        self.pca: Optional[PCA] = None
        # each dimension has a boundary that is set by the percentiles of the pca reduced data
        self._bucket_boundaries = None
        self._pca_trained: bool = False
        self._pca_training_data = []
        # minimum number of samples to train the pca
        self._pca_min_samples: int = pca_min_samples

        self.stats = {"hits": 0, "misses": 0}

    def search(self, question: str):
        """Cache lookup implementation (used eager strategy from paper), returns first cache entry on hit. """
        if self._index is None or self._index.ntotal == 0:
            return None

        q_vec = self.embed_model.encode(question).reshape(1, -1).astype("float32")

        # search against every query we have seen so far
        distances, indices = self._index.search(q_vec, 1)

        if indices[0][0] == -1:
            return None

        dist = float(distances[0][0])
        faiss_pos = int(indices[0][0])

        # figure out the region of the query, then get the threshold for that region
        region = self._compute_region_key(q_vec.flatten())
        threshold = self._thresholds.get(region)

        print(f"region: {region}, threshold: {threshold}, dist: {dist}")
        if threshold is None:
            return None

        # if distance is within the threshold according to this formula (from paper) we can return the cached entry
        if dist <= (1.0 + self.deviation) * threshold:
            entry_id = self._ids[faiss_pos]

            self._entries.move_to_end(entry_id)
            self.stats["hits"] += 1
            
            return self._entries[entry_id]

        return None

    def insert(self, query: str, answer: str, ranked_chunks: List[str], metadata: Optional[Dict] = None):
        """Insert a new result into the cache and update thresholds."""
        embedding = self.embed_model.encode(query).reshape(1, -1).astype("float32").flatten()

        if self._embedding_dim is None:
            self._embedding_dim = len(embedding)
            # initialize the index 
            self._index = faiss.IndexFlatL2(self._embedding_dim)

        # threshold learning
        if self._index.ntotal > 0:
            distances, _ = self._index.search(embedding.reshape(1, -1), 1)
            d_nearest = float(distances[0][0])

            region = self._compute_region_key(embedding)
            if region in self._thresholds:
                self._thresholds[region] = (1.0 - self.alpha) * self._thresholds[region] + self.alpha * d_nearest        
            else:
                self._thresholds[region] = d_nearest
        else:
            region = self._compute_region_key(embedding)
            self._thresholds.setdefault(region, 0.0)

        # gather data to train the PCA
        if not self._pca_trained:
            self._pca_training_data.append(embedding.copy())
            if len(self._pca_training_data) >= self._pca_min_samples:
                self._train_pca()

        # evict if at capacity
        while len(self._entries) >= self.capacity:
            evicted_id, _ = self._entries.popitem(last=False)
            idx = self._ids.index(evicted_id)
            self._ids.pop(idx)

        # insert the new entry
        entry = CacheEntry(query=query, embedding=embedding, answer=answer, ranked_chunks=ranked_chunks, metadata=metadata or {})
        self._entries[self._next_id] = entry
        self._ids.append(self._next_id)
        self._next_id += 1

        self._rebuild_index()
        print(f'self._ids: {self._ids}')
        print(f'self._entries: {self._entries}')
        self.stats["misses"] += 1

    def get_stats(self):
        total = self.stats["hits"] + self.stats["misses"]
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_ratio": self.stats["hits"] / total if total > 0 else 0.0,
            "cache_size": len(self._entries),
            "capacity": self.capacity,
            "active_regions": len(self._thresholds),
        }

    def _rebuild_index(self):
        # if not self._entries:
        #     self._index = faiss.IndexFlatL2(self._embedding_dim or 1)
        #     return
        embeddings = np.vstack([e.embedding for e in self._entries.values()]).astype("float32")
        self._index = faiss.IndexFlatL2(embeddings.shape[1])
        self._index.add(embeddings)

    def _compute_region_key(self, embedding: np.ndarray):
        """project embedding via pca, bucket each reduced dimension"""
        # this is a fallback until the pca is trained
        if not self._pca_trained:
            return "global"

        reduced = self.pca.transform(embedding.reshape(1, -1))

        buckets = []
        for dim in range(self.d_reduced):
            bounds = self._bucket_boundaries[dim]
            bucket = int(np.searchsorted(bounds, reduced[dim]))
            if bucket >= self.n_buckets:
                buckets.append(str(self.n_buckets - 1))
            else:
                buckets.append(str(bucket))
        
        return ",".join(buckets)

    def _train_pca(self):
        data = np.array(self._pca_training_data, dtype="float32")
        n_components = min(self.d_reduced, data.shape[0], data.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(data)
        reduced = self.pca.transform(data)

        self._bucket_boundaries = []
        
        percentiles = np.linspace(0, 100, self.n_buckets + 1)[1:-1]
        for dim in range(n_components):
            bounds = np.percentile(reduced[:, dim], percentiles)
            self._bucket_boundaries.append(bounds)

        # pad just in case n_components < d_reduced
        while len(self._bucket_boundaries) < self.d_reduced:
            self._bucket_boundaries.append(np.zeros(len(percentiles)))
        self._bucket_boundaries = np.array(self._bucket_boundaries)

        self.d_reduced = n_components
        self._pca_trained = True
        self._pca_training_data.clear() # we don't need this anymore