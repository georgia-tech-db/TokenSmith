"""
Chunk-level semantic cache inspired by QVCache paper (https://arxiv.org/abs/2602.02057)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from src.embedder import CachedEmbedder
from sklearn.decomposition import PCA


@dataclass
class CacheEntry:
    chunk_id: int
    text: str
    embedding: np.ndarray
    batch_id: int


class SemanticCache:
    def __init__(self, capacity: int = 100, alpha: float = 0.9, deviation: float = 0.25, n_buckets: int = 4, d_reduced: int = 4, pca_min_samples: int = 10, embed_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"):
        self.capacity = capacity
        self.alpha = alpha
        self.deviation = deviation
        self.n_buckets = n_buckets
        self.d_reduced = d_reduced
        self.embed_model = CachedEmbedder(embed_model)

        # Cached chunks — parallel to FAISS index positions
        self._entries: List[CacheEntry] = []
        self._seen_ids: set = set()
        self._index: Optional[faiss.IndexFlatL2] = None
        self._embedding_dim: Optional[int] = None

        # batch_id is for eviction order
        self._batch_order: OrderedDict[int, List[int]] = OrderedDict()
        self._next_batch_id: int = 0

        # PCA for dimensionality reduction for region specific thresholding
        self._thresholds: Dict[Tuple[int, str], float] = {} 
        self.pca: Optional[PCA] = None
        # each dimension has a boundary that is set by the percentiles of the pca reduced data
        self._bucket_boundaries = None
        self._pca_trained: bool = False
        self._pca_training_data: List[np.ndarray] = []
        self._pca_min_samples: int = pca_min_samples

        self.stats = {"hits": 0, "misses": 0}

        self._pca_seed_queries = [
            # acid props
            "What is ACID in databases?",
            "Explain atomicity in database transactions",
            "How does consistency work in ACID?",
            "What does durability guarantee in a database system?",
            "Describe the isolation property of transactions",
            # serializability
            "What is serializability in databases?",
            "Explain conflict serializability",
            "What is a schedule in transaction processing?",
            "How do you test if a schedule is serializable?",
            "What is the difference between serial and serializable schedules?",
        ]        

    def search(self, question: str, k: int = 10):
        """
        does the eager cache search as implemented in the paper
        """
        print(f"[CACHE] search called — index={'None' if self._index is None else self._index.ntotal}, k={k}", flush=True)
        if self._index is None or self._index.ntotal < k:
            return None

        q_vec = self.embed_model.encode([question]).astype("float32")

        print(f"[CACHE] Finding {k} nearest neighbors in cache (ntotal={self._index.ntotal})", flush=True)
        distances, indices = self._index.search(q_vec, k)

        if any(indices[0][i] < 0 for i in range(k)):
            # print(f"[CACHE] Invalid indices found in cache", flush=True)
            return None

        # distance to kth neighbor in cache, this should be within threshold distance
        d_k = float(distances[0][k - 1])

        region = self._compute_region_key(q_vec.flatten())
        threshold = self._thresholds.get((k, region))
        
        print(f"[CACHE] region: {region}, threshold: {threshold}, dist: {d_k}", flush=True)
        print(f"[CACHE] thresholds: {self._thresholds}", flush=True)
        if threshold is None:
            return None

        # if distance is within the threshold according to this formula (from paper) we can return the cached entry
        #  the cached entry is just the vectors (chunks) that we have in the cache which are within threshold distance
        if d_k <= (1.0 + self.deviation) * threshold:
            chunk_texts = [self._entries[int(indices[0][i])].text for i in range(k)]

            # we promote the batches that contained the contributing chunks
            bseen = set()
            for i in range(k):
                batchid = self._entries[int(indices[0][i])].batch_id
                if batchid not in bseen and batchid in self._batch_order:
                    self._batch_order.move_to_end(batchid)
                    bseen.add(batchid)

            self.stats["hits"] += 1
            return chunk_texts

        return None

    def insert(self, query: str, ranked_chunks: List[Tuple[str, float]], chunk_ids: List[int], k: int = 10):
        """insert retrieved chunks and learn thresholds"""
        if not ranked_chunks:
            return

        # ranked_chunks may be list of tuples of chunk text and scorefrom cross encoder rerank 
        chunk_texts = [c[0] if isinstance(c, tuple) else c for c in ranked_chunks]

        q_vec = self.embed_model.encode([query]).astype("float32").flatten()
        chunk_embeddings = self.embed_model.encode(chunk_texts).astype("float32")

        # print(f"chunk_embeddings: {chunk_embeddings.shape}")
        # print(f"q_vec: {q_vec.shape}")

        if self._embedding_dim is None:
            self._embedding_dim = len(chunk_embeddings[0])
            self._index = faiss.IndexFlatL2(self._embedding_dim)

        # threshold learning
        # calculate L2 distances from query to each retrieved chunk
        dists = np.sum((chunk_embeddings - q_vec.reshape(1, -1)) ** 2, axis=1)
        sorted_order = np.argsort(dists)

        region = self._compute_region_key(q_vec)
        nearest_k = min(k + 1, len(ranked_chunks) + 1)
        for ki in range(1, nearest_k):
            d_k_l2 = float(dists[sorted_order[ki - 1]])

            # key is index of the chunk and the region
            key = (ki, region)
            if key in self._thresholds:
                self._thresholds[key] = (
                    (1.0 - self.alpha) * self._thresholds[key]
                    + self.alpha * d_k_l2
                )
            else:
                self._thresholds[key] = d_k_l2

        # gather pca data for training
        if not self._pca_trained:
            self._pca_training_data.append(q_vec.copy())
            for seed_query in self._pca_seed_queries:
                self._pca_training_data.append(self.embed_model.encode([seed_query]).astype("float32").flatten())
            if len(self._pca_training_data) >= self._pca_min_samples:
                self._train_pca()
                print(f"[CACHE] PCA trained with {len(self._pca_training_data)} samples", flush=True)
                print(f"[CACHE] PCA boundaries: {self._bucket_boundaries}", flush=True)

        # filter out chunks already in the cache
        new_items = [(cid, text, emb) 
        for cid, text, emb in zip(chunk_ids, chunk_texts, chunk_embeddings) if cid not in self._seen_ids]
        
        # possible that theres only duplicate chunks
        if not new_items:
            self.stats["misses"] += 1
            return

        # evict oldest batches until there is room
        chunks_to_add = len(chunk_texts)
        while len(self._entries) + chunks_to_add > self.capacity and self._batch_order:
            self._evict_lru_batch()

        # insert chunk embeddings 
        batch_id = self._next_batch_id
        self._next_batch_id += 1

        chunk_indices = []
        for cid, text, emb in new_items:
            idx = len(self._entries)
            self._entries.append(CacheEntry(chunk_id=cid, text=text, embedding=emb, batch_id=batch_id))
            self._seen_ids.add(cid)
            chunk_indices.append(idx)

        self._batch_order[batch_id] = chunk_indices
        self._rebuild_index()

        # print(f"self._entries: {self._entries}")
        self.stats["misses"] += 1

    def get_stats(self):
        total = self.stats["hits"] + self.stats["misses"]
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_ratio": self.stats["hits"] / total if total > 0 else 0.0,
            "cache_size": len(self._entries),
            "capacity": self.capacity,
            "num_batches": len(self._batch_order),
            "active_thresholds": len(self._thresholds),
        }


    def _evict_lru_batch(self):
        """remove the lru batch and rebuild the index"""
        if not self._batch_order:
            return

        _, evicted_indices = self._batch_order.popitem(last=False)
        for idx in sorted(evicted_indices, reverse=True):
            self._seen_ids.remove(self._entries[idx].chunk_id)
            self._entries.pop(idx)
        
        # rebuild the batch order
        new_order: OrderedDict[int, List[int]] = OrderedDict()


        for batch_id in self._batch_order:
            new_order[batch_id] = []

        for i, chunk in enumerate(self._entries):
            if chunk.batch_id in new_order:
                new_order[chunk.batch_id].append(i)
        self._batch_order = new_order

    def _rebuild_index(self):
        if not self._entries:
            self._index = faiss.IndexFlatL2(self._embedding_dim)
            return
        matrix = np.vstack([c.embedding for c in self._entries]).astype("float32")
        self._index = faiss.IndexFlatL2(matrix.shape[1])
        self._index.add(matrix)

    def _compute_region_key(self, embedding: np.ndarray):
        """project embedding via pca, bucket each reduced dimension"""
        # this is a fallback until the pca is trained
        if not self._pca_trained:
            return "global"
        reduced = self.pca.transform(embedding.reshape(1, -1))[0]
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
        print(data.shape)
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
