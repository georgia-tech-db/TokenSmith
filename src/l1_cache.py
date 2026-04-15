"""
In-memory L1 cache for retrieval + ranking outputs.

Key: query embedding hash (from normalized query text)
Value: normalized query text, top chunk ids, top chunk scores, and LRU/TTL metadata
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.embedder import CachedEmbedder


_EMBEDDER_CACHE: Dict[Tuple[str, int], CachedEmbedder] = {}


def _get_embedder(model_path: str, context_window: int) -> CachedEmbedder:
    key = (model_path, int(context_window))
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = CachedEmbedder(model_path, n_ctx=int(context_window))
    return _EMBEDDER_CACHE[key]


def normalize_query_text(query: str) -> str:
    """Normalize query text for stable embedding hash keys."""
    return re.sub(r"\s+", " ", query.strip().lower())


@dataclass
class L1CacheEntry:
    normalized_query_text: str
    top_chunk_ids: List[int]
    top_chunk_scores: List[float]
    params_signature: str
    created_at: float
    expires_at: float
    last_access_at: float
    access_count: int


class L1RetrievalCache:
    def __init__(self, max_entries: int = 256, ttl_seconds: int = 600):
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._entries: "OrderedDict[str, L1CacheEntry]" = OrderedDict()

    def _now(self) -> float:
        return time.time()

    def _is_expired(self, entry: L1CacheEntry, now: float) -> bool:
        return entry.expires_at <= now

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, v in self._entries.items() if self._is_expired(v, now)]
        for key in expired:
            self._entries.pop(key, None)

    def _evict_lru_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def _params_signature(params: Dict[str, Any]) -> str:
        ordered = sorted(params.items(), key=lambda kv: kv[0])
        return hashlib.sha256(repr(ordered).encode("utf-8")).hexdigest()

    def _make_query_embedding_hash(
        self,
        normalized_query_text: str,
        embed_model: str,
        embedding_context_window: int,
    ) -> str:
        embedder = _get_embedder(embed_model, embedding_context_window)
        vec = embedder.encode([normalized_query_text], normalize=True).astype(np.float32)
        return hashlib.sha256(vec.tobytes()).hexdigest()

    def get(
        self,
        query: str,
        embed_model: str,
        embedding_context_window: int,
        params: Dict[str, Any],
    ) -> Optional[L1CacheEntry]:
        now = self._now()
        self._evict_expired(now)

        normalized_query_text = normalize_query_text(query)
        key = self._make_query_embedding_hash(
            normalized_query_text,
            embed_model,
            embedding_context_window,
        )

        entry = self._entries.get(key)
        if entry is None:
            return None

        expected_sig = self._params_signature(params)
        if entry.params_signature != expected_sig:
            self._entries.pop(key, None)
            return None

        if self._is_expired(entry, now):
            self._entries.pop(key, None)
            return None

        entry.last_access_at = now
        entry.access_count += 1
        self._entries.move_to_end(key)
        return entry

    def set(
        self,
        query: str,
        embed_model: str,
        embedding_context_window: int,
        top_chunk_ids: List[int],
        top_chunk_scores: List[float],
        params: Dict[str, Any],
    ) -> L1CacheEntry:
        now = self._now()
        self._evict_expired(now)

        normalized_query_text = normalize_query_text(query)
        key = self._make_query_embedding_hash(
            normalized_query_text,
            embed_model,
            embedding_context_window,
        )

        entry = L1CacheEntry(
            normalized_query_text=normalized_query_text,
            top_chunk_ids=[int(i) for i in top_chunk_ids],
            top_chunk_scores=[float(s) for s in top_chunk_scores],
            params_signature=self._params_signature(params),
            created_at=now,
            expires_at=now + self.ttl_seconds,
            last_access_at=now,
            access_count=1,
        )

        if key in self._entries:
            self._entries.pop(key, None)
        self._entries[key] = entry
        self._evict_lru_if_needed()
        return entry

    def stats(self) -> Dict[str, int]:
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }
