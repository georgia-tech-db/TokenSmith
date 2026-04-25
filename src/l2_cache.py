"""
In-memory L2 cache for final generated answers.

Key: normalized query text
Value: answer text and LRU/TTL metadata
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.l1_cache import normalize_query_text


@dataclass
class L2CacheEntry:
    normalized_query_text: str
    answer_text: str
    params_signature: str
    created_at: float
    expires_at: float
    last_access_at: float
    access_count: int


class L2AnswerCache:
    def __init__(self, max_entries: int = 256, ttl_seconds: int = 600):
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._entries: "OrderedDict[str, L2CacheEntry]" = OrderedDict()

    def _now(self) -> float:
        return time.time()

    def _is_expired(self, entry: L2CacheEntry, now: float) -> bool:
        return entry.expires_at <= now

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, v in self._entries.items() if self._is_expired(v, now)]
        for key in expired:
            self._entries.pop(key, None)

    def _evict_lru_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    @staticmethod
    def _params_signature(params: Dict[str, Any]) -> str:
        ordered = sorted(params.items(), key=lambda kv: kv[0])
        return hashlib.sha256(repr(ordered).encode("utf-8")).hexdigest()

    def get(self, query: str, params: Dict[str, Any]) -> Optional[L2CacheEntry]:
        now = self._now()
        self._evict_expired(now)

        key = normalize_query_text(query)
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

    def set(self, query: str, answer_text: str, params: Dict[str, Any]) -> L2CacheEntry:
        now = self._now()
        self._evict_expired(now)

        key = normalize_query_text(query)
        entry = L2CacheEntry(
            normalized_query_text=key,
            answer_text=str(answer_text),
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
