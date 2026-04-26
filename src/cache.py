import argparse
import json
import hashlib
from typing import Dict, Optional, Any, List, Deque
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import CrossEncoder

from src.embedder import SentenceTransformer
from src.config import RAGConfig
from src.retriever import BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts, filter_retrieved_chunks


class BaseResponseCache(ABC):
    @abstractmethod
    def lookup(self, config_key: str, query_embedding: np.ndarray, current_question: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def store(self, config_key: str, normalized_question: str, question_embedding: Optional[np.ndarray], payload: Dict[str, Any]) -> None:
        pass
        
    @abstractmethod
    def clear(self) -> None:
        pass
        
    @abstractmethod
    def make_config_key(self, cfg: RAGConfig, args: argparse.Namespace, golden_chunks: Optional[List[str]]) -> str:
        pass
        
    @abstractmethod
    def compute_embedding(self, question: str, retrievers: List[Any], embed_model: str) -> Optional[np.ndarray]:
        pass
        
    @abstractmethod
    def normalize_question(self, q: str) -> str:
        pass


class SemanticCache(BaseResponseCache):
    def __init__(self, bi_encoder_threshold: float, cross_encoder_threshold: float, max_entries: int = 50):
        self.cache: Dict[str, Deque[Dict[str, Any]]] = {}
        self.bi_encoder_threshold = bi_encoder_threshold
        self.cross_encoder_threshold = cross_encoder_threshold
        self.max_entries = max_entries
        self.question_embedders: Dict[str, SentenceTransformer] = {}
        self.cross_encoder_model: Optional[CrossEncoder] = None

    def _get_cross_encoder(self) -> CrossEncoder:
        """Return a global cross-encoder model instance, initializing if needed."""
        if self.cross_encoder_model is None:
            self.cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return self.cross_encoder_model

    def normalize_question(self, q: str) -> str:
        """Normalize a question string: lowercase, strip, and collapse spaces."""
        return " ".join((q or "").strip().lower().split())

    def make_config_key(self, cfg: RAGConfig, args: argparse.Namespace, golden_chunks: Optional[List[str]]) -> str:
        """
        Create a unique JSON key for semantic cache based on config, arguments, and optional golden chunks.
        """
        try: 
            payload = cfg.get_config_state()
            payload["system_prompt_mode"] = getattr(args, "system_prompt_mode", None) or cfg.system_prompt_mode
            payload["index_prefix"] = getattr(args, "index_prefix", None)
        except Exception:
            payload = {
                "gen_model": getattr(args, "model_path", None) or cfg.gen_model,
                "embed_model": cfg.embed_model,
                "top_k": cfg.top_k,
                "system_prompt_mode": getattr(args, "system_prompt_mode", None) or cfg.system_prompt_mode,
                "ensemble_method": cfg.ensemble_method,
                "ranker_weights": cfg.ranker_weights,
                "use_hyde": cfg.use_hyde,
                "use_indexed_chunks": cfg.use_indexed_chunks,
                "disable_chunks": cfg.disable_chunks,
                "use_golden_chunks": bool(golden_chunks and cfg.use_golden_chunks),
                "index_prefix": getattr(args, "index_prefix", None),
            }

        if golden_chunks and cfg.use_golden_chunks:
            signature = hashlib.sha256("||".join(golden_chunks).encode("utf-8")).hexdigest()
            payload["golden_signature"] = signature

        return json.dumps(payload, sort_keys=True)

    def lookup(self, config_key: str, query_embedding: np.ndarray, current_question: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached answer if semantically similar to the current question.
        """
        entries = self.cache.get(config_key, [])
        if not entries or query_embedding is None:
            return None

        # Step 1: Bi-Encoder filter (fast cosine similarity)
        candidates = [
            entry for entry in entries
            if np.dot(entry["embedding"], query_embedding) > self.bi_encoder_threshold
        ]
        if not candidates:
            return None

        # Step 2: Cross-Encoder verification
        ce_model = self._get_cross_encoder()
        pairs = [[current_question, c["question"]] for c in candidates]
        ce_scores = ce_model.predict(pairs, show_progress_bar=False)
        best_idx = int(np.argmax(ce_scores))

        if ce_scores[best_idx] > self.cross_encoder_threshold:
            return candidates[best_idx]["payload"]
        return None

    def store(self, config_key: str, normalized_question: str, question_embedding: Optional[np.ndarray], payload: Dict[str, Any]) -> None:
        """
        Store a question, its embedding, and the generated answer in the semantic cache.
        Evict oldest entries if cache exceeds self.max_entries.
        """
        if question_embedding is None:
            return

        if config_key not in self.cache:
            self.cache[config_key] = deque()
        entries = self.cache[config_key]
        entries.append({
            "question": normalized_question,
            "embedding": question_embedding.astype(np.float32),
            "payload": payload,
        })

        if len(entries) > self.max_entries:
            entries.popleft()
            
    def clear(self) -> None:
        self.cache.clear()

    def _get_question_embedder(self, retrievers: List[Any], embed_model: str) -> Optional[SentenceTransformer]:
        """
        Get or initialize a SentenceTransformer for encoding questions.
        Prefers the embedder from any FAISSRetriever in the retrievers list.
        """
        for retriever in retrievers:
            if isinstance(retriever, FAISSRetriever):
                return retriever._ensure_embedder()

        model_path = embed_model
        if not model_path:
            return None

        embedder = self.question_embedders.get(model_path)
        if embedder is None:
            embedder = SentenceTransformer(model_path)
            self.question_embedders[model_path] = embedder

        return embedder

    def compute_embedding(self, question: str, retrievers: List[Any], embed_model: str) -> Optional[np.ndarray]:
        """
        Compute a normalized embedding vector for a question using the configured embedder.
        """
        embedder = self._get_question_embedder(retrievers, embed_model)
        if not embedder:
            return None

        vec = embedder.encode([question], batch_size=1, normalize=True, show_progress_bar=False)
        if vec.size == 0:
            return None

        return vec[0]


class NoOpCache(BaseResponseCache):
    def lookup(self, config_key: str, query_embedding: np.ndarray, current_question: str) -> Optional[Dict[str, Any]]:
        return None

    def store(self, config_key: str, normalized_question: str, question_embedding: Optional[np.ndarray], payload: Dict[str, Any]) -> None:
        pass
        
    def clear(self) -> None:
        pass
        
    def make_config_key(self, cfg: RAGConfig, args: argparse.Namespace, golden_chunks: Optional[List[str]]) -> str:
        return ""
        
    def compute_embedding(self, question: str, retrievers: List[Any], embed_model: str) -> Optional[np.ndarray]:
        return None
        
    def normalize_question(self, q: str) -> str:
        return ""


_GLOBAL_SEMANTIC_CACHE: Optional[SemanticCache] = None

def get_cache(cfg: RAGConfig) -> BaseResponseCache:
    """Return a configured cache layer, either SemanticCache or NoOpCache depending on config."""
    global _GLOBAL_SEMANTIC_CACHE
    if getattr(cfg, 'semantic_cache_enabled', False):
        if _GLOBAL_SEMANTIC_CACHE is None:
            _GLOBAL_SEMANTIC_CACHE = SemanticCache(
                bi_encoder_threshold=cfg.semantic_cache_bi_encoder_threshold,
                cross_encoder_threshold=cfg.semantic_cache_cross_encoder_threshold
            )
        return _GLOBAL_SEMANTIC_CACHE
    return NoOpCache()
