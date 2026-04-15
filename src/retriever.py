"""
Core retrieval logic and artifact loading.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

from src.artifacts import (
    ArtifactBundle,
    ArtifactValidationError,
    artifact_file_map,
    load_manifest,
    sha256_file,
    validate_bundle,
)
from src.config import RAGConfig
from src.embedder import CachedEmbedder
from src.index_builder import preprocess_for_bm25

_EMBED_CACHE: Dict[str, CachedEmbedder] = {}


def _get_embedder(model_name: str) -> CachedEmbedder:
    """Return a cached CachedEmbedder instance, creating one if needed."""
    if model_name not in _EMBED_CACHE:
        _EMBED_CACHE[model_name] = CachedEmbedder(model_name)
    return _EMBED_CACHE[model_name]


def _load_pickle(path: pathlib.Path) -> Any:
    """Deserialize and return an object from a pickle file."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_optional_pickle(path: pathlib.Path) -> Any:
    """Load a pickle file if it exists, otherwise return None."""
    if not path.exists():
        return None
    return _load_pickle(path)


def _load_optional_array(path: pathlib.Path) -> Optional[np.ndarray]:
    """Load a NumPy array from a .npy file if it exists, otherwise return None."""
    if not path.exists():
        return None
    return np.load(path)


def _load_optional_faiss(path: pathlib.Path) -> Optional[faiss.Index]:
    """Load a FAISS index from disk if the file exists, otherwise return None."""
    if not path.exists():
        return None
    return faiss.read_index(str(path))


def _load_page_to_chunk_map(path: pathlib.Path) -> Dict[int, List[int]]:
    """Load a page-to-chunk-id mapping from a JSON file, converting keys to ints."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw_map = json.load(handle)
    return {int(page): [int(chunk_id) for chunk_id in chunk_ids] for page, chunk_ids in raw_map.items()}


def load_artifact_bundle(artifacts_dir: os.PathLike, index_prefix: str) -> ArtifactBundle:
    """Load all index artifacts from disk into an ArtifactBundle.

    Reads required chunk-level artifacts and optional section-level artifacts,
    then validates the bundle against its manifest (if present), including
    source-document hash verification.

    Args:
        artifacts_dir: Directory containing the persisted artifact files.
        index_prefix: Prefix used to locate artifact files within the directory.

    Returns:
        A fully populated and validated ArtifactBundle.

    Raises:
        ArtifactValidationError: If the manifest references missing or
            hash-mismatched files, or if the bundle fails consistency checks.
    """
    artifacts_path = pathlib.Path(artifacts_dir)
    file_map = artifact_file_map(index_prefix)
    manifest = load_manifest(artifacts_path, index_prefix)

    chunk_index = faiss.read_index(str(artifacts_path / file_map["chunk_index"]))
    chunk_bm25 = _load_pickle(artifacts_path / file_map["chunk_bm25"])
    chunks = _load_pickle(artifacts_path / file_map["chunks"])
    sources = _load_pickle(artifacts_path / file_map["sources"])
    metadata = _load_pickle(artifacts_path / file_map["metadata"])
    page_to_chunk_map = _load_page_to_chunk_map(artifacts_path / file_map["page_to_chunk_map"])

    bundle = ArtifactBundle(
        chunk_index=chunk_index,
        chunk_bm25=chunk_bm25,
        chunks=chunks,
        sources=sources,
        metadata=metadata,
        page_to_chunk_map=page_to_chunk_map,
        chunk_embeddings=_load_optional_array(artifacts_path / file_map["chunk_embeddings"]),
        section_index=_load_optional_faiss(artifacts_path / file_map["section_index"]),
        section_bm25=_load_optional_pickle(artifacts_path / file_map["section_bm25"]),
        sections=_load_optional_pickle(artifacts_path / file_map["sections"]) or [],
        section_sources=_load_optional_pickle(artifacts_path / file_map["section_sources"]) or [],
        section_meta=_load_optional_pickle(artifacts_path / file_map["section_meta"]) or [],
        section_embeddings=_load_optional_array(artifacts_path / file_map["section_embeddings"]),
        manifest=manifest,
    )

    if manifest is not None:
        source_document = manifest.get("source_document")
        source_sha256 = manifest.get("source_document_sha256")
        if source_document:
            source_path = pathlib.Path(source_document)
            if not source_path.exists():
                raise ArtifactValidationError(f"Artifact manifest references missing source document: {source_path}")
            if source_sha256 and sha256_file(source_path) != source_sha256:
                raise ArtifactValidationError(
                    f"Artifact source document hash mismatch for {source_path}. "
                    "Rebuild the index artifacts before running retrieval."
                )

        for _label, relative_path in manifest.get("files", {}).items():
            file_path = artifacts_path / relative_path
            if not file_path.exists():
                raise ArtifactValidationError(f"Artifact manifest references missing file: {file_path}")
        validate_bundle(bundle)

    return bundle


def load_artifacts(artifacts_dir: os.PathLike, index_prefix: str) -> Tuple[faiss.Index, List[str], List[str], Any]:
    """
    Backwards-compatible loader for legacy callers.
    """
    bundle = load_artifact_bundle(artifacts_dir, index_prefix)
    return bundle.chunk_index, bundle.chunk_bm25, bundle.chunks, bundle.sources, bundle.metadata


def get_page_numbers(chunk_indices: Sequence[int], metadata: Sequence[dict]) -> Dict[int, List[int]]:
    """Map each chunk index to its list of source page numbers from metadata."""
    if not metadata or not chunk_indices:
        return {}

    page_map: Dict[int, List[int]] = {}
    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)
        if 0 <= chunk_idx < len(metadata):
            pages = metadata[chunk_idx].get("page_numbers")
            if pages is None and "page_number" in metadata[chunk_idx]:
                pages = [int(metadata[chunk_idx]["page_number"])]
            if pages is None:
                continue
            page_map[chunk_idx] = [int(page) for page in pages]
    return page_map


def filter_retrieved_chunks(cfg: RAGConfig, chunks: Sequence[str], ordered: Sequence[int]) -> List[int]:
    """Return the top-k chunk ids from an ordered ranking based on the RAG config."""
    return [int(chunk_id) for chunk_id in ordered[: cfg.top_k]]


class Retriever(ABC):
    """Abstract base class for retrieval strategies (FAISS, BM25, keyword, etc.)."""

    @abstractmethod
    def get_scores(
        self,
        query: str,
        pool_size: int,
        texts: Sequence[str],
        candidate_ids: Optional[Iterable[int]] = None,
    ) -> Dict[int, float]:
        """Retrieve the top pool_size scores for a query."""


class FAISSRetriever(Retriever):
    """Retriever backed by a FAISS vector index for dense similarity search."""

    name = "faiss"

    def __init__(self, index: faiss.Index, embed_model: str, embeddings: Optional[np.ndarray] = None):
        self.index = index
        self.embed_model = embed_model
        self.embedder: Optional[CachedEmbedder] = None
        self._embedder_failed = False
        self.embeddings = embeddings

    def _ensure_embedder(self) -> Optional[CachedEmbedder]:
        """Lazily load the embedder, marking it unavailable after the first hard failure."""
        if self._embedder_failed:
            return None
        if self.embedder is None:
            try:
                self.embedder = _get_embedder(self.embed_model)
            except Exception as exc:
                print(f"WARNING: FAISS retriever unavailable for {self.embed_model}: {exc}")
                self._embedder_failed = True
                return None
        return self.embedder

    def _search_global(self, query_vector: np.ndarray, pool_size: int, texts: Sequence[str]) -> Dict[int, float]:
        """Search the full FAISS index and return scores as 1/(1+distance)."""
        distances, indices = self.index.search(query_vector, pool_size)
        scores: Dict[int, float] = {}
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(texts):
                scores[int(idx)] = 1.0 / (1.0 + float(dist))
        return scores

    def _search_candidates(self, query_vector: np.ndarray, candidate_ids: Sequence[int], pool_size: int) -> Dict[int, float]:
        """Score only the given candidate chunk ids using preloaded embeddings."""
        if self.embeddings is None:
            return {}

        unique_candidate_ids = np.array(
            sorted({int(idx) for idx in candidate_ids}),
            dtype=np.int32,
        )
        if unique_candidate_ids.size == 0:
            return {}

        candidate_vectors = self.embeddings[unique_candidate_ids]
        distances = np.sum((candidate_vectors - query_vector[0]) ** 2, axis=1)
        limit = min(pool_size, len(unique_candidate_ids))
        top_positions = np.argpartition(distances, limit - 1)[:limit] if limit < len(unique_candidate_ids) else np.arange(len(unique_candidate_ids))
        ranked_positions = top_positions[np.argsort(distances[top_positions])]

        return {
            int(unique_candidate_ids[position]): 1.0 / (1.0 + float(distances[position]))
            for position in ranked_positions
        }

    def get_scores(
        self,
        query: str,
        pool_size: int,
        texts: Sequence[str],
        candidate_ids: Optional[Iterable[int]] = None,
    ) -> Dict[int, float]:
        """Encode the query and return FAISS similarity scores for the top chunks.

        Falls back to global search filtered by candidate_ids when
        preloaded embeddings are unavailable.
        """
        embedder = self._ensure_embedder()
        if embedder is None:
            return {}

        try:
            query_vector = embedder.encode([query]).astype("float32")
        except Exception as exc:
            print(f"WARNING: FAISS query embedding failed for {self.embed_model}: {exc}")
            self._embedder_failed = True
            self.embedder = None
            return {}
        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: index={self.index.d} query={query_vector.shape[1]}"
            )

        if candidate_ids is not None:
            candidate_list = list(candidate_ids)
            if self.embeddings is not None:
                return self._search_candidates(query_vector, candidate_list, pool_size)

            global_scores = self._search_global(query_vector, len(texts), texts)
            candidate_set = {int(idx) for idx in candidate_list}
            filtered = {idx: score for idx, score in global_scores.items() if idx in candidate_set}
            return dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:pool_size])

        return self._search_global(query_vector, pool_size, texts)


class BM25Retriever(Retriever):
    """Retriever using BM25 term-frequency scoring for sparse keyword matching."""

    name = "bm25"

    def __init__(self, index: Any):
        self.index = index

    def get_scores(
        self,
        query: str,
        pool_size: int,
        texts: Sequence[str],
        candidate_ids: Optional[Iterable[int]] = None,
    ) -> Dict[int, float]:
        """Score chunks using BM25, optionally restricted to candidate_ids."""
        tokenized_query = preprocess_for_bm25(query)
        all_scores = np.asarray(self.index.get_scores(tokenized_query))

        if candidate_ids is not None:
            candidate_array = np.array(
                sorted({int(idx) for idx in candidate_ids}),
                dtype=np.int32,
            )
            if candidate_array.size == 0:
                return {}
            candidate_scores = all_scores[candidate_array]
            limit = min(pool_size, len(candidate_array))
            top_positions = np.argpartition(-candidate_scores, limit - 1)[:limit] if limit < len(candidate_array) else np.arange(len(candidate_array))
            ranked_positions = top_positions[np.argsort(-candidate_scores[top_positions])]
            return {
                int(candidate_array[position]): float(candidate_scores[position])
                for position in ranked_positions
            }

        num_candidates = min(pool_size, len(all_scores))
        top_indices = np.argpartition(-all_scores, kth=num_candidates - 1)[:num_candidates]
        ranked_indices = top_indices[np.argsort(-all_scores[top_indices])]
        return {int(idx): float(all_scores[idx]) for idx in ranked_indices if 0 <= idx < len(texts)}


class IndexKeywordRetriever(Retriever):
    """Retriever that matches query keywords against a pre-extracted book index."""

    name = "index_keywords"

    def __init__(self, extracted_index_path: os.PathLike, page_to_chunk_map_path: os.PathLike):
        self.page_to_chunk_map: Dict[str, List[int]] = {}
        self._lemmatizer = self._build_lemmatizer()

        if os.path.exists(extracted_index_path):
            with open(extracted_index_path, encoding="utf-8") as handle:
                raw_index = json.load(handle)
            self.phrase_to_pages: Dict[str, List[int]] = {}
            self.token_to_phrases: Dict[str, List[str]] = {}

            for phrase, pages in raw_index.items():
                normalized_tokens = []
                for word in phrase.lower().split():
                    cleaned = word.strip('.,!?()[]:"\'')
                    if cleaned:
                        normalized_tokens.append(self._lemmatize_word(cleaned, self._lemmatizer))

                normalized_phrase = " ".join(normalized_tokens)
                self.phrase_to_pages[normalized_phrase] = pages
                for token in normalized_tokens:
                    self.token_to_phrases.setdefault(token, []).append(normalized_phrase)
        else:
            self.phrase_to_pages = {}
            self.token_to_phrases = {}

        if os.path.exists(page_to_chunk_map_path):
            with open(page_to_chunk_map_path, encoding="utf-8") as handle:
                self.page_to_chunk_map = json.load(handle)

    def get_scores(
        self,
        query: str,
        pool_size: int,
        texts: Sequence[str],
        candidate_ids: Optional[Iterable[int]] = None,
    ) -> Dict[int, float]:
        """Score chunks by counting keyword hits via the book index, normalized to [0, 1]."""
        keywords = self._extract_keywords(query)
        candidate_set = None if candidate_ids is None else {int(idx) for idx in candidate_ids}
        chunk_hit_counts: Dict[int, int] = {}

        for keyword in keywords:
            for phrase in self.token_to_phrases.get(keyword, []):
                for page_no in self.phrase_to_pages[phrase]:
                    for chunk_id in self.page_to_chunk_map.get(str(page_no), []):
                        chunk_id = int(chunk_id)
                        if chunk_id < 0 or chunk_id >= len(texts):
                            continue
                        if candidate_set is not None and chunk_id not in candidate_set:
                            continue
                        chunk_hit_counts[chunk_id] = chunk_hit_counts.get(chunk_id, 0) + 1

        if not chunk_hit_counts:
            return {}

        max_hits = max(chunk_hit_counts.values())
        ranked = sorted(
            chunk_hit_counts.items(),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )[:pool_size]
        return {chunk_id: float(hit_count) / max_hits for chunk_id, hit_count in ranked}

    @staticmethod
    def _build_lemmatizer() -> Optional[WordNetLemmatizer]:
        """Create a lemmatizer when the WordNet corpus is available, else return None."""
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            return None
        return WordNetLemmatizer()

    @staticmethod
    def _lemmatize_word(word: str, lemmatizer: Optional[WordNetLemmatizer]) -> str:
        """Lemmatize a word, trying noun then verb POS tags."""
        if lemmatizer is None:
            return word
        lemma = lemmatizer.lemmatize(word, pos="n")
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, pos="v")
        return lemma

    def _extract_keywords(self, query: str) -> List[str]:
        """Tokenize and lemmatize a query string into normalized keywords."""
        return [
            self._lemmatize_word(token, self._lemmatizer)
            for token in preprocess_for_bm25(query)
        ]
