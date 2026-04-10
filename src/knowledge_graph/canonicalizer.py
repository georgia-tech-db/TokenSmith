import json
import logging
from collections import Counter
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from src.knowledge_graph.models import ExtractionResult, CanonicalizationResult
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.utils.normalizer import Normalizer
from src.knowledge_graph.utils.prompts import SYNONYM_PROMPT, SYNONYM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Canonicalizer:
    """Semantic canonicalization of KG keywords.

    Args:
        corpus_description: Human-readable description of the corpus
            (e.g. Title of the textbook or main topic of the document).
            Injected into the LLM system prompt as domain context.
        api_key: OpenRouter API key for the LLM verification step.
        embedding_model: Sentence-transformer model name for keyword embedding.
        similarity_threshold: Cosine similarity threshold for complete-linkage
            clustering. A group forms only when ALL pairs in it exceed this value.
        max_group_size: Maximum keywords per LLM call. Oversized clusters are
            force-split into fixed-size chunks before the LLM step.
        llm_model: OpenRouter model identifier.
        batch_size: Number of small groups (≤5 keywords) to batch per LLM call.
        fallback_threshold: Cosine similarity threshold used at query time when a
            keyword is not in the synonym table (embedding-based fallback).
    """

    def __init__(
        self,
        corpus_description: str,
        api_key: str,
        embedding_model: str,
        similarity_threshold: float = 0.78,
        max_group_size: int = 30,
        llm_model: str = "openai/gpt-4o-mini",
        batch_size: int = 15,
        fallback_threshold: float = 0.85,
        retries: int = 1,
        normalizer: Normalizer | None = None,
    ):
        self.corpus_description = corpus_description
        self.similarity_threshold = similarity_threshold
        self.max_group_size = max_group_size
        self.llm_model = llm_model
        self.batch_size = batch_size
        self.fallback_threshold = fallback_threshold
        self._normalizer = normalizer or Normalizer()
        self.retries = retries
        self._client = OpenRouterClient(api_key, retries=retries)

        logger.info("Loading embedding model: %s", embedding_model)
        self._model = SentenceTransformer(embedding_model)
        self._embedding_model_name = embedding_model
        self._llm_calls = 0

    def get_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "corpus_description": self.corpus_description,
            "embedding_model": self._embedding_model_name,
            "similarity_threshold": self.similarity_threshold,
            "max_group_size": self.max_group_size,
            "llm_model": self.llm_model,
            "batch_size": self.batch_size,
            "fallback_threshold": self.fallback_threshold,
            "retries": self.retries,
        }

    def canonicalize(
        self, extractions: list[ExtractionResult]
    ) -> tuple[list[ExtractionResult], CanonicalizationResult]:
        """Run canonicalization on a list of extraction results.

        Returns:
            Updated extractions (nodes replaced by canonical forms) and a
            CanonicalizationResult carrying the artifacts and run statistics.
        """
        all_keywords = self._collect_keywords(extractions)
        n = len(all_keywords)
        logger.info("Canonicalizing %d unique keywords…", n)

        # 2a — embed
        logger.info("  [2a] Embedding keywords…")
        embeddings = self._embed(all_keywords)

        # 2b — cluster
        logger.info("  [2b] Complete-linkage clustering (θ=%.2f)…",
                    self.similarity_threshold)
        groups = self._cluster(all_keywords, embeddings)
        singletons = [g[0] for g in groups if len(g) == 1]
        non_singletons = [g for g in groups if len(g) > 1]
        logger.info(
            "       %d singletons, %d candidate groups", len(
                singletons), len(non_singletons)
        )

        # 2c — LLM verification
        logger.info("  [2c] LLM verification (%d groups)…",
                    len(non_singletons))
        self._llm_calls = 0
        synonym_table = self._verify_with_llm(non_singletons)

        # 2d — build structures
        canonical_keywords = sorted(
            set(synonym_table.values()) | set(singletons))

        logger.info("  [2d] Embedding %d canonical keywords…",
                    len(canonical_keywords))
        canonical_embeddings = self._embed(canonical_keywords)

        counts = Counter(synonym_table.values())
        merges_performed = sum(c - 1 for c in counts.values() if c > 1)

        stats = {
            "keywords_after_stage1": n,
            "candidate_groups": len(non_singletons),
            "singletons": len(singletons),
            "merges_performed": merges_performed,
            "canonical_keywords_final": len(canonical_keywords),
            "llm_calls": self._llm_calls,
        }

        logger.info(
            "Canonicalization done: %d → %d keywords, %d merges, %d LLM calls",
            n, len(canonical_keywords), merges_performed, self._llm_calls,
        )

        updated = self._apply(extractions, synonym_table)
        result = CanonicalizationResult(
            synonym_table=synonym_table,
            canonical_keywords=canonical_keywords,
            canonical_embeddings=canonical_embeddings,
            stats=stats,
        )
        return updated, result

    @staticmethod
    def _collect_keywords(extractions: list[ExtractionResult]) -> list[str]:
        # List preserves stable order for embedding index alignment, set provides dedup.
        seen: set[str] = set()
        keywords: list[str] = []
        for er in extractions:
            for kw in er.keywords:
                if kw not in seen:
                    keywords.append(kw)
                    seen.add(kw)
        return keywords

    def _embed(self, keywords: list[str]) -> np.ndarray:
        return self._model.encode(keywords, show_progress_bar=False)

    def _cluster(self, keywords: list[str], embeddings: np.ndarray) -> list[list[str]]:
        """Complete-linkage clustering.

        A group forms only when ALL pairs within it have cosine similarity ≥
        self.similarity_threshold (equivalently, distance ≤ 1 − threshold).
        Oversized groups are force-split into max_group_size chunks.
        """
        n = len(keywords)
        if n == 1:
            return [keywords]

        sim = cosine_similarity(embeddings)
        np.fill_diagonal(sim, 1.0)
        dist = np.clip(1.0 - sim, 0.0, None)

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="complete")
        labels = fcluster(Z, t=1.0 - self.similarity_threshold,
                          criterion="distance")

        raw_groups: dict[int, list[str]] = {}
        for kw, label in zip(keywords, labels):
            raw_groups.setdefault(int(label), []).append(kw)

        result: list[list[str]] = []
        for group in raw_groups.values():
            if len(group) <= self.max_group_size:
                result.append(group)
            else:
                for i in range(0, len(group), self.max_group_size):
                    result.append(group[i: i + self.max_group_size])
        return result

    def _verify_with_llm(self, groups: list[list[str]]) -> dict[str, str]:
        """Return a partial synonym table for all keywords in non-singleton groups."""
        partial: dict[str, str] = {}

        small = [g for g in groups if len(g) <= 5]
        large = [g for g in groups if len(g) > 5]

        for i in range(0, len(small), self.batch_size):
            partial.update(self._llm_call(small[i: i + self.batch_size]))

        for group in large:
            partial.update(self._llm_call([group]))

        return partial

    def _normalize_kw(self, kw: str) -> str:
        """Normalize a single keyword using the configured Normalizer or strip+lower."""
        result = self._normalizer.normalize([kw])
        return result[0] if result else kw.strip().lower()

    def _llm_call(self, groups: list[list[str]]) -> dict[str, str]:
        """One OpenRouter API call covering a batch of candidate groups.

        Returns a keyword → canonical mapping only for keywords that the LLM
        confirms are true synonyms. Standalone or unmentioned keywords are omitted;
        callers treat a missing entry as "no synonym found".
        """
        groups_text = "\n".join(
            f"Group {i + 1}: {json.dumps(g)}" for i, g in enumerate(groups)
        )

        system_prompt = SYNONYM_SYSTEM_PROMPT.format(
            corpus_description=self.corpus_description)
        user_prompt = SYNONYM_PROMPT.format(groups_text=groups_text)

        partial: dict[str, str] = {}

        try:
            content = self._client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            self._llm_calls += 1

            parsed = json.loads(content)
            for group_result in parsed.get("groups", []):
                for sg in group_result.get("synonym_groups", []):
                    canonical = self._normalize_kw(sg.get("canonical", ""))
                    for member in sg.get("members", []):
                        if member:
                            partial[self._normalize_kw(member)] = canonical

        except Exception as e:
            logger.warning(
                "LLM call failed after all attempts (%s) — batch skipped", e)

        return partial

    @staticmethod
    def _apply(
        extractions: list[ExtractionResult], synonym_table: dict[str, str]
    ) -> list[ExtractionResult]:
        updated = []
        for er in extractions:
            seen: set[str] = set()
            canonical_nodes: list[str] = []
            for kw in er.keywords:
                canonical = synonym_table.get(kw, kw)
                if canonical not in seen:
                    canonical_nodes.append(canonical)
                    seen.add(canonical)
            updated.append(ExtractionResult(
                chunk_id=er.chunk_id, keywords=canonical_nodes))
        return updated


class MockCanonicalizer:
    """Drop-in replacement for Canonicalizer that replays a pre-saved result.

    Loads a cache file produced by generate_canon_cache.py and returns the
    stored extractions and CanonicalizationResult without running any model
    or LLM. Useful for iterating on pipeline stages that follow canonicalization.

    Args:
        cache_path: Path to the JSON cache file (relative to repo root or absolute).
    """

    def __init__(self, cache_path: str):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._updated_extractions = [
            ExtractionResult(chunk_id=e["chunk_id"], keywords=e["keywords"])
            for e in data["updated_extractions"]
        ]
        self._result = CanonicalizationResult(
            synonym_table=data["synonym_table"],
            canonical_keywords=data["canonical_keywords"],
            canonical_embeddings=np.array(
                data["canonical_embeddings"], dtype=np.float32),
            stats=data.get("stats", {}),
        )
        logger.warning("MockCanonicalizer: loaded cache from %s", cache_path)

    def get_config(self) -> dict[str, Any]:
        return {"class": self.__class__.__name__}

    def canonicalize(
        self, extractions: list[ExtractionResult]
    ) -> tuple[list[ExtractionResult], CanonicalizationResult]:
        logger.warning(
            "MockCanonicalizer: returning cached canonicalization, input ignored")
        return self._updated_extractions, self._result
