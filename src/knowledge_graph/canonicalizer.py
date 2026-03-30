import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.knowledge_graph.models import ExtractionResult
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.utils.normalizer import Normalizer
from src.knowledge_graph.utils.prompts import SYNONYM_PROMPT, SYNONYM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class CanonicalizationResult:
    synonym_table: dict[str, str]
    canonical_keywords: list[str]
    canonical_embeddings: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


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
        embedding_model: str = "all-MiniLM-L6-v2",
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
        logger.info("  [2b] Complete-linkage clustering (θ=%.2f)…", self.similarity_threshold)
        groups = self._cluster(all_keywords, embeddings)
        singletons = [g[0] for g in groups if len(g) == 1]
        non_singletons = [g for g in groups if len(g) > 1]
        logger.info(
            "       %d singletons, %d candidate groups", len(singletons), len(non_singletons)
        )

        # 2c — LLM verification
        logger.info("  [2c] LLM verification (%d groups)…", len(non_singletons))
        self._llm_calls = 0
        partial_table = self._verify_with_llm(non_singletons)

        # 2d — build structures
        synonym_table, canonical_keywords = self._build_canonical_structures(
            singletons, partial_table
        )

        logger.info("  [2d] Embedding %d canonical keywords…", len(canonical_keywords))
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_keywords(extractions: list[ExtractionResult]) -> list[str]:
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
        labels = fcluster(Z, t=1.0 - self.similarity_threshold, criterion="distance")

        raw_groups: dict[int, list[str]] = {}
        for kw, label in zip(keywords, labels):
            raw_groups.setdefault(int(label), []).append(kw)

        result: list[list[str]] = []
        for group in raw_groups.values():
            if len(group) <= self.max_group_size:
                result.append(group)
            else:
                for i in range(0, len(group), self.max_group_size):
                    result.append(group[i : i + self.max_group_size])
        return result

    def _verify_with_llm(self, groups: list[list[str]]) -> dict[str, str]:
        """Return a partial synonym table for all keywords in non-singleton groups."""
        partial: dict[str, str] = {}

        small = [g for g in groups if len(g) <= 5]
        large = [g for g in groups if len(g) > 5]

        for i in range(0, len(small), self.batch_size):
            partial.update(self._llm_call(small[i : i + self.batch_size]))

        for group in large:
            partial.update(self._llm_call([group]))

        return partial

    def _normalize_kw(self, kw: str) -> str:
        """Normalize a single keyword using the configured Normalizer or strip+lower."""
        result = self._normalizer.normalize([kw])
        return result[0] if result else kw.strip().lower()

    def _llm_call(self, groups: list[list[str]]) -> dict[str, str]:
        """One OpenRouter API call covering a batch of candidate groups.

        Returns a keyword → canonical mapping for every keyword in the batch.
        Keywords not mentioned by the LLM fall back to mapping to themselves.
        """
        groups_text = "\n".join(
            f"Group {i + 1}: {json.dumps(g)}" for i, g in enumerate(groups)
        )

        
        system_prompt = SYNONYM_SYSTEM_PROMPT.format(corpus_description=self.corpus_description)
        user_prompt = SYNONYM_PROMPT.format(groups_text=groups_text)

        all_group_kws = {kw for g in groups for kw in g}
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
                for kw in group_result.get("standalone", []):
                    if kw:
                        norm = self._normalize_kw(kw)
                        partial[norm] = norm

        except Exception as e:
            logger.warning(
                "LLM call failed after all attempts (%s) — treating batch as standalone", e
            )

        # Fallback: any keyword the LLM didn't mention maps to itself
        for kw in all_group_kws:
            partial.setdefault(kw, kw)

        return partial

    @staticmethod
    def _build_canonical_structures(
        singletons: list[str],
        partial_table: dict[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        synonym_table = dict(partial_table)

        for kw in singletons:
            synonym_table.setdefault(kw, kw)

        canonical_keywords = sorted(set(synonym_table.values()))

        # Ensure every canonical form maps to itself
        for canonical in canonical_keywords:
            synonym_table.setdefault(canonical, canonical)

        return synonym_table, canonical_keywords

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
            updated.append(ExtractionResult(chunk_id=er.chunk_id, keywords=canonical_nodes))
        return updated
