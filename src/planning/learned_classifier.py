"""
Learned Query Classifier
------------------------
Zero-shot prototype classifier: embed a set of exemplar queries per category,
mean-pool to form a prototype vector, then at inference embed the query and
softmax cosine similarities to produce calibrated category probabilities.

No training needed — uses the same embedder already loaded for retrieval.

Used by CostModelPlanner as a drop-in alternative to the regex-based
HeuristicQueryPlanner. Both expose `classify_query(query) -> Classification`
so the cost-model doesn't care which one it gets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import numpy as np


@dataclass
class Classification:
    category: str
    confidence: float                         # softmax probability of top class
    all_scores: Dict[str, float] = field(default_factory=dict)


class Classifier(Protocol):
    """Minimal interface CostModelPlanner relies on."""

    def classify_query(self, query: str) -> Classification: ...


class PrototypeClassifier:
    """
    Zero-shot prototype classifier via cosine similarity over query embeddings.

    Construction is O(num_categories * num_exemplars) embedder calls; classify
    is one embedder call + a few dot products. The embedder is reused — pass
    in the same SentenceTransformer instance the retrievers use.
    """

    def __init__(
        self,
        embedder,
        prototypes: Dict[str, List[str]],
        softmax_temp: float = 0.1,
    ):
        if not prototypes:
            raise ValueError("PrototypeClassifier requires non-empty prototypes")
        if softmax_temp <= 0:
            raise ValueError("softmax_temp must be > 0")
        self.embedder = embedder
        self.softmax_temp = float(softmax_temp)
        self.prototypes = {cat: list(exs) for cat, exs in prototypes.items()}
        self.proto_embs: Dict[str, np.ndarray] = self._build_prototypes()

    def _build_prototypes(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for cat, exemplars in self.prototypes.items():
            if not exemplars:
                raise ValueError(f"Category {cat!r} has no exemplars")
            embs = self.embedder.encode(list(exemplars), normalize=True)
            embs = np.asarray(embs, dtype=np.float32)
            mean = embs.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm == 0.0:
                raise ValueError(f"Zero-norm prototype for category {cat!r}")
            out[cat] = mean / norm
        return out

    def classify_query(self, query: str) -> Classification:
        if not query or not query.strip():
            # Degenerate input — pick the first category at confidence 1/N.
            n = len(self.proto_embs)
            cat = next(iter(self.proto_embs))
            return Classification(
                category=cat,
                confidence=1.0 / n,
                all_scores={c: 1.0 / n for c in self.proto_embs},
            )
        q_arr = self.embedder.encode([query], normalize=True)
        q = np.asarray(q_arr, dtype=np.float32)[0]
        # Re-normalize defensively in case the embedder didn't honor normalize=True.
        nq = float(np.linalg.norm(q))
        if nq > 0:
            q = q / nq
        sims = {cat: float(q @ emb) for cat, emb in self.proto_embs.items()}
        probs = self._softmax(sims)
        top = max(probs, key=probs.get)
        return Classification(category=top, confidence=probs[top], all_scores=probs)

    def _softmax(self, sims: Dict[str, float]) -> Dict[str, float]:
        max_s = max(sims.values())
        exps = {cat: math.exp((s - max_s) / self.softmax_temp) for cat, s in sims.items()}
        total = sum(exps.values())
        return {cat: e / total for cat, e in exps.items()}
