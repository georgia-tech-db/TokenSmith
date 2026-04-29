"""Unit tests for PrototypeClassifier.

Stubs the embedder so the test never loads the real Qwen3 GGUF model.
The stub maps each text to a deterministic vector based on its content,
which is enough to exercise the prototype-build / softmax / argmax paths.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pytest

from src.planning.learned_classifier import Classification, PrototypeClassifier


# Map each "category tag" (a substring inside the exemplar) to a unit basis
# vector. This makes the cosine similarity exactly 1.0 between query and
# matching prototype, and exactly 0.0 between query and non-matching.
_CATEGORY_TAGS = ["alpha", "beta", "gamma", "delta"]


class _StubEmbedder:
    """Deterministic embedder: tags inside the text pick a basis vector.

    Each text contributes one basis vector per matching tag (averaged if
    multiple). If no tag is found, returns the zero vector — which lets us
    test the degenerate case.
    """

    def __init__(self):
        self.dim = len(_CATEGORY_TAGS)
        self.basis = {tag: np.eye(self.dim)[i] for i, tag in enumerate(_CATEGORY_TAGS)}
        self.calls: List[str] = []

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            self.calls.append(t)
            tags = [tag for tag in _CATEGORY_TAGS if tag in t.lower()]
            if not tags:
                continue
            v = np.mean([self.basis[tag] for tag in tags], axis=0)
            n = float(np.linalg.norm(v))
            if normalize and n > 0:
                v = v / n
            out[i] = v
        return out


@pytest.fixture()
def embedder() -> _StubEmbedder:
    return _StubEmbedder()


@pytest.fixture()
def prototypes() -> dict:
    # Two exemplars per category, each tagged unambiguously.
    return {
        "alpha": ["alpha example one", "alpha example two"],
        "beta":  ["beta example one",  "beta example two"],
        "gamma": ["gamma example one", "gamma example two"],
    }


@pytest.fixture()
def clf(embedder, prototypes) -> PrototypeClassifier:
    return PrototypeClassifier(embedder, prototypes, softmax_temp=0.1)


def test_prototypes_built_at_construction(embedder, prototypes):
    PrototypeClassifier(embedder, prototypes)
    # Should have embedded each exemplar exactly once.
    assert len(embedder.calls) == sum(len(v) for v in prototypes.values())


def test_classify_returns_correct_category(clf):
    result = clf.classify_query("alpha alpha test query")
    assert isinstance(result, Classification)
    assert result.category == "alpha"


def test_confidence_bounded_between_zero_and_one(clf):
    result = clf.classify_query("beta beta test query")
    assert 0.0 < result.confidence <= 1.0


def test_all_scores_sum_to_one(clf):
    result = clf.classify_query("gamma test")
    total = sum(result.all_scores.values())
    assert pytest.approx(total, abs=1e-5) == 1.0


def test_strongest_signal_dominates(clf):
    # With softmax_temp=0.1, a dot product of 1.0 vs 0.0 should produce a
    # near-degenerate distribution heavily favoring the matching prototype.
    result = clf.classify_query("alpha")
    assert result.confidence > 0.95


def test_ambiguous_input_yields_low_confidence(clf):
    # "alpha beta" matches two prototypes equally — confidence should be
    # closer to 0.5 than to 1.0.
    result = clf.classify_query("alpha beta")
    assert 0.3 < result.confidence < 0.7


def test_empty_query_returns_uniform(clf):
    result = clf.classify_query("")
    n = 3  # three categories in fixture
    assert pytest.approx(result.confidence, abs=1e-5) == 1.0 / n
    assert sorted(result.all_scores.keys()) == ["alpha", "beta", "gamma"]


def test_zero_norm_prototype_rejected(embedder):
    # If all exemplars for a category have no tags, the mean is zero.
    bad_prototypes = {
        "alpha": ["alpha example"],
        "blank": ["nothing matches here"],
    }
    with pytest.raises(ValueError, match="Zero-norm prototype"):
        PrototypeClassifier(embedder, bad_prototypes)


def test_empty_prototypes_rejected(embedder):
    with pytest.raises(ValueError):
        PrototypeClassifier(embedder, {})


def test_empty_exemplar_list_rejected(embedder):
    with pytest.raises(ValueError):
        PrototypeClassifier(embedder, {"alpha": []})


def test_invalid_temp_rejected(embedder, prototypes):
    with pytest.raises(ValueError):
        PrototypeClassifier(embedder, prototypes, softmax_temp=0.0)
    with pytest.raises(ValueError):
        PrototypeClassifier(embedder, prototypes, softmax_temp=-1.0)


def test_classify_does_not_mutate_state(clf):
    # Calling classify_query twice with the same input should produce the
    # same result — no hidden state should accumulate.
    a = clf.classify_query("alpha test")
    b = clf.classify_query("alpha test")
    assert a == b
