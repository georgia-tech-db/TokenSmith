import numpy as np
import pytest

from src.embedder import SentenceTransformer, _candidate_contexts, _load_embedding_model


pytestmark = pytest.mark.unit


class SingleOnlyModel:
    def __init__(self):
        self.calls = []

    def create_embedding(self, input_value):
        self.calls.append(input_value)
        if isinstance(input_value, list):
            raise AssertionError("single-input path expected")
        value = float(len(str(input_value)))
        return {"data": [{"embedding": [value, value + 1.0]}]}


class BatchCapableModel:
    def __init__(self):
        self.calls = []

    def create_embedding(self, input_value):
        self.calls.append(input_value)
        if isinstance(input_value, list):
            return {
                "data": [
                    {"embedding": [float(index), float(index) + 0.5]}
                    for index, _ in enumerate(input_value, start=1)
                ]
            }
        return {"data": [{"embedding": [1.0, 1.5]}]}


def _build_embedder(model):
    embedder = SentenceTransformer.__new__(SentenceTransformer)
    embedder.model = model
    embedder.model_path = "fake.gguf"
    embedder.n_ctx = 4096
    embedder._embedding_dimension = 2
    return embedder


def test_sentence_transformer_encode_defaults_to_single_item_calls(monkeypatch):
    monkeypatch.delenv("TOKENSMITH_ENABLE_BATCH_EMBEDDINGS", raising=False)
    embedder = _build_embedder(SingleOnlyModel())

    embeddings = embedder.encode(["alpha", "beta"], batch_size=8, show_progress_bar=False)

    assert embeddings.shape == (2, 2)
    assert np.allclose(embeddings[0], np.array([5.0, 6.0], dtype=np.float32))
    assert embedder.model.calls == ["alpha", "beta"]


def test_sentence_transformer_encode_supports_opt_in_batch_api(monkeypatch):
    monkeypatch.setenv("TOKENSMITH_ENABLE_BATCH_EMBEDDINGS", "1")
    embedder = _build_embedder(BatchCapableModel())

    embeddings = embedder.encode(["alpha", "beta"], batch_size=8, show_progress_bar=False)

    assert embeddings.shape == (2, 2)
    assert np.allclose(embeddings[1], np.array([2.0, 2.5], dtype=np.float32))
    assert embedder.model.calls == [["alpha", "beta"]]


def test_candidate_contexts_descend_without_duplicates():
    assert _candidate_contexts(4096) == [4096, 2048, 1024, 512]
    assert _candidate_contexts(1024) == [1024, 512]


def test_load_embedding_model_retries_with_smaller_context(monkeypatch):
    class FakeLlama:
        def __init__(self, **kwargs):
            if kwargs["n_ctx"] > 1024:
                raise ValueError("context too large")
            self.kwargs = kwargs

    monkeypatch.delenv("TOKENSMITH_FORCE_CPU", raising=False)
    monkeypatch.setattr("src.embedder.Llama", FakeLlama)

    model, actual_n_ctx = _load_embedding_model("fake.gguf", n_ctx=4096, n_threads=1, verbose=False)

    assert actual_n_ctx == 1024
    assert model.kwargs["n_ctx"] == 1024
