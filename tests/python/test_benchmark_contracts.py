import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.benchmarks import test_buzzdb_embeddings as benchmark
from tests.benchmarks.test_buzzdb_grounding import validate_grounding_case


class BenchmarkContractTests(unittest.TestCase):
    def cache_state(self):
        chunks = [{"tokensmithChunkId": "a", "text": "First passage"},
                  {"tokensmithChunkId": "b", "text": "Second passage"}]
        return SimpleNamespace(
            chunks=chunks,
            bundle={
                "metadata": {
                    "model": benchmark.MODEL["name"], "modelDigest": benchmark.MODEL["digest"],
                    "ollamaVersion": benchmark.MODEL["ollamaVersion"],
                    "fixtureSha256": benchmark.sha256_bytes(benchmark.FIXTURE_PATH.read_bytes()),
                    "chunksSha256": benchmark.chunks_sha256(chunks),
                    "inputSha256": benchmark.embedding_input_sha256(),
                },
                "chunk_ids": ["a", "b"],
                "passage_embeddings": np.ones((2, benchmark.MODEL["dimensions"])),
            },
        )

    def validate(self, state):
        benchmark.validate_bundle(state.bundle, state.chunks)

    def test_valid_cache_is_accepted(self):
        self.validate(self.cache_state())

    def test_cache_rejects_changed_text_even_when_chunk_ids_stay_the_same(self):
        state = self.cache_state()
        state.chunks[0]["text"] = "Changed chunker output"
        with self.assertRaisesRegex(AssertionError, "chunksSha256"):
            self.validate(state)

    def test_cache_cannot_omit_duplicate_or_reorder_chunks(self):
        for ids in (["a"], ["a", "a"], ["b", "a"]):
            with self.subTest(ids=ids):
                state = self.cache_state()
                state.bundle["chunk_ids"] = ids
                with self.assertRaisesRegex(AssertionError, "missing, duplicating, or reordering"):
                    self.validate(state)

    def test_cache_rejects_invalid_vectors(self):
        for value in (0, float("nan"), float("inf")):
            with self.subTest(value=value):
                state = self.cache_state()
                state.bundle["passage_embeddings"][0] = value
                with self.assertRaisesRegex(AssertionError, "invalid passage_embeddings"):
                    self.validate(state)

    def test_forbidden_chunk_ids_are_checked_even_without_forbidden_text(self):
        failures = validate_grounding_case(
            {"expectedChunkIds": ["a"], "forbiddenChunkIds": ["b"]}, [], ["a", "b"], "answer"
        )
        self.assertTrue(any("forbidden chunks" in failure for failure in failures))

    def test_cache_rejects_changed_model_or_embedding_input(self):
        for key in ("modelDigest", "ollamaVersion", "inputSha256"):
            with self.subTest(key=key):
                state = self.cache_state()
                state.bundle["metadata"][key] = "changed"
                with self.assertRaisesRegex(AssertionError, key):
                    self.validate(state)

    def test_installed_model_and_runtime_must_match_the_pin(self):
        version = {"version": benchmark.MODEL["ollamaVersion"]}
        models = {"models": [{"name": benchmark.MODEL["name"], "digest": benchmark.MODEL["digest"]}]}
        benchmark.validate_ollama_metadata(version, models)
        with self.assertRaisesRegex(AssertionError, "missing or changed"):
            benchmark.validate_ollama_metadata(version, {"models": []})
        models["models"][0]["digest"] = "new-model-under-same-tag"
        with self.assertRaisesRegex(AssertionError, "missing or changed"):
            benchmark.validate_ollama_metadata(version, models)
        with self.assertRaisesRegex(AssertionError, "requires Ollama"):
            benchmark.validate_ollama_metadata({"version": "different"}, models)

    def test_retrieval_adapter_embeds_live_query_and_rejects_keyword_only_fallback(self):
        instance = benchmark.BuzzDBEmbeddingBenchmarkTests()
        instance.user_data_path = "unused-test-path"
        instance.measurements = {"fresh_query_embedding_seconds": 0.0}

        def search(payload):
            self.assertEqual(payload["searchMode"], "hybrid")
            embed, reason = benchmark.engine.resolve_embedding_provider_for_key(benchmark.MODEL_KEY, payload["embeddingModels"])
            self.assertIsNone(reason)
            self.assertEqual(embed(payload["query"]), [1, 2, 3])
            benchmark.engine.vector_search("args unused by mock")
            return {"sources": [], "reason": None}

        with patch.object(benchmark.engine, "search_library", side_effect=search), \
                patch.object(benchmark.engine, "ollama_embedding", return_value=[1, 2, 3]) as embed_query, \
                patch.object(benchmark.store, "vector_search", return_value=[(1, 0.9)]) as vector_search:
            instance.retrieve_sources("A fresh question?", 8)
            embed_query.assert_called_once_with("A fresh question?", benchmark.MODEL_SPEC)
            vector_search.return_value = []
            with self.assertRaisesRegex(AssertionError, "keyword-only"):
                instance.retrieve_sources("Another fresh question?", 8)

    def test_retrieval_reports_embedding_failure_even_if_search_falls_back(self):
        instance = benchmark.BuzzDBEmbeddingBenchmarkTests()
        instance.user_data_path = "unused-test-path"
        instance.measurements = {"fresh_query_embedding_seconds": 0.0}

        def search(payload):
            try:
                benchmark.engine.ollama_embedding(payload["query"], benchmark.MODEL_SPEC)
            except RuntimeError:
                pass
            return {"sources": [], "reason": None}

        with patch.object(benchmark.engine, "search_library", side_effect=search), \
                patch.object(benchmark.engine, "ollama_embedding", side_effect=RuntimeError("runner missing")):
            with self.assertRaisesRegex(AssertionError, "Query embedding failed: runner missing"):
                instance.retrieve_sources("A question?", 8)


if __name__ == "__main__":
    unittest.main()
