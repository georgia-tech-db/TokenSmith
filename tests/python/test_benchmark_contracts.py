import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tests.benchmarks import test_buzzdb_fastembed as benchmark
from tests.benchmarks.test_buzzdb_grounding import validate_grounding_case


class BenchmarkContractTests(unittest.TestCase):
    def cache_state(self):
        chunks = [{"tokensmithChunkId": "a", "text": "First passage"},
                  {"tokensmithChunkId": "b", "text": "Second passage"}]
        return SimpleNamespace(
            cases=[{"id": "question", "question": "A question?"}],
            chunks=chunks,
            chunk_by_id={chunk["tokensmithChunkId"]: chunk for chunk in chunks},
            bundle={
                "metadata": {
                    "model": benchmark.FASTEMBED_MODEL,
                    "fixtureSha256": benchmark.fixture_sha256(),
                    "chunksSha256": benchmark.chunks_sha256(chunks),
                    "caseIds": ["question"], "queryTexts": ["A question?"],
                },
                "chunk_ids": ["a", "b"], "query_texts": ["A question?"],
                "passage_embeddings": np.ones((2, 3)), "query_embeddings": np.ones((1, 3)),
            },
        )

    def validate(self, state):
        benchmark.BuzzDBFastEmbedBenchmarkTests.validate_bundle.__func__(state)

    def test_valid_cache_is_accepted(self):
        self.validate(self.cache_state())

    def test_cache_rejects_changed_text_even_when_chunk_ids_stay_the_same(self):
        state = self.cache_state()
        state.chunks[0]["text"] = "Changed chunker output"
        with self.assertRaisesRegex(AssertionError, "parsed chunks"):
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

    def test_retrieval_adapter_embeds_live_query_and_rejects_keyword_only_fallback(self):
        instance = benchmark.BuzzDBFastEmbedBenchmarkTests()
        instance.user_data_path = "unused-test-path"
        instance.measurements = {"fresh_query_embedding_seconds": 0.0}
        instance.embedder = Mock()
        instance.embedder.query_embed.return_value = [[1, 2, 3]]

        def search(payload):
            self.assertEqual(payload["searchMode"], "hybrid")
            embed, reason = benchmark.engine.resolve_embedding_provider_for_key("test", [])
            self.assertIsNone(reason)
            self.assertEqual(embed(payload["query"]), [1, 2, 3])
            benchmark.engine.vector_search("args unused by mock")
            return {"sources": [], "reason": None}

        with patch.object(benchmark.engine, "search_library", side_effect=search), \
                patch.object(benchmark.store, "vector_search", return_value=[(1, 0.9)]) as vector_search:
            instance.retrieve_sources("A fresh question?", 8)
            instance.embedder.query_embed.assert_called_once_with(
                ["A fresh question?"], batch_size=benchmark.FASTEMBED_BATCH_SIZE
            )
            vector_search.return_value = []
            with self.assertRaisesRegex(AssertionError, "keyword-only"):
                instance.retrieve_sources("Another fresh question?", 8)


if __name__ == "__main__":
    unittest.main()
