import tempfile
import unittest
from pathlib import Path

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_vocab as vocab


SAMPLE_TEXTS = [
    "The database stores each relation as a table of tuples.",
    "A database index speeds up lookups over a large table.",
    "Query the database with SQL; the query optimizer picks a plan.",
    "Transactions keep the database consistent under concurrent queries.",
    "The buffer pool caches database pages read from disk.",
]


class BuildVocabularyTests(unittest.TestCase):
    def test_counts_high_frequency_words_and_drops_rares(self) -> None:
        result = vocab.build_vocabulary(SAMPLE_TEXTS, min_count=3)
        self.assertIn("database", result)
        self.assertGreaterEqual(result["database"], 5)
        # "optimizer" appears once, below the min_count threshold.
        self.assertNotIn("optimizer", result)

    def test_excludes_stop_words_and_short_tokens(self) -> None:
        result = vocab.build_vocabulary(SAMPLE_TEXTS, min_count=1)
        self.assertNotIn("the", result)
        self.assertNotIn("of", result)
        self.assertNotIn("up", result)  # shorter than MIN_VOCAB_WORD_LENGTH

    def test_respects_max_words_keeping_most_frequent(self) -> None:
        result = vocab.build_vocabulary(SAMPLE_TEXTS, min_count=1, max_words=2)
        self.assertEqual(len(result), 2)
        self.assertIn("database", result)


class CorrectQueryTests(unittest.TestCase):
    WORDS = ["database", "table", "query", "index", "transaction", "buffer"]

    def test_repairs_obvious_typo(self) -> None:
        corrected, replacements = vocab.correct_query("how does the databasse work", self.WORDS)
        self.assertEqual(corrected, "how does the database work")
        self.assertEqual(replacements, [{"from": "databasse", "to": "database"}])

    def test_leaves_exact_matches_untouched(self) -> None:
        corrected, replacements = vocab.correct_query("database index query", self.WORDS)
        self.assertEqual(corrected, "database index query")
        self.assertEqual(replacements, [])

    def test_ignores_short_tokens(self) -> None:
        corrected, replacements = vocab.correct_query("teh dbs", ["the", "dbs"])
        self.assertEqual(corrected, "teh dbs")
        self.assertEqual(replacements, [])

    def test_leaves_words_with_no_close_match(self) -> None:
        corrected, replacements = vocab.correct_query("explain xylophone please", self.WORDS)
        self.assertEqual(corrected, "explain xylophone please")
        self.assertEqual(replacements, [])

    def test_preserves_leading_capitalization(self) -> None:
        corrected, _ = vocab.correct_query("Databasse basics", self.WORDS)
        self.assertEqual(corrected, "Database basics")

    def test_deduplicates_repeated_replacements(self) -> None:
        _, replacements = vocab.correct_query("databasse and databasse", self.WORDS)
        self.assertEqual(replacements, [{"from": "databasse", "to": "database"}])


class VocabPersistenceTests(unittest.TestCase):
    def test_round_trip_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(vocab.load_vocabulary(tmp, "material-1"))

            vocab.save_vocabulary(tmp, "material-1", {"database": 9, "table": 4}, min_count=4)
            loaded = vocab.load_vocabulary(tmp, "material-1")
            self.assertEqual(loaded, {"database": 9, "table": 4})

            vocab.delete_vocabulary(tmp, "material-1")
            self.assertIsNone(vocab.load_vocabulary(tmp, "material-1"))
            # Deleting a missing vocabulary is a no-op.
            vocab.delete_vocabulary(tmp, "material-1")


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._rebuild = engine.VOCAB_FORCE_REBUILD
        self._iter_chunk_texts = engine.iter_chunk_texts

    def tearDown(self) -> None:
        engine.VOCAB_FORCE_REBUILD = self._rebuild
        engine.iter_chunk_texts = self._iter_chunk_texts

    def test_disabled_flag_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = engine.apply_query_typo_correction(tmp, "databasse tuning", ["m1"], False)
        self.assertEqual(result, "databasse tuning")

    def test_lazy_builds_vocabulary_and_corrects_query(self) -> None:
        engine.VOCAB_FORCE_REBUILD = False
        engine.iter_chunk_texts = lambda _user_data_path, _material_ids: SAMPLE_TEXTS

        with tempfile.TemporaryDirectory() as tmp:
            result = engine.apply_query_typo_correction(tmp, "how does the databasse work", ["m1"], True)
            self.assertEqual(result, "how does the database work")
            # The vocabulary was persisted for reuse on the next query.
            self.assertTrue(vocab.vocab_path(tmp, "m1").exists())


if __name__ == "__main__":
    unittest.main()
