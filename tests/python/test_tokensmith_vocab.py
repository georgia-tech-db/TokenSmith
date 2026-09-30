import tempfile
import unittest

from python_engine import tokensmith_store as store
from python_engine import tokensmith_vocab as vocab


SAMPLE_TEXTS = [
    "The database stores each relation as a table of tuples.",
    "A database index speeds up lookups over a large table.",
    "Query the database with SQL; the query optimizer picks a plan.",
    "Transactions keep the database consistent under concurrent queries.",
    "The buffer pool caches database pages read from disk.",
]


def insert_collection(conn, collection_id, chunks):
    conn.execute("INSERT INTO collections(id, name) VALUES (?, ?)", (collection_id, f"Material {collection_id}"))
    conn.execute("INSERT INTO folders(id, path) VALUES (?, ?)", (collection_id, f"/folder/{collection_id}"))
    conn.execute("INSERT INTO collection_items VALUES (?, ?)", (collection_id, collection_id))
    conn.execute(
        "INSERT INTO tokensmith_collection_state(collection_id, status, is_active, added_at) VALUES (?, ?, ?, ?)",
        (collection_id, "ready", 1, store.now_iso()),
    )
    conn.execute(
        "INSERT INTO documents(id, folder_id, document_time, document_path) VALUES (?, ?, 0, ?)",
        (collection_id, collection_id, f"/folder/{collection_id}/paper.pdf"),
    )
    for text in chunks:
        conn.execute(
            "INSERT INTO chunks(document_id, chunk_text, file) VALUES (?, ?, ?)",
            (collection_id, text, "paper.pdf"),
        )


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

    def test_keeps_correctly_spelled_possessives_intact(self) -> None:
        # The query is tokenized the same way the vocabulary was built, so trailing
        # punctuation does not make a correctly spelled word look like a typo.
        corrected, replacements = vocab.correct_query("the databases' pages", ["databases", "pages"])
        self.assertEqual(corrected, "the databases' pages")
        self.assertEqual(replacements, [])

    def test_deduplicates_repeated_replacements(self) -> None:
        _, replacements = vocab.correct_query("databasse and databasse", self.WORDS)
        self.assertEqual(replacements, [{"from": "databasse", "to": "database"}])


class VocabularyStoreTests(unittest.TestCase):
    def test_vocabulary_is_scoped_to_its_collection(self) -> None:
        with tempfile.TemporaryDirectory() as user_data_path:
            store.init_db(user_data_path)
            with store.connect(user_data_path) as conn:
                insert_collection(conn, 1, SAMPLE_TEXTS)
                insert_collection(conn, 2, ["Buffer replacement evicts a cold page."])

            store.save_collection_vocabulary(user_data_path, "1", {"database": 5, "relation": 4})
            store.save_collection_vocabulary(user_data_path, "2", {"eviction": 3})

            self.assertEqual(store.collection_vocabulary_words(user_data_path, ["1"]), ["database", "relation"])
            self.assertEqual(store.collection_vocabulary_words(user_data_path, ["2"]), ["eviction"])
            # Active collections are searched together, so their words merge.
            self.assertEqual(
                store.collection_vocabulary_words(user_data_path, ["1", "2"]),
                ["database", "eviction", "relation"],
            )
            self.assertEqual(store.collection_vocabulary_words(user_data_path, []), [])

    def test_reindexing_replaces_the_previous_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as user_data_path:
            store.init_db(user_data_path)
            with store.connect(user_data_path) as conn:
                insert_collection(conn, 1, SAMPLE_TEXTS)

            store.save_collection_vocabulary(user_data_path, "1", {"stale": 9})
            store.save_collection_vocabulary(user_data_path, "1", {"fresh": 9})
            self.assertEqual(store.collection_vocabulary_words(user_data_path, ["1"]), ["fresh"])

    def test_deleting_a_collection_removes_its_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as user_data_path:
            store.init_db(user_data_path)
            with store.connect(user_data_path) as conn:
                insert_collection(conn, 1, SAMPLE_TEXTS)
            store.save_collection_vocabulary(user_data_path, "1", {"database": 5})

            with store.connect(user_data_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("DELETE FROM collections WHERE id = 1")

            self.assertEqual(store.collection_vocabulary_words(user_data_path, ["1"]), [])

    def test_backfill_covers_collections_indexed_before_vocabularies_existed(self) -> None:
        with tempfile.TemporaryDirectory() as user_data_path:
            store.init_db(user_data_path)
            with store.connect(user_data_path) as conn:
                insert_collection(conn, 1, SAMPLE_TEXTS)

            build = lambda texts: vocab.build_vocabulary(texts, min_count=1)
            self.assertEqual(store.backfill_collection_vocabularies(user_data_path, build), 1)
            self.assertIn("database", store.collection_vocabulary_words(user_data_path, ["1"]))

            # It runs once, so no later search pays for it again.
            self.assertEqual(store.backfill_collection_vocabularies(user_data_path, build), 0)


if __name__ == "__main__":
    unittest.main()
