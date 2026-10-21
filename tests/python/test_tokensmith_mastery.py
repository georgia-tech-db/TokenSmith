import tempfile
import unittest

from python_engine import tokensmith_store as store


class TopicMasteryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.question = 0

    def record(self, topic, correct, assisted=False, session_id="s1"):
        self.question += 1
        return store.record_practice_attempt(self.root, session_id, f"q{self.question}", topic, correct, assisted)

    def test_first_correct_answer_moves_mastery_from_even_odds(self):
        entry = self.record("B+ trees", True)
        self.assertEqual(entry["topic"], "B+ trees")
        self.assertAlmostEqual(entry["mastery"], 2 / 3)
        self.assertEqual(entry["attempts"], 1)
        self.assertTrue(entry["updatedAt"])

    def test_incorrect_answer_counts_against_the_topic(self):
        self.assertAlmostEqual(self.record("joins", False)["mastery"], 1 / 3)

    def test_assisted_correct_answer_earns_half_credit(self):
        self.assertAlmostEqual(self.record("joins", True, assisted=True)["mastery"], 1.5 / 2.5)

    def test_answers_accumulate_per_topic_and_topic_is_trimmed(self):
        self.record("  joins ", True)
        self.record("joins", False)
        entry = self.record("joins", True, assisted=True)
        self.assertAlmostEqual(entry["mastery"], 2.5 / 4.5)
        self.assertEqual(entry["attempts"], 3)
        self.assertEqual([item["topic"] for item in store.list_topic_mastery(self.root)], ["joins"])

    def test_saving_the_same_answer_twice_counts_once(self):
        store.record_practice_attempt(self.root, "s1", "q1", "joins", True, False)
        entry = store.record_practice_attempt(self.root, "s1", "q1", "joins", False, False)
        self.assertEqual(entry["attempts"], 1)
        self.assertAlmostEqual(entry["mastery"], 2 / 3)
        self.assertEqual(store.record_practice_attempt(self.root, "s2", "q1", "joins", False, False)["attempts"], 2)

    def test_blank_topic_records_nothing(self):
        self.assertIsNone(self.record("", True))
        self.assertIsNone(self.record("   ", True))
        self.assertEqual(store.list_topic_mastery(self.root), [])

    def test_mastery_stays_strictly_between_zero_and_one(self):
        for _ in range(200):
            high = self.record("high", True)
            low = self.record("low", False)
        self.assertLess(high["mastery"], 1.0)
        self.assertGreater(low["mastery"], 0.0)
        self.assertEqual(high["attempts"], 200)

    def test_list_returns_every_topic_in_order(self):
        self.record("joins", True)
        self.record("indexes", False)
        entries = store.list_topic_mastery(self.root)
        self.assertEqual([item["topic"] for item in entries], ["indexes", "joins"])
        self.assertAlmostEqual(entries[0]["mastery"], 1 / 3)


class SchemaUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name

    def names(self, kind):
        with store.connect(self.root) as conn:
            return {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}

    def test_schema_11_database_gains_the_attempt_table_without_losing_data(self):
        store.init_db(self.root)
        with store.connect(self.root) as conn:
            conn.execute("DROP TABLE practice_attempts")
            store.set_schema_value(conn, "version", "11")
            conn.execute("INSERT INTO collections(id, name) VALUES (7, 'Databases')")
        store.init_db(self.root)
        self.assertIn("practice_attempts", self.names("table"))
        self.assertEqual(store.get_schema_value(self.root, "version"), "13")
        with store.connect(self.root) as conn:
            self.assertEqual(conn.execute("SELECT name FROM collections WHERE id = 7").fetchone()["name"], "Databases")

    def test_schema_12_attempt_log_is_migrated_and_the_mastery_table_removed(self):
        store.init_db(self.root)
        with store.connect(self.root) as conn:
            conn.execute("DROP TABLE practice_attempts")
            conn.executescript(
                """
                CREATE TABLE practice_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, question_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL, question TEXT NOT NULL, student_answer TEXT NOT NULL,
                    verdict TEXT NOT NULL, checks_json TEXT NOT NULL, reference_answer TEXT, source_keys_json TEXT NOT NULL,
                    topic TEXT, assisted INTEGER NOT NULL DEFAULT 0, hint_used INTEGER NOT NULL DEFAULT 0,
                    answer_revealed INTEGER NOT NULL DEFAULT 0, duration_ms INTEGER, created_at TEXT NOT NULL
                );
                CREATE TABLE topic_mastery (topic TEXT PRIMARY KEY, alpha REAL, beta REAL, attempts INTEGER, updated_at TEXT);
                CREATE INDEX idx_practice_attempts_session_id ON practice_attempts(session_id);
                CREATE INDEX idx_practice_attempts_topic ON practice_attempts(topic);
                """
            )
            rows = [
                ("s1", "q1", "correct", " joins ", 0),
                ("s1", "q2", "incorrect", "joins", 1),
                ("s1", "q3", "needs_work", "joins", 0),
                ("s1", "q4", "correct", None, 0),
                ("s1", "q1", "incorrect", "joins", 0),
            ]
            for session_id, question_id, verdict, topic, assisted in rows:
                conn.execute(
                    "INSERT INTO practice_attempts(session_id, question_id, attempt_number, question, student_answer, verdict,"
                    " checks_json, source_keys_json, topic, assisted, created_at) VALUES (?, ?, 1, 'Q', 'A', ?, '[]', '[]', ?, ?, ?)",
                    (session_id, question_id, verdict, topic, assisted, "2026-10-14T00:00:00Z"),
                )
            store.set_schema_value(conn, "version", "12")

        store.init_db(self.root)

        self.assertNotIn("topic_mastery", self.names("table"))
        self.assertNotIn("practice_attempts_v12", self.names("table"))
        self.assertIn("idx_practice_attempts_topic", self.names("index"))
        self.assertNotIn("idx_practice_attempts_session_id", self.names("index"))
        [entry] = store.list_topic_mastery(self.root)
        self.assertEqual((entry["topic"], entry["attempts"]), ("joins", 2))
        self.assertAlmostEqual(entry["mastery"], 2 / 4)
        store.init_db(self.root)
        self.assertEqual(store.list_topic_mastery(self.root), [entry])


if __name__ == "__main__":
    unittest.main()
