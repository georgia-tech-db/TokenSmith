import json
import re
import tempfile
import unittest
from pathlib import Path

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "buzzdb" / "buzzdb-book.tokensmith.md"
CASES_PATH = ROOT / "tests" / "benchmarks" / "buzzdb_grounding_cases.json"
BUZZDB_COLLECTION_ID = "1"


def case_retrieval_query(case):
    return case.get("retrievalQuery") or case["question"]


def normalize_text(value):
    text = str(value).casefold().replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


class BuzzDBGroundingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.user_data_path = cls.temp_dir.name
        cls.cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        cls.chunk_by_id = {}
        cls.chunk_id_by_rowid = {}

        store.init_db(cls.user_data_path)
        chunks = engine.chunk_tokensmith_markdown(FIXTURE_PATH.read_text(encoding="utf-8"))
        if not chunks:
            raise AssertionError(f"No TokenSmith chunks parsed from {FIXTURE_PATH}")

        with store.connect(cls.user_data_path) as conn:
            conn.execute("INSERT INTO collections(id, name) VALUES (?, ?)", (1, "BuzzDBBook"))
            conn.execute("INSERT INTO folders(id, path) VALUES (?, ?)", (1, str(FIXTURE_PATH.parent)))
            conn.execute("INSERT INTO collection_items VALUES (?, ?)", (1, 1))
            conn.execute(
                "INSERT INTO tokensmith_collection_state"
                "(collection_id, status, is_active, added_at, file_count, chunk_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, "ready", 1, store.now_iso(), 1, len(chunks)),
            )
            document_id = store.insert_document(conn, 1, str(FIXTURE_PATH))

            for chunk in chunks:
                chunk_id = chunk.get("tokensmithChunkId")
                if not chunk_id:
                    continue
                stored_chunk = {
                    **chunk,
                    "path": str(FIXTURE_PATH),
                    "documentTitle": "BuzzDB Book",
                }
                rowid = store.insert_chunk(conn, document_id, stored_chunk)
                cls.chunk_by_id[chunk_id] = stored_chunk
                cls.chunk_id_by_rowid[rowid] = chunk_id

        cls.assert_all_case_chunks_exist()

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @classmethod
    def assert_all_case_chunks_exist(cls):
        missing = []
        for case in cls.cases:
            for chunk_id in case.get("expectedChunkIds", []):
                if chunk_id not in cls.chunk_by_id:
                    missing.append(f"{case['id']} expected {chunk_id}")
            for chunk_id in case.get("forbiddenChunkIds", []):
                if chunk_id not in cls.chunk_by_id:
                    missing.append(f"{case['id']} forbidden {chunk_id}")
        if missing:
            raise AssertionError("Missing BuzzDB fixture chunks: " + ", ".join(missing))

    def retrieve_case(self, case):
        question = case_retrieval_query(case)
        top_k = int(case.get("topK", 8))
        terms = store.keyword_terms_for_query(
            self.user_data_path,
            question,
            [BUZZDB_COLLECTION_ID],
        )
        hits = store.keyword_search(
            self.user_data_path,
            question,
            [BUZZDB_COLLECTION_ID],
            top_k,
            terms,
        )
        hit_ids = [self.chunk_id_by_rowid[rowid] for rowid, _score in hits]
        context = "\n\n".join(self.chunk_by_id[chunk_id]["text"] for chunk_id in hit_ids)
        return terms, hit_ids, normalize_text(context)

    def test_buzzdb_grounding_cases(self):
        passed = 0
        for case in self.cases:
            with self.subTest(case=case["id"]):
                terms, hit_ids, normalized_context = self.retrieve_case(case)
                expected = case.get("expectedChunkIds", [])
                min_expected_hits = int(case.get("minExpectedHits", len(expected)))
                matched = [chunk_id for chunk_id in expected if chunk_id in hit_ids]

                self.assertGreaterEqual(
                    len(matched),
                    min_expected_hits,
                    f"{case['id']} matched {matched}, expected at least "
                    f"{min_expected_hits} of {expected}; terms={terms}; hits={hit_ids}",
                )

                forbidden_hits = [
                    chunk_id
                    for chunk_id in case.get("forbiddenChunkIds", [])
                    if chunk_id in hit_ids
                ]
                self.assertEqual(
                    forbidden_hits,
                    [],
                    f"{case['id']} retrieved forbidden chunks {forbidden_hits}; "
                    f"terms={terms}; hits={hit_ids}",
                )

                for required in case.get("requiredContext", []):
                    self.assertIn(
                        normalize_text(required),
                        normalized_context,
                        f"{case['id']} missing required context {required!r}; "
                        f"terms={terms}; hits={hit_ids}",
                    )

                for forbidden in case.get("forbiddenContext", []):
                    self.assertNotIn(
                        normalize_text(forbidden),
                        normalized_context,
                        f"{case['id']} included forbidden context {forbidden!r}; "
                        f"terms={terms}; hits={hit_ids}",
                    )

                passed += 1

        print(f"BuzzDB grounding benchmark: {passed}/{len(self.cases)} cases passed.")


if __name__ == "__main__":
    unittest.main()
