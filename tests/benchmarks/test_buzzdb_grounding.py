import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "buzzdb" / "buzzdb-book.tokensmith.md"
CASES_PATH = ROOT / "tests" / "benchmarks" / "buzzdb_grounding_cases.json"
BUZZDB_COLLECTION_ID = "1"


def case_retrieval_query(case):
    return case.get("retrievalQuery") or case["question"]


def normalize_text(value):
    text = str(value).casefold().replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def load_buzzdb_fixture(user_data_path):
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    chunk_by_id = {}
    chunk_id_by_rowid = {}

    store.init_db(user_data_path)
    chunks = engine.chunk_tokensmith_markdown(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise AssertionError(f"No TokenSmith chunks parsed from {FIXTURE_PATH}")

    with store.connect(user_data_path) as conn:
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
            chunk_by_id[chunk_id] = stored_chunk
            chunk_id_by_rowid[rowid] = chunk_id

    assert_all_case_chunks_exist(cases, chunk_by_id)
    return cases, chunk_by_id, chunk_id_by_rowid


def assert_all_case_chunks_exist(cases, chunk_by_id):
    missing = []
    for case in cases:
        for chunk_id in case.get("expectedChunkIds", []):
            if chunk_id not in chunk_by_id:
                missing.append(f"{case['id']} expected {chunk_id}")
        for chunk_id in case.get("forbiddenChunkIds", []):
            if chunk_id not in chunk_by_id:
                missing.append(f"{case['id']} forbidden {chunk_id}")
    if missing:
        raise AssertionError("Missing BuzzDB fixture chunks: " + ", ".join(missing))


def retrieve_case(user_data_path, chunk_by_id, chunk_id_by_rowid, case):
    question = case_retrieval_query(case)
    top_k = int(case.get("topK", 8))
    terms = store.keyword_terms_for_query(
        user_data_path,
        question,
        [BUZZDB_COLLECTION_ID],
    )
    hits = store.keyword_search(
        user_data_path,
        question,
        [BUZZDB_COLLECTION_ID],
        top_k,
        terms,
    )
    hit_ids = [chunk_id_by_rowid[rowid] for rowid, _score in hits]
    context = "\n\n".join(chunk_by_id[chunk_id]["text"] for chunk_id in hit_ids)
    return terms, hit_ids, normalize_text(context)


def validate_grounding_case(case, terms, hit_ids, normalized_context):
    failures = []
    expected = case.get("expectedChunkIds", [])
    min_expected_hits = int(case.get("minExpectedHits", len(expected)))
    matched = [chunk_id for chunk_id in expected if chunk_id in hit_ids]

    if len(matched) < min_expected_hits:
        failures.append(
            f"matched {matched}, expected at least {min_expected_hits} of {expected}; "
            f"terms={terms}; hits={hit_ids}"
        )

    forbidden_hits = [
        chunk_id
        for chunk_id in case.get("forbiddenChunkIds", [])
        if chunk_id in hit_ids
    ]
    if forbidden_hits:
        failures.append(
            f"retrieved forbidden chunks {forbidden_hits}; terms={terms}; hits={hit_ids}"
        )

    for required in case.get("requiredContext", []):
        if normalize_text(required) not in normalized_context:
            failures.append(
                f"missing required context {required!r}; terms={terms}; hits={hit_ids}"
            )

    for forbidden in case.get("forbiddenContext", []):
        if normalize_text(forbidden) in normalized_context:
            failures.append(
                f"included forbidden context {forbidden!r}; terms={terms}; hits={hit_ids}"
            )

    return failures


def run_buzzdb_grounding_benchmark():
    with tempfile.TemporaryDirectory() as user_data_path:
        cases, chunk_by_id, chunk_id_by_rowid = load_buzzdb_fixture(user_data_path)
        results = []

        for case in cases:
            terms, hit_ids, normalized_context = retrieve_case(
                user_data_path,
                chunk_by_id,
                chunk_id_by_rowid,
                case,
            )
            failures = validate_grounding_case(case, terms, hit_ids, normalized_context)
            results.append({
                "id": case["id"],
                "passed": not failures,
                "terms": terms,
                "hits": hit_ids,
                "failures": failures,
            })

    return {
        "name": "grounding",
        "label": "grounding",
        "unit": "cases",
        "passed": sum(1 for result in results if result["passed"]),
        "total": len(results),
        "cases": results,
    }


class BuzzDBGroundingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.user_data_path = cls.temp_dir.name
        cls.cases, cls.chunk_by_id, cls.chunk_id_by_rowid = load_buzzdb_fixture(cls.user_data_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_buzzdb_grounding_cases(self):
        passed = 0
        for case in self.cases:
            with self.subTest(case=case["id"]):
                terms, hit_ids, normalized_context = retrieve_case(
                    self.user_data_path,
                    self.chunk_by_id,
                    self.chunk_id_by_rowid,
                    case,
                )
                failures = validate_grounding_case(case, terms, hit_ids, normalized_context)
                self.assertEqual([], failures, f"{case['id']} failed: {'; '.join(failures)}")

                passed += 1

        print(f"BuzzDB grounding benchmark: {passed}/{len(self.cases)} cases passed.")


def main():
    if "--json" in sys.argv:
        result = run_buzzdb_grounding_benchmark()
        print(json.dumps(result, sort_keys=True))
        if result["passed"] != result["total"]:
            raise SystemExit(1)
        return

    unittest.main()


if __name__ == "__main__":
    main()
