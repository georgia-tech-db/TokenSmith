import hashlib
import json
import os
import re
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "buzzdb" / "buzzdb-book.tokensmith.md"
CASES_PATH = ROOT / "tests" / "benchmarks" / "buzzdb_grounding_cases.json"
COLLECTION_ID = "1"
FASTEMBED_MODEL = os.environ.get("TOKENSMITH_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
FASTEMBED_MODEL_KEY = f"fastembed:{FASTEMBED_MODEL}"
FASTEMBED_BATCH_SIZE = int(os.environ.get("TOKENSMITH_FASTEMBED_BATCH_SIZE", "64"))
MIN_HYBRID_RECALL = float(os.environ.get("TOKENSMITH_FASTEMBED_MIN_HYBRID_RECALL", "0.80"))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fixture_sha256():
    return sha256_bytes(FIXTURE_PATH.read_bytes())


def load_cases():
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def case_retrieval_query(case):
    return case.get("retrievalQuery") or case["question"]


def cases_sha256(cases):
    payload = [
        {
            "id": case["id"],
            "retrievalQuery": case_retrieval_query(case),
        }
        for case in cases
    ]
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def normalize_text(value):
    text = str(value).casefold().replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def explicit_fastembed_benchmark():
    return (
        os.environ.get("TOKENSMITH_RUN_FASTEMBED_BENCHMARK") == "1"
        or os.environ.get("TOKENSMITH_REQUIRE_FASTEMBED") == "1"
        or bool(os.environ.get("TOKENSMITH_FASTEMBED_EMBEDDINGS_PATH"))
        or bool(os.environ.get("TOKENSMITH_FASTEMBED_EMBEDDINGS_URL"))
    )


def require_fastembed():
    try:
        from fastembed import TextEmbedding
    except Exception as error:
        raise AssertionError(
            "FastEmbed is required to build this benchmark cache. Install it with "
            "`python -m pip install -r requirements-embedding-benchmark.txt`."
        ) from error
    return TextEmbedding


def instantiate_text_embedding(TextEmbedding):
    kwargs = {"model_name": FASTEMBED_MODEL}
    cache_dir = os.environ.get("TOKENSMITH_FASTEMBED_CACHE_DIR")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        return TextEmbedding(**kwargs, providers=["CPUExecutionProvider"])
    except TypeError:
        return TextEmbedding(**kwargs)


def embed_many(embedder, method_name, texts):
    method = getattr(embedder, method_name, None)
    if method is not None:
        try:
            return list(method(texts, batch_size=FASTEMBED_BATCH_SIZE))
        except TypeError:
            return list(method(texts))

    prefix = "query: " if method_name == "query_embed" else "passage: "
    try:
        return list(embedder.embed([f"{prefix}{text}" for text in texts], batch_size=FASTEMBED_BATCH_SIZE))
    except TypeError:
        return list(embedder.embed([f"{prefix}{text}" for text in texts]))


def embed_in_batches(embedder, method_name, texts, label):
    vectors = []
    start = time.perf_counter()
    for offset in range(0, len(texts), FASTEMBED_BATCH_SIZE):
        batch = texts[offset:offset + FASTEMBED_BATCH_SIZE]
        vectors.extend(embed_many(embedder, method_name, batch))
        print(f"{label}: {min(offset + len(batch), len(texts))}/{len(texts)}", flush=True)
    return np.asarray(vectors, dtype=np.float32), time.perf_counter() - start


def all_chunks():
    chunks = engine.chunk_tokensmith_markdown(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise AssertionError(f"No TokenSmith chunks parsed from {FIXTURE_PATH}")
    return [chunk for chunk in chunks if chunk.get("tokensmithChunkId")]


def compute_embedding_bundle():
    cases = load_cases()
    chunks = all_chunks()
    measurements = {}
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    TextEmbedding = require_fastembed()
    start = time.perf_counter()
    embedder = instantiate_text_embedding(TextEmbedding)
    measurements["model_load_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    embed_many(embedder, "query_embed", ["What is a database buffer manager?"])
    measurements["warmup_seconds"] = time.perf_counter() - start

    chunk_ids = [chunk["tokensmithChunkId"] for chunk in chunks]
    passage_texts = [chunk["text"] for chunk in chunks]
    query_texts = [case_retrieval_query(case) for case in cases]
    passage_embeddings, passage_seconds = embed_in_batches(embedder, "passage_embed", passage_texts, "Embedded passages")
    query_embeddings, query_seconds = embed_in_batches(
        embedder,
        "query_embed",
        query_texts,
        "Embedded benchmark queries",
    )
    measurements["passage_embedding_seconds"] = passage_seconds
    measurements["query_embedding_seconds"] = query_seconds
    measurements["chunk_count"] = len(chunks)
    measurements["case_count"] = len(cases)

    metadata = {
        "model": FASTEMBED_MODEL,
        "modelKey": FASTEMBED_MODEL_KEY,
        "fixture": str(FIXTURE_PATH.relative_to(ROOT)),
        "fixtureSha256": fixture_sha256(),
        "cases": str(CASES_PATH.relative_to(ROOT)),
        "casesSha256": cases_sha256(cases),
        "chunkCount": len(chunks),
        "caseIds": [case["id"] for case in cases],
        "questions": [case["question"] for case in cases],
        "queryTexts": query_texts,
        "measurements": measurements,
    }
    return {
        "metadata": metadata,
        "chunk_ids": chunk_ids,
        "passage_embeddings": passage_embeddings,
        "query_texts": query_texts,
        "query_embeddings": query_embeddings,
    }


def write_embedding_bundle(output_path, bundle):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(bundle["metadata"], sort_keys=True)),
        chunk_ids=np.asarray(bundle["chunk_ids"]),
        passage_embeddings=bundle["passage_embeddings"],
        query_texts=np.asarray(bundle["query_texts"]),
        query_embeddings=bundle["query_embeddings"],
    )


def download_embedding_bundle(url):
    destination = Path(tempfile.gettempdir()) / "tokensmith-buzzdb-fastembed-cache.npz"
    start = time.perf_counter()
    urllib.request.urlretrieve(url, destination)
    elapsed = time.perf_counter() - start
    expected_sha256 = os.environ.get("TOKENSMITH_FASTEMBED_EMBEDDINGS_SHA256")
    if expected_sha256:
        actual_sha256 = sha256_bytes(destination.read_bytes())
        if actual_sha256 != expected_sha256:
            raise AssertionError(
                f"Downloaded embedding cache checksum mismatch: expected {expected_sha256}, received {actual_sha256}."
            )
    return destination, elapsed


def load_embedding_bundle():
    path = os.environ.get("TOKENSMITH_FASTEMBED_EMBEDDINGS_PATH")
    download_seconds = None
    if not path and os.environ.get("TOKENSMITH_FASTEMBED_EMBEDDINGS_URL"):
        downloaded_path, download_seconds = download_embedding_bundle(os.environ["TOKENSMITH_FASTEMBED_EMBEDDINGS_URL"])
        path = str(downloaded_path)

    if path:
        start = time.perf_counter()
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        if "query_texts" in data.files:
            query_texts = [str(query) for query in data["query_texts"].tolist()]
        else:
            query_texts = [str(query) for query in metadata.get("queryTexts") or metadata.get("questions") or []]
        load_seconds = time.perf_counter() - start
        if download_seconds is not None:
            metadata.setdefault("measurements", {})["download_seconds"] = download_seconds
        metadata.setdefault("measurements", {})["cache_load_seconds"] = load_seconds
        return {
            "metadata": metadata,
            "chunk_ids": [str(chunk_id) for chunk_id in data["chunk_ids"].tolist()],
            "passage_embeddings": np.asarray(data["passage_embeddings"], dtype=np.float32),
            "query_texts": query_texts,
            "query_embeddings": np.asarray(data["query_embeddings"], dtype=np.float32),
        }

    return compute_embedding_bundle()


class BuzzDBFastEmbedBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not explicit_fastembed_benchmark():
            raise unittest.SkipTest("FastEmbed benchmark is opt-in.")

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.user_data_path = cls.temp_dir.name
        cls.cases = load_cases()
        cls.chunks = all_chunks()
        cls.chunk_by_id = {chunk["tokensmithChunkId"]: chunk for chunk in cls.chunks}
        cls.chunk_id_by_rowid = {}
        cls.bundle = load_embedding_bundle()
        cls.measurements = cls.bundle["metadata"].get("measurements", {})
        cls.validate_bundle()
        cls.query_embedding_by_text = {
            query_text: np.asarray(embedding, dtype=np.float32)
            for query_text, embedding in zip(cls.bundle["query_texts"], cls.bundle["query_embeddings"])
        }
        cls.index_fixture_from_bundle()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp_dir"):
            cls.temp_dir.cleanup()

    @classmethod
    def validate_bundle(cls):
        metadata = cls.bundle["metadata"]
        expected_case_ids = [case["id"] for case in cls.cases]
        expected_query_texts = [case_retrieval_query(case) for case in cls.cases]
        if metadata.get("model") != FASTEMBED_MODEL:
            raise AssertionError(f"Embedding cache model is {metadata.get('model')}; expected {FASTEMBED_MODEL}.")
        if metadata.get("fixtureSha256") != fixture_sha256():
            raise AssertionError("Embedding cache does not match the current BuzzDBBook fixture.")
        if metadata.get("caseIds") != expected_case_ids:
            raise AssertionError("Embedding cache case order does not match the benchmark cases.")
        metadata_query_texts = metadata.get("queryTexts") or metadata.get("questions") or []
        if metadata_query_texts != expected_query_texts:
            raise AssertionError("Embedding cache query texts do not match the benchmark retrieval queries.")
        if cls.bundle["query_texts"] != expected_query_texts:
            raise AssertionError("Embedding cache query vector keys do not match the benchmark retrieval queries.")
        if cls.bundle["passage_embeddings"].shape[0] != len(cls.bundle["chunk_ids"]):
            raise AssertionError("Embedding cache has a chunk/vector count mismatch.")
        if cls.bundle["query_embeddings"].shape[0] != len(cls.bundle["query_texts"]):
            raise AssertionError("Embedding cache has a query/vector count mismatch.")
        missing = [chunk_id for chunk_id in cls.bundle["chunk_ids"] if chunk_id not in cls.chunk_by_id]
        if missing:
            raise AssertionError(f"Embedding cache references unknown chunks: {missing[:5]}")

    @classmethod
    def index_fixture_from_bundle(cls):
        start = time.perf_counter()
        store.init_db(cls.user_data_path)
        with store.connect(cls.user_data_path) as conn:
            conn.execute("INSERT INTO collections(id, name, embedding_model) VALUES (?, ?, ?)", (1, "BuzzDBBook", FASTEMBED_MODEL_KEY))
            conn.execute("INSERT INTO folders(id, path) VALUES (?, ?)", (1, str(FIXTURE_PATH.parent)))
            conn.execute("INSERT INTO collection_items VALUES (?, ?)", (1, 1))
            conn.execute(
                "INSERT INTO tokensmith_collection_state"
                "(collection_id, status, is_active, added_at, file_count, chunk_count, embedding_model_id, embedding_model_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (1, "ready", 1, store.now_iso(), 1, len(cls.bundle["chunk_ids"]), FASTEMBED_MODEL_KEY, FASTEMBED_MODEL),
            )
            document_id = store.insert_document(conn, 1, str(FIXTURE_PATH))

            for chunk_id, embedding in zip(cls.bundle["chunk_ids"], cls.bundle["passage_embeddings"]):
                chunk = {
                    **cls.chunk_by_id[chunk_id],
                    "path": str(FIXTURE_PATH),
                    "documentTitle": "BuzzDB Book",
                }
                rowid = store.insert_chunk(conn, document_id, chunk)
                blob, _dim = store.vector_to_blob(embedding)
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings(model, folder_id, chunk_id, embedding) VALUES (?, ?, ?, ?)",
                    (FASTEMBED_MODEL_KEY, 1, rowid, blob),
                )
                cls.chunk_id_by_rowid[rowid] = chunk_id

        cls.measurements["db_insert_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        store.rebuild_faiss(cls.user_data_path, FASTEMBED_MODEL_KEY)
        cls.measurements["faiss_rebuild_seconds"] = time.perf_counter() - start

    def retrieve_ids(self, mode, question, query_embedding, top_k):
        candidate_limit = max(top_k * 8, top_k) if mode == "hybrid" else top_k
        vector_hits = store.vector_search(
            self.user_data_path,
            query_embedding,
            [COLLECTION_ID],
            candidate_limit,
            FASTEMBED_MODEL_KEY,
        )
        if mode == "vector":
            hits = engine.combine_search_hits("vector", vector_hits, [], top_k)
        else:
            keyword_terms = store.keyword_terms_for_query(self.user_data_path, question, [COLLECTION_ID])
            keyword_hits = store.keyword_search(
                self.user_data_path,
                question,
                [COLLECTION_ID],
                candidate_limit,
                keyword_terms,
            )
            hits = engine.combine_search_hits("hybrid", vector_hits, keyword_hits, top_k)
        return [self.chunk_id_by_rowid[rowid] for rowid, _score in hits]

    def case_is_grounded(self, case, hit_ids):
        matched = [chunk_id for chunk_id in case.get("expectedChunkIds", []) if chunk_id in hit_ids]
        if len(matched) < int(case.get("minExpectedHits", 1)):
            return False
        context = normalize_text("\n\n".join(self.chunk_by_id[chunk_id]["text"] for chunk_id in hit_ids))
        for required in case.get("requiredContext", []):
            if normalize_text(required) not in context:
                return False
        for forbidden in case.get("forbiddenContext", []):
            if normalize_text(forbidden) in context:
                return False
        return True

    def test_fastembed_vector_and_hybrid_retrieval(self):
        vector_passed = 0
        hybrid_passed = 0
        query_seconds = 0.0
        failures = []

        for case in self.cases:
            top_k = int(case.get("topK", 8))
            query_text = case_retrieval_query(case)
            self.assertIn(
                query_text,
                self.query_embedding_by_text,
                f"Embedding cache is missing the exact benchmark retrieval query for {case['id']}.",
            )
            query_embedding = self.query_embedding_by_text[query_text]
            start = time.perf_counter()
            vector_ids = self.retrieve_ids("vector", query_text, query_embedding, top_k)
            hybrid_ids = self.retrieve_ids("hybrid", query_text, query_embedding, top_k)
            query_seconds += time.perf_counter() - start

            if self.case_is_grounded(case, vector_ids):
                vector_passed += 1
            if self.case_is_grounded(case, hybrid_ids):
                hybrid_passed += 1
            else:
                failures.append(f"{case['id']}: hybrid hits={hybrid_ids}")

        total = len(self.cases)
        hybrid_recall = hybrid_passed / total
        self.measurements["query_seconds"] = query_seconds
        self.measurements["vector_grounding"] = f"{vector_passed}/{total}"
        self.measurements["hybrid_grounding"] = f"{hybrid_passed}/{total}"

        print(
            "FastEmbed BuzzDB benchmark: "
            f"model={FASTEMBED_MODEL}; chunks={len(self.bundle['chunk_ids'])}; "
            f"load={self.measurements.get('model_load_seconds', 0.0):.2f}s; "
            f"warmup={self.measurements.get('warmup_seconds', 0.0):.2f}s; "
            f"embed={self.measurements.get('passage_embedding_seconds', 0.0):.2f}s; "
            f"query_embed={self.measurements.get('query_embedding_seconds', 0.0):.2f}s; "
            f"cache_load={self.measurements.get('cache_load_seconds', 0.0):.2f}s; "
            f"insert={self.measurements['db_insert_seconds']:.2f}s; "
            f"faiss={self.measurements['faiss_rebuild_seconds']:.2f}s; "
            f"queries={query_seconds:.2f}s; "
            f"vector={vector_passed}/{total}; hybrid={hybrid_passed}/{total}"
        )

        self.assertGreaterEqual(
            hybrid_recall,
            MIN_HYBRID_RECALL,
            "FastEmbed hybrid grounding fell below threshold. " + "; ".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
