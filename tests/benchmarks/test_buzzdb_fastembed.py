import hashlib
import json
import os
import re
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store
from tests.benchmarks.test_buzzdb_grounding import validate_grounding_case


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "buzzdb" / "buzzdb-book.tokensmith.md"
CASES_PATH = ROOT / "tests" / "benchmarks" / "buzzdb_grounding_cases.json"
COLLECTION_ID = "1"
FASTEMBED_MODEL = os.environ.get("TOKENSMITH_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
FASTEMBED_MODEL_KEY = f"fastembed:{FASTEMBED_MODEL}"
FASTEMBED_BATCH_SIZE = int(os.environ.get("TOKENSMITH_FASTEMBED_BATCH_SIZE", "64"))


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
    kwargs = {"model_name": FASTEMBED_MODEL, "threads": int(os.environ.get("TOKENSMITH_FASTEMBED_THREADS", "2"))}
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


def chunks_sha256(chunks):
    return sha256_bytes(json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode("utf-8"))


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
        "chunksSha256": chunks_sha256(chunks),
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
        cls.addClassCleanup(cls.temp_dir.cleanup)
        cls.user_data_path = cls.temp_dir.name
        cls.cases = load_cases()
        cls.chunks = all_chunks()
        cls.chunk_by_id = {chunk["tokensmithChunkId"]: chunk for chunk in cls.chunks}
        cls.chunk_id_by_rowid = {}
        cls.bundle = load_embedding_bundle()
        cls.measurements = cls.bundle["metadata"].get("measurements", {})
        cls.validate_bundle()
        # Only book vectors are reused. Query inference and all retrieval stages run afresh.
        cls.embedder = instantiate_text_embedding(require_fastembed())
        cls.measurements["fresh_query_embedding_seconds"] = 0.0
        cls.index_fixture_from_bundle()

    @classmethod
    def validate_bundle(cls):
        metadata = cls.bundle["metadata"]
        expected_case_ids = [case["id"] for case in cls.cases]
        expected_query_texts = [case_retrieval_query(case) for case in cls.cases]
        if metadata.get("model") != FASTEMBED_MODEL:
            raise AssertionError(f"Embedding cache model is {metadata.get('model')}; expected {FASTEMBED_MODEL}.")
        if metadata.get("fixtureSha256") != fixture_sha256():
            raise AssertionError("Embedding cache does not match the current BuzzDBBook fixture.")
        if metadata.get("chunksSha256") != chunks_sha256(cls.chunks):
            raise AssertionError("Embedding cache does not match the current parsed chunks; rebuild it.")
        if cls.bundle["chunk_ids"] != [chunk["tokensmithChunkId"] for chunk in cls.chunks]:
            raise AssertionError("Embedding cache is missing, duplicating, or reordering book chunks.")
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
        for key in ("passage_embeddings", "query_embeddings"):
            vectors = cls.bundle[key]
            if vectors.ndim != 2 or not np.isfinite(vectors).all() or not np.all(np.linalg.norm(vectors, axis=1) > 0):
                raise AssertionError(f"Embedding cache contains invalid {key}.")

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

    def retrieve_sources(self, question, top_k):
        def embed_query(text):
            start = time.perf_counter()
            vector = embed_many(self.embedder, "query_embed", [text])[0]
            self.measurements["fresh_query_embedding_seconds"] += time.perf_counter() - start
            return vector

        vector_hit_count = 0

        def vector_search(*args, **kwargs):
            nonlocal vector_hit_count
            hits = store.vector_search(*args, **kwargs)
            vector_hit_count += len(hits)
            return hits

        # Adapt the test-only CPU embedder; ranking, FTS, FAISS and source selection are production code.
        with patch.object(engine, "resolve_embedding_provider_for_key", return_value=(embed_query, None)), \
                patch.object(engine, "vector_search", side_effect=vector_search):
            result = engine.search_library({
                "userDataPath": self.user_data_path,
                "query": question,
                "searchMode": "hybrid",
                "limit": top_k,
                "materials": [{"id": COLLECTION_ID, "status": "ready", "isActive": True}],
            })
        self.assertGreater(vector_hit_count, 0, "Hybrid benchmark must not silently fall back to keyword-only search.")
        self.assertIsNone(result.get("reason"), result.get("reason"))
        return result

    def test_fastembed_hybrid_retrieval(self):
        query_seconds = 0.0
        results = []

        for case in self.cases:
            top_k = int(case.get("topK", 8))
            query_text = case_retrieval_query(case)
            start = time.perf_counter()
            hits = []
            try:
                response = self.retrieve_sources(query_text, top_k)
                sources = response["sources"]
                hits = list(dict.fromkeys(
                    chunk_id for source in sources
                    for chunk_id in (source.get("sourceChunkIds") or [source["tokensmithChunkId"]])
                ))
                context = normalize_text("\n\n".join(source["context"] for source in sources))
                failures = validate_grounding_case(case, response.get("keywordTerms", []), hits, context)
            except Exception as error:
                failures = [str(error)]
            query_seconds += time.perf_counter() - start
            results.append({"id": case["id"], "passed": not failures, "hits": hits, "failures": failures})

        total = len(self.cases)
        hybrid_passed = sum(result["passed"] for result in results)
        self.measurements["query_seconds"] = query_seconds
        self.measurements["hybrid_grounding"] = f"{hybrid_passed}/{total}"
        type(self).report = {
            "name": "hybrid_retrieval", "label": "hybrid evidence coverage (real CPU embeddings)",
            "unit": "questions", "passed": hybrid_passed, "total": total,
            "model": FASTEMBED_MODEL, "cases": results, "measurements": self.measurements,
        }

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
            f"fresh_query_embed={self.measurements['fresh_query_embedding_seconds']:.2f}s; "
            f"hybrid={hybrid_passed}/{total}"
        )

        self.assertGreater(total, 0, "No benchmark cases were loaded.")
        self.assertEqual(
            hybrid_passed, total,
            "Hybrid evidence regression: " + json.dumps([result for result in results if not result["passed"]]),
        )


if __name__ == "__main__":
    if "--json" in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(BuzzDBFastEmbedBenchmarkTests)
        result = unittest.TextTestRunner().run(suite)
        report = getattr(BuzzDBFastEmbedBenchmarkTests, "report", None)
        if report is None:
            report = {"name": "hybrid_retrieval", "label": "hybrid evidence coverage", "unit": "checks",
                      "passed": 0, "total": 1, "cases": [{"id": "setup", "passed": False,
                      "failures": [str(error) for _, error in result.errors] or ["Benchmark did not run."]}]}
        print(json.dumps(report))
        raise SystemExit(0 if result.wasSuccessful() and not result.skipped and report["passed"] == report["total"] else 1)
    else:
        unittest.main()
