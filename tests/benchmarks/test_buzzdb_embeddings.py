import hashlib
import inspect
import json
import os
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
from tests.benchmarks.test_buzzdb_grounding import normalize_text, validate_grounding_case


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/fixtures/buzzdb/buzzdb-book.tokensmith.md"
CASES_PATH = ROOT / "tests/benchmarks/buzzdb_grounding_cases.json"
MODEL = json.loads((ROOT / "tests/benchmarks/embedding-model.json").read_text())
BASE_URL = engine.normalize_ollama_base_url(os.environ.get("TOKENSMITH_BENCHMARK_OLLAMA_URL", ""))
MODEL_SPEC = {"engine": "ollama", "role": "embedder", "ollamaModelName": MODEL["name"], "ollamaBaseUrl": BASE_URL}
MODEL_KEY = engine.ollama_embedding_model_key(MODEL_SPEC)
COLLECTION_ID = "1"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def chunks_sha256(chunks):
    return sha256_bytes(json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode())


def embedding_input_sha256():
    source = inspect.getsource(engine.ollama_embedding) + inspect.getsource(engine.normalize_text)
    return sha256_bytes(f"{source}\n{engine.OLLAMA_EMBEDDING_TEXT_LIMIT}".encode())


def validate_ollama_metadata(version, models):
    if version.get("version") != MODEL["ollamaVersion"]:
        raise AssertionError(f"Benchmark requires Ollama {MODEL['ollamaVersion']}; received {version.get('version')}.")
    installed = next((item for item in models.get("models", []) if item.get("name") == MODEL["name"]), None)
    if installed is None or installed.get("digest") != MODEL["digest"]:
        raise AssertionError(f"Benchmark requires {MODEL['name']} with digest {MODEL['digest']}; model missing or changed.")


def verify_ollama():
    def get(path):
        with urllib.request.urlopen(f"{BASE_URL}/api/{path}", timeout=15) as response:
            return json.load(response)
    validate_ollama_metadata(get("version"), get("tags"))


def all_chunks():
    chunks = engine.chunk_tokensmith_markdown(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not chunks or any(not chunk.get("tokensmithChunkId") for chunk in chunks):
        raise AssertionError("BuzzDB fixture must contain nonempty, identified chunks.")
    return chunks


def compute_embedding_bundle():
    verify_ollama()
    chunks = all_chunks()
    start = time.perf_counter()
    vectors = []
    for index, chunk in enumerate(chunks, 1):
        vectors.append(engine.ollama_embedding(chunk["text"], MODEL_SPEC))
        if index % 64 == 0 or index == len(chunks):
            print(f"Embedded book chunks: {index}/{len(chunks)}", flush=True)
    return {
        "metadata": {
            **cache_metadata(chunks),
            "buildSeconds": time.perf_counter() - start,
        },
        "chunk_ids": [chunk["tokensmithChunkId"] for chunk in chunks],
        "passage_embeddings": np.asarray(vectors, dtype=np.float32),
    }


def write_embedding_bundle(output_path, bundle):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata_json=np.asarray(json.dumps(bundle["metadata"], sort_keys=True)),
                        chunk_ids=np.asarray(bundle["chunk_ids"]), passage_embeddings=bundle["passage_embeddings"])


def load_embedding_bundle():
    path = os.environ.get("TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH")
    if not path:
        return compute_embedding_bundle()
    with np.load(path, allow_pickle=False) as data:
        return {"metadata": json.loads(str(data["metadata_json"])),
                "chunk_ids": [str(value) for value in data["chunk_ids"].tolist()],
                "passage_embeddings": np.asarray(data["passage_embeddings"], dtype=np.float32)}


def cache_metadata(chunks):
    return {"model": MODEL["name"], "modelDigest": MODEL["digest"], "ollamaVersion": MODEL["ollamaVersion"],
                "fixtureSha256": sha256_bytes(FIXTURE_PATH.read_bytes()), "chunksSha256": chunks_sha256(chunks),
                "inputSha256": embedding_input_sha256()}


def validate_bundle(bundle, chunks):
    metadata = bundle["metadata"]
    for key, value in cache_metadata(chunks).items():
        if metadata.get(key) != value:
            raise AssertionError(f"Book embedding cache has stale {key}; rebuild it.")
    if bundle["chunk_ids"] != [chunk["tokensmithChunkId"] for chunk in chunks]:
        raise AssertionError("Embedding cache is missing, duplicating, or reordering book chunks.")
    vectors = bundle["passage_embeddings"]
    if (vectors.shape != (len(chunks), MODEL["dimensions"]) or not np.isfinite(vectors).all()
            or not np.all(np.linalg.norm(vectors, axis=1) > 0)):
        raise AssertionError("Embedding cache contains invalid passage_embeddings.")


class BuzzDBEmbeddingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("TOKENSMITH_RUN_EMBEDDING_BENCHMARK") != "1":
            raise unittest.SkipTest("Live embedding benchmark is opt-in.")
        verify_ollama()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp_dir.cleanup)
        cls.user_data_path = cls.temp_dir.name
        cls.cases = json.loads(CASES_PATH.read_text())
        cls.chunks = all_chunks()
        cls.bundle = load_embedding_bundle()
        validate_bundle(cls.bundle, cls.chunks)
        cls.measurements = {"chunk_count": len(cls.chunks), "fresh_query_embedding_seconds": 0.0}
        cls.index_fixture_from_bundle()

    @classmethod
    def index_fixture_from_bundle(cls):
        start = time.perf_counter()
        store.init_db(cls.user_data_path)
        with store.connect(cls.user_data_path) as conn:
            conn.execute("INSERT INTO collections(id, name, embedding_model) VALUES (?, ?, ?)", (1, "BuzzDBBook", MODEL_KEY))
            conn.execute("INSERT INTO folders(id, path) VALUES (?, ?)", (1, str(FIXTURE_PATH.parent)))
            conn.execute("INSERT INTO collection_items VALUES (?, ?)", (1, 1))
            conn.execute(
                "INSERT INTO tokensmith_collection_state"
                "(collection_id, status, is_active, added_at, file_count, chunk_count, embedding_model_id, embedding_model_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (1, "ready", 1, store.now_iso(), 1, len(cls.chunks), MODEL_KEY, MODEL["name"]),
            )
            document_id = store.insert_document(conn, 1, str(FIXTURE_PATH))
            for chunk, embedding in zip(cls.chunks, cls.bundle["passage_embeddings"]):
                rowid = store.insert_chunk(conn, document_id, {**chunk, "path": str(FIXTURE_PATH), "documentTitle": "BuzzDB Book"})
                blob, _dim = store.vector_to_blob(embedding)
                conn.execute("INSERT INTO embeddings(model, folder_id, chunk_id, embedding) VALUES (?, ?, ?, ?)",
                             (MODEL_KEY, 1, rowid, blob))
        store.rebuild_faiss(cls.user_data_path, MODEL_KEY)
        cls.measurements["index_rebuild_seconds"] = time.perf_counter() - start

    def retrieve_sources(self, question, top_k):
        vector_hit_count = 0
        query_count = 0
        embedding_errors = []
        original_embed = engine.ollama_embedding

        def embed_query(*args, **kwargs):
            nonlocal query_count
            start = time.perf_counter()
            try:
                vector = original_embed(*args, **kwargs)
            except Exception as error:
                embedding_errors.append(str(error))
                raise
            self.measurements["fresh_query_embedding_seconds"] += time.perf_counter() - start
            query_count += 1
            return vector

        def vector_search(*args, **kwargs):
            nonlocal vector_hit_count
            hits = store.vector_search(*args, **kwargs)
            vector_hit_count += len(hits)
            return hits

        # Observe calls without replacing the app's provider, embedding logic, or retrieval stages.
        with patch.object(engine, "ollama_embedding", side_effect=embed_query), \
                patch.object(engine, "vector_search", side_effect=vector_search):
            result = engine.search_library({
                "userDataPath": self.user_data_path, "query": question, "searchMode": "hybrid", "limit": top_k,
                "materials": [{"id": COLLECTION_ID, "status": "ready", "isActive": True}],
                "embeddingModels": [MODEL_SPEC],
            })
        self.assertFalse(embedding_errors, "Query embedding failed: " + "; ".join(embedding_errors))
        self.assertEqual(query_count, 1, "Each retrieval must embed the current question afresh.")
        self.assertGreater(vector_hit_count, 0, "Hybrid benchmark must not silently fall back to keyword-only search.")
        self.assertIsNone(result.get("reason"), result.get("reason"))
        return result

    def test_ollama_hybrid_retrieval(self):
        start = time.perf_counter()
        results = []
        for case in self.cases:
            hits = []
            sources = []
            try:
                response = self.retrieve_sources(case.get("retrievalQuery") or case["question"], int(case.get("topK", 8)))
                sources = response["sources"]
                hits = list(dict.fromkeys(chunk_id for source in sources
                                         for chunk_id in (source.get("sourceChunkIds") or [source["tokensmithChunkId"]])))
                context = normalize_text("\n\n".join(source["context"] for source in sources))
                failures = validate_grounding_case(case, response.get("keywordTerms", []), hits, context)
            except Exception as error:
                failures = [str(error)]
            results.append({
                "id": case["id"], "question": case["question"],
                "referenceAnswer": case["referenceAnswer"],
                "passed": not failures, "hits": hits, "failures": failures,
                "evidenceLabel": "Retrieved evidence (rank order)",
                "evidence": [{"chunkIds": source.get("sourceChunkIds") or [source["tokensmithChunkId"]],
                              "section": source.get("sectionHeader", ""), "context": source["context"]}
                             for source in sources],
            })
        self.measurements["retrieval_seconds"] = time.perf_counter() - start
        passed = sum(result["passed"] for result in results)
        type(self).report = {
            "name": "hybrid_retrieval", "label": "hybrid evidence coverage (app's Ollama Nomic model)",
            "unit": "questions", "passed": passed, "total": len(results), "cases": results,
            "model": MODEL["name"], "modelDigest": MODEL["digest"], "ollamaVersion": MODEL["ollamaVersion"],
            "measurements": self.measurements,
        }
        print(f"Ollama Nomic hybrid benchmark: {passed}/{len(results)} questions; {self.measurements}")
        self.assertGreater(len(results), 0, "No benchmark cases were loaded.")
        self.assertEqual(passed, len(results), "Hybrid evidence regression: " + json.dumps(
            [result for result in results if not result["passed"]]))


if __name__ == "__main__":
    if "--json" in sys.argv:
        result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(BuzzDBEmbeddingBenchmarkTests))
        report = getattr(BuzzDBEmbeddingBenchmarkTests, "report", None)
        if report is None:
            report = {"name": "hybrid_retrieval", "label": "hybrid evidence coverage", "unit": "checks",
                      "passed": 0, "total": 1, "cases": [{"id": "setup", "passed": False,
                      "failures": [str(error) for _, error in result.errors] or ["Benchmark did not run."]}]}
        print(json.dumps(report))
        raise SystemExit(0 if result.wasSuccessful() and not result.skipped and report["passed"] == report["total"] else 1)
    else:
        unittest.main()
