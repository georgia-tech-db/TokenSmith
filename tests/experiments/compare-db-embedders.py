"""Opt-in local embedding benchmark; never reads or changes the student library.

Run with app_runtime/python/bin/python tests/experiments/compare-db-embedders.py
Requires already downloaded nomic-embed-text and embeddinggemma in local Ollama.
"""
import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store

FIXTURE = ROOT / "tests/fixtures/buzzdb/buzzdb-book.tokensmith.md"
CASES = ROOT / "tests/benchmarks/buzzdb_grounding_cases.json"
MODELS = ["nomic-embed-text:latest", "embeddinggemma:latest"]
PROFILES = ["recommended", "plain"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api(route, payload=None):
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/" + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
    return result, (time.perf_counter() - start) * 1000


def formatted(model, profile, text, kind):
    text = engine.normalize_text(text)
    if profile == "plain":
        return text
    if model.startswith("nomic"):
        return ("search_query: " if kind == "query" else "search_document: ") + text
    return ("task: search result | query: " if kind == "query" else "title: none | text: ") + text


def embed(model, texts):
    response, elapsed = api("embed", {
        "model": model, "input": texts, "truncate": False,
        "options": {"num_ctx": 2048}, "keep_alive": "30m",
    })
    vectors = np.asarray(response["embeddings"], dtype=np.float32)
    assert vectors.shape == (len(texts), 768), vectors.shape
    assert np.isfinite(vectors).all() and (np.linalg.norm(vectors, axis=1) > 0).all()
    return vectors, {
        "wall_ms": elapsed,
        "server_ms": response.get("total_duration", 0) / 1e6,
        "load_ms": response.get("load_duration", 0) / 1e6,
        "tokens": response.get("prompt_eval_count"),
    }


def distribution(values):
    return {"n": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "mean": float(np.mean(values)),
            "min": float(min(values)), "max": float(max(values))}


def metrics(case, ids, chunk_by_id):
    expected = set(case["expectedChunkIds"])
    matches = expected.intersection(ids)
    first = next((i + 1 for i, chunk_id in enumerate(ids) if chunk_id in expected), None)
    normalize = lambda s: " ".join(s.casefold().replace("*", "").replace("`", "").split())
    context = normalize("\n\n".join(chunk_by_id[x]["text"] for x in ids))
    missing_context = [x for x in case.get("requiredContext", []) if normalize(x) not in context]
    forbidden_context = [x for x in case.get("forbiddenContext", []) if normalize(x) in context]
    forbidden_ids = set(case.get("forbiddenChunkIds", [])).intersection(ids)
    minimum_pass = len(matches) >= case.get("minExpectedHits", 1)
    return {"ids": ids, "matched": sorted(matches), "first_relevant_rank": first,
            "hit": bool(matches), "recall": len(matches) / len(expected),
            "mrr": 1 / first if first else 0, "minimum_evidence_pass": minimum_pass,
            "strict_grounding_pass": minimum_pass and not missing_context and not forbidden_context and not forbidden_ids,
            "missing_context": missing_context, "forbidden_context": forbidden_context,
            "forbidden_ids": sorted(forbidden_ids)}


def build_index(directory, model, chunks, vectors):
    directory.mkdir(parents=True, exist_ok=True)
    # A fresh benchmark-only directory is mandatory: avoid duplicate rows on reruns.
    if (directory / "tokensmith.sqlite").exists():
        return
    store.init_db(str(directory))
    key = "ollama:" + model
    with store.connect(str(directory)) as conn:
        conn.execute("INSERT INTO collections(id,name,embedding_model) VALUES (1,?,?)", ("BuzzDBBook", key))
        conn.execute("INSERT INTO folders(id,path) VALUES (1,?)", (str(FIXTURE.parent),))
        conn.execute("INSERT INTO collection_items VALUES (1,1)")
        conn.execute("INSERT INTO tokensmith_collection_state(collection_id,status,is_active,added_at,file_count,chunk_count) VALUES (1,'ready',1,?,1,?)", (store.now_iso(), len(chunks)))
        doc = store.insert_document(conn, 1, str(FIXTURE))
        for chunk, vector in zip(chunks, vectors):
            row = store.insert_chunk(conn, doc, {**chunk, "path": str(FIXTURE), "documentTitle": "BuzzDB Book"})
            blob, _ = store.vector_to_blob(vector)
            conn.execute("INSERT INTO embeddings(model,folder_id,chunk_id,embedding) VALUES (?,1,?,?)", (key, row, blob))
    store.rebuild_faiss(str(directory), key)


def retrieve(directory, model, query, vector, chunks):
    ids_by_row = {i + 1: c["tokensmithChunkId"] for i, c in enumerate(chunks)}
    timings, ids = {}, {}
    start = time.perf_counter()
    hits = store.vector_search(str(directory), vector, ["1"], 8, "ollama:" + model)
    assert len(hits) == 8, "Vector search must succeed; no silent keyword fallback."
    timings["vector8_ms"] = (time.perf_counter() - start) * 1000
    for k in (1, 4, 8):
        ids[f"vector{k}"] = [ids_by_row[row] for row, _ in hits[:k]]
    with patch.object(engine, "resolve_embedding_provider_for_key", return_value=(lambda _: vector, None)):
        for k in (4, 8):
            start = time.perf_counter()
            result = engine.search_library({"userDataPath": str(directory), "query": query, "limit": k,
                "searchMode": "hybrid", "materials": [{"id": "1", "status": "ready", "isActive": True}]})
            timings[f"hybrid{k}_ms"] = (time.perf_counter() - start) * 1000
            assert not result["reason"]
            ids[f"hybrid{k}"] = [ids_by_row[source["chunkRowid"]] for source in result["sources"]]
    return ids, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/embedding-comparison")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = engine.chunk_tokensmith_markdown(FIXTURE.read_text())
    chunks = [c for c in chunks if c.get("tokensmithChunkId")]
    by_id = {c["tokensmithChunkId"]: c for c in chunks}
    cases = json.loads(CASES.read_text())
    assert len(by_id) == len(chunks)
    for case in cases:
        assert set(case["expectedChunkIds"]).issubset(by_id)
    assert store.faiss is not None, "FAISS is required for this benchmark."
    # The production index is exact inner product over normalized vectors.
    store.faiss.omp_set_num_threads(1)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "fixture_sha256": sha(FIXTURE),
        "cases_sha256": sha(CASES), "chunk_count": len(chunks), "question_count": len(cases),
        "cases": cases, "repeats": args.repeats, "context_tokens": 2048, "truncate": False,
        "platform": platform.platform(), "ollama_version": api("version")[0],
        "initial_loaded_models": api("ps")[0], "models": {}, "configurations": {},
        "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / "python_engine/tokensmith_engine.py", ROOT / "python_engine/tokensmith_store.py"]},
        "method": "Real local Ollama embeddings; identical existing questions/chunks; exact cosine top-k and production hybrid source selection. Hybrid timed after live query embedding using that vector, avoiding duplicate model calls. No generator or question rewriting. Plain mirrors current text formatting; recommended uses each model's published retrieval prefixes. 5 shuffled rounds by default, alternating model order. Cold start means model unloaded, not OS file cache flushed."}
    for name in ("machdep.cpu.brand_string", "hw.memsize"):
        report[name] = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    def save():
        (args.out / "results.json").write_text(json.dumps(report, indent=2))
    for model in MODELS:
        report["models"][model] = api("show", {"model": model})[0]
    report["model_tags"] = api("tags")[0]
    save()
    configurations = []
    for model in MODELS:
        # Explicit unload isolates model loading from warm query measurements.
        api("embed", {"model": model, "input": [], "keep_alive": 0})
        _, cold = embed(model, [formatted(model, "recommended", "How does a database manage memory?", "query")])
        report["models"][model]["cold_first_query"] = cold
        for profile in PROFILES:
            label = model.replace(":", "-") + "-" + profile
            config = {"model": model, "profile": profile, "queries": [], "document_batches": []}
            report["configurations"][label] = config
            directory = args.out / label
            cache = args.out / (label + ".npz")
            signature = hashlib.sha256((sha(FIXTURE) + model + profile +
                json.dumps(report["models"][model].get("model_info", {}), sort_keys=True)).encode()).hexdigest()
            if cache.exists():
                bundle = np.load(cache, allow_pickle=False)
                assert str(bundle["signature"]) == signature, "Stale embedding cache"
                vectors = bundle["vectors"]
                config["document_batches"] = json.loads(str(bundle["timings"]))
                config["document_cache_reused"] = True
            else:
                blocks = []
                for offset in range(0, len(chunks), 16):
                    batch = chunks[offset:offset + 16]
                    block, timing = embed(model, [formatted(model, profile, c["text"], "document") for c in batch])
                    blocks.append(block)
                    config["document_batches"].append({"offset": offset, "count": len(batch), **timing})
                    if offset % 160 == 0:
                        print(f"Indexing {label}: {min(offset + 16, len(chunks))}/{len(chunks)}", flush=True)
                        save()
                vectors = np.concatenate(blocks)
                np.savez_compressed(cache, vectors=vectors, signature=signature, timings=json.dumps(config["document_batches"]))
            assert vectors.shape == (len(chunks), 768)
            start = time.perf_counter()
            build_index(directory, model, chunks, vectors)
            config["db_and_faiss_build_ms"] = (time.perf_counter() - start) * 1000
            config["document_embedding_seconds"] = sum(x["wall_ms"] for x in config["document_batches"]) / 1000
            config["vector_bytes"] = vectors.nbytes
            configurations.append((label, model, profile, directory))
            save()
            print(f"Ready {label}: {config['document_embedding_seconds']:.1f}s embedding", flush=True)
    report["loaded_models_before_queries"] = api("ps")[0]
    for label, model, profile, directory in configurations:
        for _ in range(2):
            vector, _ = embed(model, [formatted(model, profile, "How does a database manage memory?", "query")])
            retrieve(directory, model, "How does a database manage memory?", vector[0], chunks)
    rng = random.Random(20260913)
    for repetition in range(args.repeats):
        order = configurations if repetition % 2 == 0 else list(reversed(configurations))
        shuffled = list(cases)
        rng.shuffle(shuffled)
        for label, model, profile, directory in order:
            for case in shuffled:
                query = case.get("retrievalQuery") or case["question"]
                vector, timing = embed(model, [formatted(model, profile, query, "query")])
                ids, retrieval_timing = retrieve(directory, model, query, vector[0], chunks)
                report["configurations"][label]["queries"].append({"case_id": case["id"], "repeat": repetition,
                    "embedding": timing, "retrieval": retrieval_timing,
                    "metrics": {mode: metrics(case, hit_ids, by_id) for mode, hit_ids in ids.items()}})
            save()
            print(f"Queries {label}: round {repetition + 1}/{args.repeats}", flush=True)
    for label, config in report["configurations"].items():
        samples = config["queries"]
        first = [s for s in samples if s["repeat"] == 0]
        config["summary"] = {
            "embedding_ms": distribution([s["embedding"]["wall_ms"] for s in samples]),
            "vector8_total_ms": distribution([s["embedding"]["wall_ms"] + s["retrieval"]["vector8_ms"] for s in samples]),
            "hybrid4_total_ms": distribution([s["embedding"]["wall_ms"] + s["retrieval"]["hybrid4_ms"] for s in samples]),
            "hybrid8_total_ms": distribution([s["embedding"]["wall_ms"] + s["retrieval"]["hybrid8_ms"] for s in samples]),
            "accuracy": {mode: {metric: float(np.mean([s["metrics"][mode][metric] for s in first]))
                for metric in ("hit", "recall", "mrr", "minimum_evidence_pass", "strict_grounding_pass")}
                for mode in first[0]["metrics"]},
            "rankings_stable_across_repeats": all(s["metrics"] == next(f["metrics"] for f in first if f["case_id"] == s["case_id"]) for s in samples),
        }
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["loaded_models_after_queries"] = api("ps")[0]
    save()
    print(json.dumps({label: c["summary"] for label, c in report["configurations"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
