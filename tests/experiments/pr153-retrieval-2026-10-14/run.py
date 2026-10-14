"""Paired retrieval experiment for PR 153. No student library or app settings are used.

Run from repo root: python tests/experiments/pr153-retrieval-2026-10-14/run.py
Default: offline replay using frozen engine snapshots and recorded embeddings.
Use --fresh-queries for local Ollama, or --rebuild-embeddings for fresh passages too.
See README.md. Outputs default to tmp/pr153-replay; original results stay untouched.
"""
from __future__ import annotations

import contextlib
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL_INDEXED_PATH = '/Users/aj/git/TokenSmith/tests/fixtures/buzzdb/buzzdb-book.tokensmith.md'
sys.path.insert(0, str(ROOT))


def load_module(name, path, *, omit_worker_registration=False):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if omit_worker_registration:
        # The frozen working tree registers unrelated, then-uncommitted collection
        # commands. Omit only that import and dispatch table, retaining every
        # retrieval function and its original source/line numbers verbatim.
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Try) and any(
                isinstance(child, ast.Import) and any(alias.name == 'tokensmith_collections' for alias in child.names)
                for child in ast.walk(node))) and not (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COMMANDS' for t in node.targets))]
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


def verify_inputs():
    manifest = json.loads((HERE / 'input-manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen benchmark input changed or is missing: {relative}. Check out the published benchmark commit.')


verify_inputs()
import python_engine
store = load_module('python_engine.tokensmith_store', HERE / 'source/current/tokensmith_store.py')
python_engine.tokensmith_store = store
sys.modules['tokensmith_store'] = store
load_module('tokensmith_cleaning', HERE / 'source/current/tokensmith_cleaning.py')
current = load_module('python_engine.tokensmith_engine', HERE / 'source/current/tokensmith_engine.py', omit_worker_registration=True)
python_engine.tokensmith_engine = current
from tests.benchmarks import test_buzzdb_embeddings as benchmark
from tests.benchmarks.test_buzzdb_embeddings import BuzzDBEmbeddingBenchmarkTests as Fixture, MODEL_SPEC, MODEL_KEY
from tests.benchmarks.test_buzzdb_grounding import normalize_text, validate_grounding_case


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return {'n': len(values), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'mean': statistics.mean(values)}


def repaired_cosine(a, b):
    """Same cosine formula, array-safe; float64 NumPy reductions avoid slow Python loops."""
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator > 1e-12 else 0.0


def ids_for(sources):
    return list(dict.fromkeys(cid for source in sources for cid in
                (source.get('sourceChunkIds') or [source.get('tokensmithChunkId')]) if cid))


def assess(case, response, sources):
    hits = ids_for(sources)
    context = normalize_text('\n\n'.join(s['context'] for s in sources))
    failures = validate_grounding_case(case, response.get('keywordTerms', []), hits, context)
    # Coverage of annotated groups (alternatives count once), not exhaustive relevance labels.
    groups = [set(g['chunkIds']) for g in case.get('evidenceGroups', [])]
    if not groups:
        groups = [{cid} for cid in case.get('expectedChunkIds', [])]
    found = sum(bool(g.intersection(hits)) for g in groups)
    gold = set.union(*groups) if groups else set()
    first = next((rank for rank, s in enumerate(sources, 1) if gold.intersection(ids_for([s]))), None)
    return {'passed': not failures, 'failures': failures, 'hits': hits,
            'annotated_groups_found': found, 'annotated_groups_total': len(groups),
            'annotated_group_recall': found / len(groups) if groups else None,
            'reciprocal_first_relevant_rank': 1 / first if first else 0,
            'source_count': len(sources), 'context_chars': sum(len(s['context']) for s in sources),
            'context_words': sum(len(s['context'].split()) for s in sources),
            'evidence': [{'chunkIds': ids_for([s]), 'section': s.get('sectionHeader'),
                          'context': s['context'], 'retrievalMode': s.get('retrievalMode')}
                         for s in sources]}


def clip_chars(sources, budget):
    clipped = []
    for source in sources:
        if budget <= 0:
            break
        text = source['context'][:budget]
        clipped.append({**source, 'context': text})
        budget -= len(text)
    return clipped


def graph_stats(path):
    graph = json.loads((path / '1.json').read_text())
    degree = [len(v) for v in graph.values()]
    n, edges = len(graph), sum(degree)
    return {'nodes': n, 'directed_edges': edges, 'density': edges / (n * (n - 1)),
            'out_degree': distribution(degree),
            'disk_bytes': sum(p.stat().st_size for p in path.glob('*.json'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/pr153-replay')
    parser.add_argument('--embedding-cache', type=Path, default=HERE / 'passage-embeddings.npz')
    parser.add_argument('--fresh-queries', action='store_true', help='Compute new queries with pinned local Ollama/model.')
    parser.add_argument('--rebuild-embeddings', action='store_true', help='Recompute passage and query vectors with pinned local Ollama/model.')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    output = args.output.resolve()
    if output == HERE or HERE in output.parents:
        parser.error('--output must be outside the published benchmark directory')
    WORK = output / 'work'
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ['TOKENSMITH_RUN_EMBEDDING_BENCHMARK'] = '1'
    cache_path = args.embedding_cache.resolve()
    live = args.fresh_queries or args.rebuild_embeddings
    if live:
        benchmark.verify_ollama()
    if args.rebuild_embeddings:
        cache_path = output / 'passage-embeddings.npz'
        benchmark.write_embedding_bundle(cache_path, benchmark.compute_embedding_bundle())
    os.environ['TOKENSMITH_BENCHMARK_EMBEDDINGS_PATH'] = str(cache_path)
    # Offline mode validates all vectors and cache provenance but does not contact
    # a model server: those vectors were already recorded with the pinned model.
    original_insert_chunk = store.insert_chunk
    original_insert_document = store.insert_document

    def insert_frozen_chunk(conn, document_id, chunk):
        # The existing FTS schema indexes the full filename. Preserve this input
        # as inert metadata so a different checkout path cannot change BM25.
        return original_insert_chunk(conn, document_id, {**chunk, 'path': ORIGINAL_INDEXED_PATH})

    def insert_frozen_document(conn, folder_id, path):
        return original_insert_document(conn, folder_id, ORIGINAL_INDEXED_PATH)

    with (contextlib.nullcontext() if live else patch.object(benchmark, 'verify_ollama', return_value=None)), \
         patch.object(store, 'insert_chunk', side_effect=insert_frozen_chunk), \
         patch.object(store, 'insert_document', side_effect=insert_frozen_document):
        Fixture.setUpClass()
    profile = Path(Fixture.user_data_path)
    # Full PR module uses its own exact store; the fixture is identical for all methods.
    pr_store = load_module('tokensmith_store', HERE / 'source/pr/tokensmith_store.py')
    pr = load_module('pr153_engine', HERE / 'source/pr/tokensmith_engine.py')
    original_cosine = pr.embedding_similarity
    chunks = pr_store.get_chunks_by_material_id(str(profile), '1')
    assert len(chunks) == len(Fixture.chunks)
    assert isinstance(chunks[0]['embedding'], np.ndarray)
    files = [HERE / 'source/current/tokensmith_engine.py', HERE / 'source/current/tokensmith_store.py',
             HERE / 'source/current/tokensmith_cleaning.py',
             ROOT / 'tests/benchmarks/buzzdb_grounding_cases.json',
             ROOT / 'tests/benchmarks/buzzdb_reasoning_cases.json',
             ROOT / 'tests/fixtures/buzzdb/buzzdb-book.tokensmith.md',
             HERE / 'source/pr/tokensmith_engine.py', HERE / 'source/pr/tokensmith_store.py']
    report = {'metadata': {'pr': 153, 'pr_head': '10df6433a06bde4fcbaa2977fb1767c82e8139d2',
        'current_head': '79c6bfefa46a9f378525968297cf088e73094cf3',
        'checkout_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'current_working_tree_used': False, 'baseline_source': 'Frozen working-tree snapshot used for the original 2026-10-14 run, based on 79c6bfefa46a9f378525968297cf088e73094cf3; only unrelated worker command registration omitted at import.',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'cache': Fixture.bundle['metadata'], 'chunk_count': len(chunks), 'case_count': len(Fixture.cases),
        'model_spec': MODEL_SPEC, 'run_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'indexed_path_metadata': ORIGINAL_INDEXED_PATH,
        'replay_mode': 'fresh queries' if live else 'offline recorded vectors',
        'query_embeddings': 'Fresh Nomic vectors computed once per exact string.' if live else 'Recorded original query vectors; no model service calls.',
        'timing': f'{args.repeats} interleaved passes after baseline-index warm-up; query embedding service time excluded. PR graph loading, traversal, and HTML plotting included; first graph call may include lazy plotting setup.',
        'metric': 'Existing strict evidence checks; not generated-answer accuracy or exhaustive relevance judgments.',
        'array_repair': inspect.getsource(repaired_cosine),
        'source_budget': 'PR native 2K output retained, plus first K and first K clipped to each hybrid result character budget.'},
        'preflight': {}, 'graph_builds': {}, 'results': [], 'query_embedding_measurements': []}
    write_json(output / 'results.json', report)
    base_payload = {'userDataPath': str(profile), 'searchMode': 'hybrid',
        'materials': [{'id': '1', 'status': 'ready', 'isActive': True}],
        'documents': store.study_documents(str(profile)), 'embeddingModels': [MODEL_SPEC],
        'embeddingGpuEnabled': True}

    # Compute real query embeddings before timing the search algorithms.
    query_vectors = {} if live else json.loads((HERE / 'query-embeddings.json').read_text(encoding='utf-8'))
    for case in Fixture.cases:
        query = case.get('retrievalQuery') or case['question']
        for text in [query, pr.clean_query(query)]:
            if text in query_vectors:
                vector = np.asarray(query_vectors[text])
                assert vector.shape == (768,) and np.isfinite(vector).all() and np.linalg.norm(vector) > 0
                continue
            if not live:
                raise RuntimeError(f'Missing recorded query vector: {text}')
            start = time.perf_counter()
            query_vectors[text] = current.ollama_embedding(text, MODEL_SPEC)
            report['query_embedding_measurements'].append({'query': text, 'seconds': time.perf_counter() - start})
        print('Query ready:', case['id'], flush=True)
    write_json(output / 'query-embeddings.json', query_vectors)

    def embed(text, *args, **kwargs):
        return query_vectors[text]

    with patch.object(current, 'ollama_embedding', side_effect=embed), \
         patch.object(pr, 'ollama_embedding', side_effect=embed):
        query = Fixture.cases[0]['question']
        try:
            pr.search_library_with_similarity_graph({**base_payload, 'query': query, 'limit': 8})
            report['preflight']['existing_index_without_graph'] = 'unexpected success'
        except Exception as error:
            report['preflight']['existing_index_without_graph'] = f'{type(error).__name__}: {error}'
        assert store.vector_search(str(profile), query_vectors[query], ['1'], 8, MODEL_KEY)

        for name, cosine in [('pr_as_written', original_cosine), ('pr_array_repaired', repaired_cosine)]:
            pr.embedding_similarity = cosine
            start = time.perf_counter()
            print('Building graph:', name, flush=True)
            pr.build_utility_graph_and_store(str(profile), '1', chunks)
            elapsed = time.perf_counter() - start
            destination = WORK / name
            shutil.copytree(profile / 'similarity_graphs', destination, dirs_exist_ok=True)
            report['graph_builds'][name] = {**graph_stats(destination), 'seconds': elapsed}
            print('Graph built:', name, report['graph_builds'][name], flush=True)
            write_json(output / 'results.json', report)

        # Validate the repaired helper against the original on ordinary lists.
        errors = [abs(repaired_cosine(chunks[i]['embedding'], chunks[i+1]['embedding']) -
                      original_cosine(chunks[i]['embedding'].tolist(), chunks[i+1]['embedding'].tolist()))
                  for i in range(min(100, len(chunks)-1))]
        report['preflight']['repaired_cosine_max_abs_error_100_pairs'] = max(errors)

        # Warm FAISS and keyword caches. Plot paths are confined to disposable output.
        os.chdir(WORK)
        with open(output / 'runtime.log', 'w', encoding='utf-8') as log, contextlib.redirect_stderr(log):
            current.search_library({**base_payload, 'query': query, 'limit': 8})
            for k in [8, 4]:
                entries = {}
                for pass_index in range(args.repeats):
                    order = ['hybrid', 'vector', 'pr_as_written', 'pr_array_repaired']
                    random.Random(153 + pass_index).shuffle(order)
                    for name in order:
                        if name.startswith('pr_'):
                            shutil.copytree(WORK / name, profile / 'similarity_graphs', dirs_exist_ok=True)
                            pr.embedding_similarity = original_cosine if name == 'pr_as_written' else repaired_cosine
                        for case in Fixture.cases:
                            query = case.get('retrievalQuery') or case['question']
                            payload = {**base_payload, 'query': query, 'limit': k,
                                       'searchMode': 'vector' if name == 'vector' else 'hybrid'}
                            start = time.perf_counter()
                            try:
                                response = (pr.search_library_with_similarity_graph(payload) if name.startswith('pr_')
                                            else current.search_library(payload))
                                assert response.get('reason') is None, response.get('reason')
                                error = None
                            except Exception:
                                response = {'sources': []}
                                error = traceback.format_exc()
                            seconds = time.perf_counter() - start
                            key = (name, case['id'])
                            if key not in entries:
                                entries[key] = {'method': name, 'k': k, 'id': case['id'],
                                    'question': case['question'], 'query': query,
                                    'actual_query': pr.clean_query(query) if name.startswith('pr_') else query,
                                    'category': 'multi_evidence' if case.get('evidenceGroups') else 'conceptual',
                                    'seconds': [], 'error': error, 'native': assess(case, response, response['sources']),
                                    'equal_k': assess(case, response, response['sources'][:k]),
                                    '_response': response}
                            else:
                                assert entries[key]['native']['hits'] == ids_for(response['sources']), 'Unstable retrieval order'
                            entries[key]['seconds'].append(seconds)
                        print(f'Finished K={k} pass={pass_index+1} method={name}', flush=True)
                for (name, case_id), entry in entries.items():
                    case = next(c for c in Fixture.cases if c['id'] == case_id)
                    budget = entries[('hybrid', case_id)]['equal_k']['context_chars']
                    entry['hybrid_context_char_budget'] = budget
                    entry['equal_k_and_chars'] = assess(case, entry['_response'], clip_chars(entry['_response']['sources'][:k], budget))
                    del entry['_response']
                    report['results'].append(entry)
                write_json(output / 'results.json', report)
    report['summary'] = []
    for k in [8, 4]:
        for name in ['hybrid', 'vector', 'pr_as_written', 'pr_array_repaired']:
            rows = [r for r in report['results'] if r['method'] == name and r['k'] == k]
            for budget in ['native', 'equal_k', 'equal_k_and_chars']:
                report['summary'].append({'method': name, 'k': k, 'budget': budget,
                    'passed': sum(r[budget]['passed'] for r in rows), 'total': len(rows),
                    'conceptual_passed': sum(r[budget]['passed'] for r in rows if r['category'] == 'conceptual'),
                    'multi_evidence_passed': sum(r[budget]['passed'] for r in rows if r['category'] == 'multi_evidence'),
                    'annotated_group_recall': statistics.mean(r[budget]['annotated_group_recall'] for r in rows),
                    'mean_context_chars': statistics.mean(r[budget]['context_chars'] for r in rows),
                    'mean_source_count': statistics.mean(r[budget]['source_count'] for r in rows),
                    'search_seconds': distribution([t for r in rows for t in r['seconds']]),
                    'errors': sum(bool(r['error']) for r in rows)})
    report['verification'] = {
        'successful_search_calls': sum(len(r['seconds']) for r in report['results'] if not r['error']),
        'retrieved_ids_stable_across_repeats': True,
    }
    original = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    expected = {(r['method'], r['k'], r['id']): r for r in original['results']}
    mismatches = []
    for row in report['results']:
        old = expected[(row['method'], row['k'], row['id'])]
        for budget in ['native', 'equal_k', 'equal_k_and_chars']:
            if row[budget] != old[budget]:
                mismatches.append({'method': row['method'], 'k': row['k'], 'id': row['id'], 'budget': budget})
    report['verification']['original_evidence_mismatches'] = mismatches
    write_json(output / 'results.json', report)
    Fixture.doClassCleanups()
    for row in report['summary']:
        if row['budget'] == 'equal_k':
            print(json.dumps(row), flush=True)
    print(f'Results: {output / "results.json"}; original evidence mismatches: {len(mismatches)}', flush=True)
    if any(row['error'] for row in report['results']) or (mismatches and not live):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
