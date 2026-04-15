# TokenSmith Feature Implementation Notes

This document tracks feature-specific implementation details without modifying the main project README.
It currently documents the retrieval context chunk cache feature and its evaluation tooling.

## Scope

- Keep existing query embedding cache intact.
- Add retrieval context chunk caching for repeated queries under the same retrieval configuration and corpus state.
- Add runtime profiling and evaluation outputs for cache effectiveness.

## Implemented Components

### 1) Retrieval Context Chunk Cache

- File: `src/retrieval_cache.py`
- Storage: SQLite (`index/cache/retrieval_cache.db`)
- Key inputs:
  - normalized query
  - retrieval configuration signature
  - index fingerprint
- Cached value:
  - selected chunk indices (`topk_idxs`)
  - final ranked chunk texts used for generation
  - ordered retrieval scores

### 2) Pipeline Integration

- CLI/chat path:
  - `src/main.py` (`get_answer`)
  - uses context cache before retriever/ranker/reranker
  - stores result after final rerank on cache miss
- API path:
  - `src/api_server.py` (`_retrieve_and_rank`, `/api/chat`, `/api/chat/stream`)
  - same cache behavior as CLI path

### 3) Config Controls

- `src/config.py`
- `config/config.yaml`

Added fields:
- `enable_context_chunk_cache` (bool)
- `retrieval_cache_path` (str)
- `retrieval_cache_max_entries` (int)

### 4) Runtime and Evaluation Outputs

- Benchmark runtime/profile fields:
  - `tests/test_benchmarks.py`
  - emits `total_runtime_ms` and `retrieval_profile`
- Corpus + experiment runner:
  - `tests/cache_context_corpus.yaml`
  - `tests/utils/cache_context_experiment.py`
- Result output:
  - `tests/results/cache_context_experiment.json`
  - includes only `no_cache_second` and `context_chunk_cache_second`
  - checks cache hit and chunk-id equality between cached vs uncached runs

### 5) Tests Added/Updated

- Added: `tests/test_retrieval_cache.py`
- Updated: `tests/test_api_server.py` (retrieval tuple now includes profile)

## Run Commands

### Context Cache Experiment

```bash
conda run -n tokensmith python tests/utils/cache_context_experiment.py \
  --corpus tests/cache_context_corpus.yaml \
  --config config/config.yaml \
  --index-prefix textbook_index \
  --model-path models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --max-gen-tokens 180 \
  --output tests/results/cache_context_experiment.json
```

### Unit Verification

```bash
conda run -n tokensmith python -m pytest -m unit tests/test_retrieval_cache.py tests/test_api_server.py tests/test_api.py tests/test_end_to_end.py -q
```

## Maintenance Note

I will keep this file updated as feature implementation changes, including when you request design or scope changes.
