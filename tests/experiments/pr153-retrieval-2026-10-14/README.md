# PR #153 retrieval comparison

**The submitted graph search did not improve this benchmark.** On the existing BuzzDB evidence checks, it returned less complete evidence than hybrid at equal source budgets and took longer to search.

| Method | Passed at 4 sources | Passed at 8 sources | Median search at requested K=8 |
| --- | ---: | ---: | ---: |
| Current hybrid | **13/16** | **16/16** | **30 ms** |
| Current vector-only control | 10/16 | 14/16 | 16 ms |
| PR as written | 9/16 | 12/16 | 618 ms |
| PR with array-safe cosine | 9/16 | 13/16 | 843 ms |

The PR returns twice the requested source limit. Equal-budget scores evaluate its first K sources; timing includes computing its entire native result, graph loading, traversal, and HTML visualization. Embedding-service time is excluded. Even with all 16 native sources at requested K=8, both graph variants reach only 15/16. These are evidence-coverage checks, not generated-answer accuracy. The suite has 10 conceptual and 6 multi-evidence questions from one textbook and already serves as a hybrid regression suite; it is not an independent holdout.

## Inspect the recorded run

- [Full results](results.json): original per-question evidence, failures, timings, source hashes, and graph statistics.
- [HTML report](report.html): download and open locally to expand each question's retrieved passages.
- [Original runner](run_original.py): provenance of the original local experiment, retained unchanged; use `run.py` below for reproduction.
- [Dependency versions](environment.json), [input checksums](input-manifest.json), and [frozen source](source/).
- [Replay verification](replay-verification.json): a fresh virtual environment and isolated checkout reproduced all evidence in 128 single-pass searches with zero mismatches.
- [Recorded query vectors](query-embeddings.json) and [passage vectors](passage-embeddings.npz): only embeddings of the public repository fixture and benchmark questions. No student library data is included.

The run used PR revision `10df6433a06bde4fcbaa2977fb1767c82e8139d2` and a baseline working tree based on `79c6bfefa46a9f378525968297cf088e73094cf3`. The baseline included uncommitted changes, so its exact engine, store, and cleaning files are frozen here. The replay imports these snapshots rather than the checkout's application code. It omits only an unrelated collections-command import and worker dispatch registration from the baseline snapshot at import time; all retrieval function definitions remain unchanged.

The original indexed filename is also preserved as inert database metadata. The existing FTS schema indexes the full absolute path, which affects BM25 length normalization and ranking. Without freezing it, changing the checkout location changed some hybrid rankings and the four-source pass count during reproduction testing. The replay reads the actual fixture from the new checkout; it never reads or writes the historical filesystem location. This controls an existing retrieval input without changing the ranking algorithm.

## Reproduce without a model server

Use Python 3.12 and run from the repository root on the commit containing this directory:

```sh
python3.12 -m venv .venv-pr153
.venv-pr153/bin/python -m pip install -r tests/experiments/pr153-retrieval-2026-10-14/requirements.txt
.venv-pr153/bin/python tests/experiments/pr153-retrieval-2026-10-14/run.py
```

On Windows, use `.venv-pr153\Scripts\python.exe` instead of `.venv-pr153/bin/python`. Dependency versions are those recorded on macOS arm64; wheel availability may differ on other platforms.

The default replay needs no Ollama, network, API key, existing index, or ignored local cache. It verifies the frozen input checksums and embedding-cache metadata, builds a fresh isolated SQLite/FAISS index and both graphs, and repeats all 384 searches. It uses the recorded real Nomic vectors; the retrieval algorithms are executed, not mocked. It compares all retrieved passages, scores, and failures with the original evidence and exits unsuccessfully if they differ. Timing is measured anew and will depend on hardware and environment.

Outputs go to `tmp/pr153-replay/`; the original results are never overwritten. Use `--output PATH` to choose another location and `--repeats 1` for a faster evidence replay. The latter runs 128 searches and is not a three-pass latency measurement. Large generated graph files remain under the output directory's `work/` folder.

## Optional fresh embeddings

To re-embed the questions with local Ollama:

```sh
.venv-pr153/bin/python tests/experiments/pr153-retrieval-2026-10-14/run.py --fresh-queries --output tmp/pr153-fresh-queries
```

To re-embed both passages and questions:

```sh
.venv-pr153/bin/python tests/experiments/pr153-retrieval-2026-10-14/run.py --rebuild-embeddings --output tmp/pr153-fresh-all
```

These modes require an already-running **Ollama 0.31.1** with `nomic-embed-text:latest`, digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`. The runner verifies both and never downloads models. `TOKENSMITH_BENCHMARK_OLLAMA_URL` can select another local endpoint. Fresh model output can vary by hardware/runtime; differences from the recorded evidence are reported rather than treated as an offline replay failure.

## Why not merge as-is

1. At eight sources, the PR misses required evidence for the Volcano alias, WAL ordering, 2Q trace, and batching/I/O questions. The array fix recovers only the last of these. Hybrid passes all six multi-evidence questions; the graph variants pass three and four.
2. The original cosine helper rejects stored NumPy arrays, and graph construction suppresses that exception. With array-safe cosine, the actual threshold of 0.1 creates every possible directed edge: **2,475,902 links for 1,574 chunks**. This is not selective expansion on this corpus.
3. Existing indexes without graph files raise `KeyError` instead of falling back. Quality measurements use freshly built graphs and therefore do not count that migration failure.
4. The replacement path skips keyword/hybrid fusion, source-unit expansion, and existing source selection, while adding graph I/O and plotting costs. This benchmark does not isolate which change causes each failure, but the replacement shows no overall benefit here.

The diagnostic variant changes only the cosine helper to support arrays using float64 NumPy reductions. Its formula agrees with the original list implementation within `5.6e-16` on 100 sampled pairs. Thresholds, query rewriting (including the literal missing space before `in detail`), traversal depth, ranking weights, and output limits are preserved. Character-budget sensitivity checks retain the same eight-source scores; they truncate passages in retrieval order and do not emulate the app's token packer.

Keep hybrid as the default. A future experiment should add bounded, selective graph candidates to the existing pipeline, preserve its source budgets and behavior, and demonstrate an evidence gain at acceptable latency on additional datasets.
