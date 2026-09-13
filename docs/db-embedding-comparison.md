# Nomic vs EmbeddingGemma: TokenSmith DB retrieval comparison

Completed 13 September 2026 on an Apple M1 Max with 32 GiB memory, macOS 14.6.1 and Ollama 0.31.1.

**Recommendation: retain Nomic as the default for now.** EmbeddingGemma ranked labeled evidence earlier on this small test, but TokenSmith’s hybrid search found supporting evidence for all 10 questions with either model. EmbeddingGemma added approximately 86 ms to median warm hybrid retrieval. There is no established end-to-end answer-quality advantage in this experiment.

## Main results: recommended retrieval formatting

Both models received the same existing questions and book chunks, with each model’s published question/document prefixes. Accuracy below means retrieval against the repository’s frozen passage labels, not correctness of generated answers.

| Measure | Nomic | EmbeddingGemma |
|---|---:|---:|
| Labeled supporting passage ranked first | 8/10 | 9/10 |
| Labeled supporting passage in first 4, vector only | 9/10 | 10/10 |
| Labeled supporting passage in first 8, vector only | 10/10 | 10/10 |
| Labeled supporting passage in first 4, app hybrid | 10/10 | 10/10 |
| Full benchmark evidence checks pass, first 4 hybrid | 9/10 | 9/10 |
| Full benchmark evidence checks pass, first 8 hybrid | 10/10 | 10/10 |
| Question embedding, median | 36.0 ms | 120.1 ms |
| Question embedding, p95 | 52.4 ms | 150.4 ms |
| Embedding + hybrid top-4 retrieval, median | 63.5 ms | 149.4 ms |
| Embedding + hybrid top-4 retrieval, p95 | 85.9 ms | 177.8 ms |
| Book embedding, 1,574 chunks in batches of 16 | 24.66 s | 44.65 s |
| Raw float32 vector storage | 4.61 MiB | 4.61 MiB |

EmbeddingGemma’s median question embedding request was 3.33× as slow. Median hybrid retrieval was 2.35× as slow, an extra 85.9 ms. These are small absolute delays relative to many answer-generation calls; generation was not measured here.

Both models achieved 80.8% mean labeled-passage recall at vector top-8. With hybrid top-4, both achieved 83.3%. At hybrid top-8, Nomic achieved 91.7% and EmbeddingGemma 88.3%. First-hit success and coverage of all supporting passages measure different things.

## Per-question evidence ranking

Rank of the first labeled supporting passage under vector retrieval; “none” means no labeled passage in the first eight. These are label matches, not manual judgments of every returned passage.

| Question | Nomic rank | EmbeddingGemma rank |
|---|---:|---:|
| Why is page pinning needed in a buffer manager? | 1 | 1 |
| What is sequential flooding and why does it make LRU fail? | 1 | 1 |
| Why is a B+ tree preferred over a binary search tree for disk-based database indexes? | 1 | 1 |
| Why does a hash index struggle with range queries? | 1 | 2 |
| Why does the B+ tree sequence set make range scans efficient? | 1 | 1 |
| How does BuzzDB implement GROUP BY aggregation with a hash table? | 1 | 1 |
| Why does write-ahead logging require write ordering control in the buffer manager? | 1 | 1 |
| What is the difference between OLTP and OLAP workloads? | 1 | 1 |
| Why is a column store faster for analytical queries that read only a few columns? | 3 | 1 |
| Why is the Volcano model inefficient for analytical queries? | 6 | 1 |

For the Volcano-model question, EmbeddingGemma moved the first labeled passage from rank 6 to rank 1. However, it missed another required passage in its first eight vector results; Nomic passed the full evidence check at eight. Hybrid retrieval passed the full check for both at eight, and failed it for both at four. A better first result does not guarantee a more complete evidence set.

**Label limitation found during inspection:** EmbeddingGemma’s rank-1 hash-range result, `ch08.054`, discusses why hashing scatters nearby keys and hurts range queries, but is not in the benchmark’s expected IDs. Its first labeled result is therefore counted at rank 2 despite the relevant earlier passage. Preserve the frozen labels for this comparison; expand and independently review them before claiming broad accuracy improvements.

## Current plain-text formatting versus model-recommended formatting

The current `ollama_embedding` implementation normalizes text and submits it without distinguishing questions from documents. The separate plain-text runs reproduce this input-format choice. Both profiles use a 2,048-token context and disable truncation to detect overflow; all inputs succeeded unchanged.

| Model / formatting | Vector first-result hit | Vector top-4 hit | Vector top-8 hit | Hybrid top-4 hit | Median question embedding |
|---|---:|---:|---:|---:|---:|
| nomic-embed-text / plain | 5/10 | 9/10 | 9/10 | 10/10 | 37.3 ms |
| nomic-embed-text / recommended | 8/10 | 9/10 | 10/10 | 10/10 | 36.0 ms |
| embeddinggemma / plain | 6/10 | 10/10 | 10/10 | 10/10 | 119.3 ms |
| embeddinggemma / recommended | 9/10 | 10/10 | 10/10 | 10/10 | 120.1 ms |

Adding the recommended formatting raised first-result label hits from 5 to 8 for Nomic and from 6 to 9 for EmbeddingGemma. It did not improve every coverage metric: for example, Nomic’s full evidence checks at vector top-4 decreased from 9 to 8. Formatting deserves a wider evaluation before changing defaults, and document-side formatting changes require reindexing.

The formats were `search_query: …` / `search_document: …` for Nomic, and `task: search result | query: …` / `title: none | text: …` for EmbeddingGemma. No additional title metadata was supplied to either model. Sources: [Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card).

## Method and practical limits

- Data: the repository’s BuzzDB book fixture, 1,574 stable chunks, and the 10 questions and expected passage IDs in `tests/benchmarks/buzzdb_grounding_cases.json`. This is a separate local benchmark, not a reproduction of James’s earlier 0.39/0.68 experiment; that experiment’s full question set was unavailable.
- Models: `nomic-embed-text:latest`, 137M, F16; `embeddinggemma:latest`, 307.58M, BF16. Both use 768 dimensions. Exact digests, model metadata, code hashes and input file hashes are saved in the raw results.
- All four indexes were built separately in the experiment output directory. The installed student library, model selection and application source were not changed. EmbeddingGemma was downloaded into local Ollama for this test.
- Accuracy used only the 10 unique questions, scored once per configuration. Five shuffled timing rounds produced 50 live query requests per model/format, 200 total; all rankings remained stable across repetitions. Repetitions do not increase the accuracy sample size to 50.
- Vector-only search uses TokenSmith’s exact normalized FAISS inner-product index. Hybrid uses the current Python `search_library` keyword/vector fusion, source selection, diversification and exercise handling. FAISS used one CPU thread for repeatability. No bridge chunks, graph traversal, generator, question rewrite or answer grading were used.
- Warm latency includes a fresh local HTTP embedding request plus search over a resident prepared index. Hybrid retrieval reused that newly measured query vector, avoiding a second model call. Its reported total sums the embedding and hybrid phases; it excludes the intervening vector-only measurement, UI/IPC, question rewriting and generation. Local database/OS caches were warm. It is not a GUI stopwatch measurement.
- Document embedding time sums real requests in batches of 16; it excludes download, model loading, parsing, database insertion and FAISS construction. This is a throughput comparison, not the app’s full import duration.
- Both embedding models were resident on the GPU. A Gemma chat model was also resident; the benchmark did not call it or claim exclusive machine access. Latency is specific to this machine and load. Median and p95 describe 50 observations per configuration.
- Hit@k: at least one expected chunk among k. Recall@k: per-question fraction of listed expected chunks retrieved, averaged over questions. Full evidence checks additionally require the case’s minimum expected hits, required phrases and absence of forbidden phrases/IDs, evaluated against the full selected chunks. They do not prove that all evidence survives the generator’s later context budget.
- The cases use substantial DB terminology and cover ten topics; they do not adequately test novice paraphrases, numeric reasoning, long documents beyond this book, or multilingual retrieval. Labels are incomplete, as the inspected hash-range example shows.
- The one-question net first-result improvement is not statistically persuasive: paired first-hit outcomes had two Gemma-only wins and one Nomic-only win; exact two-sided McNemar p = 1.0. At top-4 there was one Gemma-only win and no Nomic-only wins, also p = 1.0. This does not establish equivalence either.

A single unloaded-model first query took 294 ms for Nomic and 1,053 ms for EmbeddingGemma. This includes loading and first inference, is not repeated, and does not flush the OS file cache. Treat it as an observation, not a reliable startup benchmark.

## Decision for TokenSmith

Keep Nomic as the default based on this test. Offer EmbeddingGemma as an optional local search model once model-role detection and input formatting are supported. Before choosing a new default, evaluate an independently labeled set of student paraphrases and multi-passage questions, holding the generator fixed. The stronger first-result ranking is promising, while the current hybrid pipeline already closes the first-hit gap on these ten questions.

## Reproduce and inspect

Download both models with Ollama, then use a fresh output directory:

```sh
app_runtime/python/bin/python tests/experiments/compare-db-embedders.py --out tmp/embedding-comparison-new --repeats 5
```

[Benchmark runner](../tests/experiments/compare-db-embedders.py) · [Raw results and all per-query timings](../tmp/embedding-comparison/results.json) · [Run log](../tmp/embedding-comparison/run.log)

Book SHA-256: `98460b2571e08cdec8103213c85b041457f7009536ea82b1d5f509d793ac60d9`

Question/label file SHA-256: `6bb78b5d6edcd96bcae3e0716335a61baf59fc15b22bc95c674259ab909137ec`
