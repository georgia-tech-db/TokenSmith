# Knowledge Graph for Query Difficulty Estimation

This module provides tools for building, benchmarking and analyzing a Knowledge Graph (KG) derived from the textbook content, primarily for estimating query difficulty.

## Usage Commands

All commands should be run from the project root.

### 1. Keyword Extraction (via LLM)
Extract keywords from text chunks using OpenRouter. It will save the results to a JSON file, which can later be used to build the KG (use `JsonExtractor)
```bash
python -m src.knowledge_graph.scripts.llm_extract_keywords --api_key <OPENROUTER_API_KEY> --chapter 12 --model qwen/qwen3-next-80b-a3b-instruct
```

### 2. Run the KG Pipeline
Build the knowledge graph from extraction results (links keywords and persists the graph). You can select multiple extraction configurations and methods.
```bash
python -m src.knowledge_graph.scripts.run_kg_pipeline
```

### 3. Benchmark Extractors
Compare performance and quality of different keyword extraction algorithms (YAKE, TF-IDF, BERT, SLM, etc.).
```bash
python -m src.knowledge_graph.scripts.benchmark_extractors --num_chunks 10
```

### 4. Analyze Query Graph Topology
Analyze a specific query against a generated knowledge graph to estimate its retrieval complexity.
```bash
python -m src.knowledge_graph.scripts.analyze_query --graph data/knowledge_graph/runs/latest/graph.json --query "What is a shared-nothing architecture?"
```

### 5. Inspect a Run
Print graph stats, section tree stats, and cross-signal (KG keyword coverage per section) for a saved run.
```bash
python -m src.knowledge_graph.scripts.inspect_run
python -m src.knowledge_graph.scripts.inspect_run --run data/knowledge_graph/runs/2025-01-01_00-00-00
```

### 6. Generate Canonicalization Cache
Run LLM canonicalization once and persist the result to a JSON cache file. Use `MockCanonicalizer` in subsequent runs to skip re-calling the LLM.
```bash
python -m src.knowledge_graph.scripts.generate_canon_cache
```

### 7. Benchmark Retrievers
Evaluate `KGNodeRetriever`, `SectionTreeRetriever`, and optionally FAISS/BM25 retrievers against a query set. Supports optional LLM relevance grading via OpenRouter.
```bash
python -m src.knowledge_graph.scripts.benchmark_retrieval --queries tests/benchmarks.yaml
python -m src.knowledge_graph.scripts.benchmark_retrieval --queries tests/benchmarks.yaml --no-llm --output results.json
```