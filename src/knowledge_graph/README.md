# Knowledge Graph for Query Difficulty Estimation

This module provides tools for building, benchmarking and analyzing a Knowledge Graph (KG) derived from the textbook content, primarily for estimating query difficulty.

## Usage Commands

All commands should be run from the project root.

### 1. Keyword Extraction (via LLM)
Extract keywords from text chunks using OpenRouter. It will save the results to a JSON file, which can later be used to build the KG (use `JsonExtractor)
```bash
python -m src.knowledge_graph.llm_extract_keywords --api_key <OPENROUTER_API_KEY> --chapter 12 --model qwen/qwen3-next-80b-a3b-instruct
```

### 2. Run the KG Pipeline
Build the knowledge graph from extraction results (links keywords and persists the graph). You can select multiple extraction configurations and methods.
```bash
python -m src.knowledge_graph.run_kg_pipeline
```

### 3. Benchmark Extractors
Compare performance and quality of different keyword extraction algorithms (YAKE, TF-IDF, BERT, SLM, etc.).
```bash
python -m src.knowledge_graph.benchmark_extractors --num_chunks 10
```