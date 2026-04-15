# Knowledge Graph

This module builds a **keyword graph** from textbook content. It extracts keywords from document chunks and connects them based on some condition (e.g. co-occurrence, TODO/semantic relationship), producing a graph that captures the relationships between concepts across the corpus.

The build pipeline is independent from the main RAG system and is designed to be run as a standalone tool. Its outputs (graph artifacts) can be used for corpus analysis, entity linking and graph-based retrieval.

---

## How It Works

```
Chunks (from index_builder)
        │
        ▼
   [Extractor]  ──  extracts keywords from each chunk
        │
        ▼
   [Linker]     ──  connects keyword pairs based on some criteria (e.g. that co-occur in the same chunk)
        │
        ▼
   JSON artifacts  ──  graph.json, chunks.json, run_metadata.json
```

1. **Extraction**: Each chunk is processed by an extractor that produces a list of keywords (e.g., `["B+ tree", "indexing", "query optimizer"]`).
2. **Linking**: Link pairs of keywords based on some criteria.
3. **Persistence**: The final graph and metadata are saved as JSON in a timestamped run directory.

---

## Running the Pipeline

```bash
python -m src.knowledge_graph.run_kg_pipeline
```

Or with a custom config:

```bash
python -m src.knowledge_graph.run_kg_pipeline --config path/to/config.yaml
```

> Use the `kg_env` conda environment, not `tokensmith`.
> ```bash
> conda activate kg_env
> python -m src.knowledge_graph.run_kg_pipeline
> ```

Each run creates a timestamped directory and updates a `latest/` symlink:

```
data/knowledge_graph/runs/
├── 2025-06-10_14-32-01/
│   ├── input/
│   │   ├── chunks.pkl        (symlink)
│   │   ├── meta.pkl          (symlink)
│   │   └── extractions.json  (copy, if using JsonExtractor)
│   ├── config.json
│   ├── graph.json
│   ├── chunks.json
│   └── run_metadata.json
└── latest/                   (symlink → most recent run)
```

---

## Configuration

The pipeline reads from the `kg_pipeline` section of `config/config.yaml`:

```yaml
kg_pipeline:
  corpus_description: "Database System Concepts, 7th edition by Silberschatz et al."
  min_cooccurrence: 0   # Prune edges with fewer co-occurrences than this
  top_n: 10             # Max keywords to extract per chunk
```

| Field | Description |
|---|---|
| `corpus_description` | Informational label stored in run metadata |
| `min_cooccurrence` | Minimum edge weight to keep. `0` keeps all edges; `2+` prunes weak connections |
| `top_n` | Target number of keywords per chunk (some extractors scale this adaptively) |

The extractor and linker are configured directly in `run_kg_pipeline.py`.

---

## Extractors

All extractors implement `BaseExtractor` and expose a single method:

```python
def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]
```

| Extractor | Description | When to use |
|---|---|---|
| `JsonExtractor` | Loads pre-computed keywords from a JSON file | Default; instant, must be regenerated if chunks change |
| `KeyBERTExtractor` | Extracts keyphrases using KeyBERT | Local, no API, quick iteration |
| `SLMExtractor` | Prompts a local GGUF model (e.g., Qwen2.5-1.5B) | Local LLM, no API key needed, very slow |
| `OpenRouterExtractor` | Calls a cloud LLM via the OpenRouter API | Highest quality, requires API key |

### JsonExtractor (default)

Reads a JSON file of pre-computed extractions. The file format is:

```json
[
  {"chunk_id": 0, "keywords": ["B+ tree", "indexing", "leaf node"]},
  {"chunk_id": 1, "keywords": ["SQL", "query", "relational algebra"]}
]
```

```python
from src.knowledge_graph.extractors.json_extractor import JsonExtractor

extractor = JsonExtractor(
    input_path="data/knowledge_graph/all__google_gemini-3-flash-preview__extractions__2.json"
)
```

### KeyBERTExtractor

Uses [KeyBERT](https://github.com/MaartenGr/KeyBERT) with a sentence-transformer model (all-MiniLM-L6-v2). No API key required.

```python
from src.knowledge_graph.extractors.keybert_extractor import KeyBERTExtractor

extractor = KeyBERTExtractor(
    model="all-MiniLM-L6-v2",
    top_n=10,
    keyphrase_ngram_range=(1, 2)  # 1- or 2-word phrases
)
```

### SLMExtractor

Runs a small local GGUF model to extract keywords via a structured prompt. Very slow if limited resources.

```python
from src.knowledge_graph.extractors.slm_extractor import SLMExtractor

extractor = SLMExtractor(
    model_path="models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
    n_threads=8,
    top_n=10
)
```

### OpenRouterExtractor

Calls any model available on [OpenRouter](https://openrouter.ai/). Supports `adaptive_top_n`, which scales `top_n` by `√(chunk_length)` for longer chunks.

```python
import os
from src.knowledge_graph.extractors.openrouter_extractor import OpenRouterExtractor

extractor = OpenRouterExtractor(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-flash-1.5",
    top_n=10,
    adaptive_top_n=True
)
```

---

## Linkers

All linkers implement `BaseLinker` and expose:

```python
def link(self, extractions: list[ExtractionResult]) -> nx.Graph
```

### CooccurrenceLinker

Creates an undirected edge between every pair of keywords that appear together in the same chunk. After processing all chunks, edges below `min_cooccurrence` and any resulting isolated nodes are removed.

```python
from src.knowledge_graph.linkers.cooccurrence_linker import CooccurrenceLinker

linker = CooccurrenceLinker(min_cooccurrence=2)
```

**Graph schema:**
- **Nodes**: keyword strings. Carry a `chunk_ids` attribute (list of chunk IDs where the keyword appears).
- **Edges**: keyword pairs. Carry `weight` (number of co-occurrences) and `chunk_ids` (chunks where both appear).

---

## Outputs

### `graph.json`

NetworkX node-link format. Compatible with most graph tools (Gephi, Cytoscape, networkx).

```json
{
  "directed": false,
  "nodes": [
    {"id": "B+ tree", "chunk_ids": [0, 5, 12]},
    {"id": "indexing", "chunk_ids": [0, 3, 5]}
  ],
  "links": [
    {"source": "B+ tree", "target": "indexing", "weight": 3, "chunk_ids": [0, 5, 12]}
  ]
}
```

### `chunks.json`

Maps chunk IDs to their text content.

```json
{
  "0": "A B+ tree is a balanced tree structure...",
  "1": "An index is a data structure that..."
}
```

### `run_metadata.json`

Records the configuration used and graph statistics for reproducibility.

```json
{
  "config": {
    "extractor": {"class": "JsonExtractor", "input_path": "..."},
    "linker":    {"class": "CooccurrenceLinker", "min_cooccurrence": 0}
  },
  "statistics": {
    "linker": {"deleted_edges": 0, "deleted_nodes": 0},
    "graph": {
      "nodes": 450, "edges": 1200,
      "density": 0.012,
      "avg_degree": 5.33,
      "avg_clustering": 0.35,
      "num_connected_components": 3,
      "largest_component_size": 440,
      "max_degree": 52
    }
  }
}
```

---

## Module Structure

```
src/knowledge_graph/
├── run_kg_pipeline.py        # CLI entry point
├── pipeline.py               # Core build_kg() orchestration
├── build.py                  # load_chunks(), path constants
├── models.py                 # Chunk, ExtractionResult, KGPipelineConfig, RunMetadata
├── openrouter_client.py      # HTTP client for OpenRouter API
├── prompts.py                # Prompt templates for SLM and OpenRouter extractors
├── extractors/
│   ├── base_extractor.py
│   ├── json_extractor.py
│   ├── keybert_extractor.py
│   ├── slm_extractor.py
│   └── openrouter_extractor.py
└── linkers/
    ├── base_linker.py
    └── cooccurrence_linker.py
```

---

## Adding a New Extractor or Linker

Subclass `BaseExtractor` or `BaseLinker`, implement the required method, and plug it into `run_kg_pipeline.py`.

```python
from src.knowledge_graph.extractors.base_extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult

class MyExtractor(BaseExtractor):
    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        results = []
        for chunk in chunks:
            keywords = my_keyword_fn(chunk.text)
            results.append(ExtractionResult(chunk_id=chunk.id, keywords=keywords))
        return results
```
