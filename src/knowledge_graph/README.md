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

## LLM extractor cache workflow

Running cloud LLM extraction (OpenRouter) is expensive. You can cache the extraction results following the next steps:

**Step 1 — Extract keywords and cache them:**
```bash
python -m src.knowledge_graph.llm_extract_keywords \
  --model google/gemini-1.5-flash \
  --chapter 3 \
  --adaptive_top_n
```
This writes to `data/knowledge_graph/extractions/` and updates the `latest.json` symlink.

**Step 2 — Build the graph from cached extractions:**
```bash
python -m src.knowledge_graph.run_kg_pipeline --extractor json
```
This resolves `extractions/latest.json` automatically.

For local extractors (KeyBERT, SLM), extraction and graph building happen in a single step via `run_kg_pipeline.py`.

---

## Running the Pipeline

```bash
conda activate tokensmith
python -m src.knowledge_graph.run_kg_pipeline
```

### Extractor selection

```bash
# Use cached extractions (default) — resolves extractions/latest.json
python -m src.knowledge_graph.run_kg_pipeline --extractor json

# Point at a specific extractions file instead of latest
python -m src.knowledge_graph.run_kg_pipeline --extractor json \
  --extractions data/knowledge_graph/extractions/all__gemini__extractions.json

# Extract inline with OpenRouter (no separate extraction step needed)
python -m src.knowledge_graph.run_kg_pipeline --extractor openrouter \
  --model google/gemini-1.5-flash \
  --adaptive_top_n

# Extract inline with KeyBERT (local, no API key)
python -m src.knowledge_graph.run_kg_pipeline --extractor keybert

# Extract inline with a local GGUF model
python -m src.knowledge_graph.run_kg_pipeline --extractor slm \
  --slm_model_path models/qwen2.5-1.5b-instruct-q5_k_m.gguf
```

### Chunk filtering

```bash
# Only build the graph for chapter 3
python -m src.knowledge_graph.run_kg_pipeline --chapter 3

# Exclude chapters 1 and 2
python -m src.knowledge_graph.run_kg_pipeline --exclude_chapters 1 2
```

### All arguments

| Argument | Applies to | Default | Description |
|---|---|---|---|
| `--config` | all | `config/config.yaml` | Path to project config YAML |
| `--extractor` | all | `json` | Extractor to use: `json`, `openrouter`, `keybert`, `slm` |
| `--chapter` | all | none | Only include chunks from this chapter |
| `--exclude_chapters` | all | none | Exclude chunks from these chapters |
| `--extractions` | `json` | `extractions/latest.json` | Path to a specific extractions JSON |
| `--api_key` | `openrouter` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--model` | `openrouter` | `qwen/qwen3-next-80b-a3b-instruct` | OpenRouter model name |
| `--adaptive_top_n` | `openrouter` | off | Scale `top_n` as `√(chunk_length)` per chunk |
| `--keybert_model` | `keybert` | `all-MiniLM-L6-v2` | Sentence-transformer model name |
| `--slm_model_path` | `slm` | `models/qwen2.5-1.5b-instruct-q5_k_m.gguf` | Path to GGUF model |
| `--slm_threads` | `slm` | `8` | Number of CPU threads |
| `--top_n` | `openrouter`, `keybert`, `slm` | `cfg.top_n` | Keywords per chunk |

Each run creates a timestamped directory and updates a `latest/` symlink:

```
data/knowledge_graph/
├── extractions/                          ← keyword extraction cache
│   ├── all__gemini-flash__extractions.json
│   ├── 3__qwen3__extractions.json
│   └── latest.json                       (symlink → most recent extraction)
└── runs/
    ├── 2025-06-10_14-32-01/
    │   ├── input/
    │   │   ├── chunks.pkl                (symlink)
    │   │   ├── meta.pkl                  (symlink)
    │   │   └── extractions.json          (copy, if using JsonExtractor)
    │   ├── config.json
    │   ├── graph.json
    │   ├── chunks.json
    │   └── run_metadata.json
    └── latest/                           (symlink → most recent run)
```

---

## Extracting Keywords (standalone)

`llm_extract_keywords.py` runs keyword extraction independently from graph building. Use it when you want to pre-compute extractions with a cloud LLM before running the pipeline.

```bash
python -m src.knowledge_graph.llm_extract_keywords \
  --model google/gemini-1.5-flash \
  --chapter 3 \
  --top_n 10

# Adaptive top_n: scales keywords per chunk by √(chunk_length)
python -m src.knowledge_graph.llm_extract_keywords \
  --model qwen/qwen3-next-80b-a3b-instruct \
  --adaptive_top_n

# Process all chapters except 1 and 2
python -m src.knowledge_graph.llm_extract_keywords \
  --exclude_chapters 1 2
```

After a successful run the output is written to `data/knowledge_graph/extractions/` and `extractions/latest.json` is updated automatically.

| Argument | Default | Description |
|---|---|---|
| `--api_key` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--model` | `qwen/qwen3-next-80b-a3b-instruct` | OpenRouter model |
| `--chapter` | all | Only process this chapter |
| `--exclude_chapters` | none | Skip these chapters |
| `--top_n` | `10` | Keywords per chunk (fixed) |
| `--adaptive_top_n` | off | Scale `top_n` as `√(chunk_length)` — overrides `--top_n` |
| `--limit` | none | Cap number of chunks (for testing) |
| `--chunk_ids` | none | Process only these chunk IDs |

---

## Configuration

The pipeline reads from the `kg_pipeline` section of `config/config.yaml`:

```yaml
kg_pipeline:
  corpus_description: "Database System Concepts, 7th edition by Silberschatz et al."
  min_cooccurrence: 0   # Prune edges with fewer co-occurrences than this
  top_n: 10             # Default keywords per chunk (used when --top_n is not passed)
```

| Field | Description |
|---|---|
| `corpus_description` | Informational label stored in run metadata |
| `min_cooccurrence` | Minimum edge weight to keep. `0` keeps all edges; `2+` prunes weak connections |
| `top_n` | Default keyword count; overridden by `--top_n` on the CLI |

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

Reads a JSON file of pre-computed extractions. Resolves `extractions/latest.json` automatically when used via the pipeline CLI.

```json
[
  {"chunk_id": 0, "keywords": ["B+ tree", "indexing", "leaf node"]},
  {"chunk_id": 1, "keywords": ["SQL", "query", "relational algebra"]}
]
```

```python
from src.knowledge_graph.extractors.json_extractor import JsonExtractor
from src.knowledge_graph.build import get_latest_extractions_path

extractor = JsonExtractor(input_path=get_latest_extractions_path())
```

### KeyBERTExtractor

Uses [KeyBERT](https://github.com/MaartenGr/KeyBERT) with a sentence-transformer model. No API key required.

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
    model="google/gemini-1.5-flash",
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
├── run_kg_pipeline.py        # CLI entry point for graph building
├── llm_extract_keywords.py   # Standalone CLI for cloud LLM keyword extraction
├── pipeline.py               # Core build_kg() orchestration
├── build.py                  # load_chunks(), path constants, get_latest_extractions_path()
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
