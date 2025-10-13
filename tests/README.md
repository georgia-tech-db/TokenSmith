# TokenSmith Testing Framework

A simplified and flexible testing framework for benchmarking TokenSmith's RAG system.

## Overview

The testing framework allows you to:
- Run benchmark questions through the full TokenSmith pipeline
- Evaluate answers using multiple metrics (semantic similarity, BLEU, keyword matching, etc.)
- Compare different configurations (models, retrieval methods, prompts)
- View results in terminal or generate HTML reports
- Use golden chunks for controlled testing

## Quick Start

### Basic Usage

Run all benchmarks with default configuration:
```bash
pytest tests/test_benchmarks.py
```
Use the `-s` flag to get the output from the LLM model. Without the `-s` flag, the command only shows the result for the metrics

Run specific benchmarks:
```bash
pytest tests/test_benchmarks.py --benchmark-ids="transactions,er_modeling"
```

### Terminal vs HTML Output

Show results in terminal (detailed output):
```bash
pytest tests/test_benchmarks.py --output-mode=terminal
```

Generate HTML report (default):
```bash
pytest tests/test_benchmarks.py --output-mode=html
```

## Configuration

### Config File (config/config.yaml)

The main configuration file supports all testing options:

```yaml
# Embedding Configuration
embed_model: "/path/to/Qwen3-Embedding-4B-Q8_0.gguf"

# Retrieval Configuration
retrieval_method: "hybrid"  # Options: hybrid, faiss, bm25, tag
faiss_weight: 0.5
bm25_weight: 0.3
tag_weight: 0.2
top_k: 5
halo_mode: "halo"

# Generator Configuration
generator_model: "models/qwen2.5-0.5b-instruct-q5_k_m.gguf"
max_gen_tokens: 400
system_prompt_mode: "tutor"  # Options: baseline, tutor, concise, detailed
enable_chunks: true

# Testing Configuration
testing:
  output_mode: "html"  # Options: terminal, html
  index_prefix: "textbook_index"
  benchmark_ids: null  # or comma-separated list
  metrics: ["all"]  # Options: text, semantic, keyword, bleu, all
  threshold_override: null
  use_golden_chunks: false
```

### Command-Line Arguments

All config options can be overridden via CLI:

#### Model Selection
```bash
# Choose generator model
pytest tests/ --generator-model="models/my-model.gguf"

# Choose embedding model (defaults to Qwen3)
pytest tests/ --embed-model="sentence-transformers/all-MiniLM-L6-v2"
```

#### Retrieval Configuration
```bash
# Set retrieval method
pytest tests/ --retrieval-method=faiss

# Configure hybrid retrieval weights
pytest tests/ --faiss-weight=0.6 --bm25-weight=0.3 --tag-weight=0.1
```

#### System Prompts
```bash
# Baseline: No system prompt
pytest tests/ --system-prompt=baseline

# Tutor: Friendly tutoring style (default)
pytest tests/ --system-prompt=tutor

# Concise: Brief, direct answers
pytest tests/ --system-prompt=concise

# Detailed: Comprehensive explanations
pytest tests/ --system-prompt=detailed
```

#### Chunks Control
```bash
# Disable chunks (test generator alone)
pytest tests/ --disable-chunks

# Enable chunks (default)
pytest tests/ --enable-chunks

# Use golden chunks from benchmarks.yaml
pytest tests/ --use-golden-chunks
```

#### Metrics Selection
```bash
# Use specific metrics
pytest tests/ --metrics=semantic --metrics=keyword

# Override similarity threshold
pytest tests/ --threshold=0.75

# List available metrics
pytest tests/ --list-metrics
```

## Benchmarks File (benchmarks.yaml)

Each benchmark includes:

```yaml
benchmarks:
  - id: "transactions"
    question: "Your question here..."
    expected_answer: "Expected answer text..."
    keywords: ["keyword1", "keyword2"]
    similarity_threshold: 0.7
    golden_chunks: null  # Optional: list of best chunks for this question
```

### Golden Chunks

Golden chunks are pre-selected text snippets that are most relevant to a question. 
To use golden chunks:
1. Add them to benchmarks.yaml:
   ```yaml
   golden_chunks:
     - "First most relevant chunk..."
     - "Second most relevant chunk..."
   ```

2. Enable in config.yaml:
   ```yaml
   testing:
     use_golden_chunks: true
   ```

   Or via CLI:
   ```bash
   pytest tests/ --use-golden-chunks
   ```

## Example Use Cases

### Compare Retrieval Methods

Test FAISS-only retrieval:
```bash
pytest tests/ --retrieval-method=faiss --output-mode=terminal
```

Test hybrid retrieval with different weights:
```bash
pytest tests/ --faiss-weight=0.7 --bm25-weight=0.2 --tag-weight=0.1
```

### Compare System Prompts

Test all prompt modes:
```bash
for mode in baseline tutor concise detailed; do
    pytest tests/ --system-prompt=$mode --output-mode=terminal
done
```

### Test Generator Alone

Disable chunks to test generator without retrieval:
```bash
pytest tests/ --disable-chunks --system-prompt=baseline
```

Or use golden chunks for controlled testing:
```bash
pytest tests/ --use-golden-chunks
```

### Run Specific Benchmarks

Test only database-related questions:
```bash
pytest tests/ --benchmark-ids="transactions,materialization"
```

### Generate Reports

Run tests and generate HTML report:
```bash
pytest tests/ --output-mode=html
# Opens: tests/results/benchmark_summary.html
```

## Results

### Terminal Output

When using `--output-mode=terminal`, you get detailed output:
```
==============================================================
  TokenSmith Benchmark Configuration
==============================================================
  Generator Model:    qwen2.5-0.5b-instruct-q5_k_m.gguf
  Embedding Model:    Qwen3-Embedding-4B-Q8_0.gguf
  Retrieval Method:   hybrid
    • FAISS weight:   0.50
    • BM25 weight:    0.30
    • Tag weight:     0.20
  System Prompt:      tutor
  Chunks Enabled:     True
  Golden Chunks:      False
  Output Mode:        terminal
  Metrics:            semantic, text, keyword, bleu
==============================================================

────────────────────────────────────────────────────────────
  Benchmark: transactions
  Question: How do atomicity, consistency, isolation, and...
────────────────────────────────────────────────────────────
  🔍 Retrieved 5 chunks

  ✅ PASSED
  Final Score:  0.847 (threshold: 0.700)
  Metric Breakdown:
    • semantic    : 0.892
    • text        : 0.834
    • keyword     : 0.815
    • bleu        : 0.847
    • keywords    : 4/5
```

### HTML Report

When using `--output-mode=html` (default), results are saved to:
- `tests/results/benchmark_results.json` - Detailed JSON results
- `tests/results/benchmark_summary.html` - Interactive HTML report
- `tests/results/failed_tests.log` - Failure details


## Adding New Features

### Add a New Metric

1. Create metric in `tests/utils/metrics/`:
   ```python
   from .base import MetricBase
   
   class MyMetric(MetricBase):
       def __init__(self):
           super().__init__(name="my_metric", weight=1.0)
       
       def calculate(self, answer, expected, keywords=None):
           # Your metric logic
           return score
   ```

2. Register in `tests/utils/metrics/__init__.py`

3. Use it:
   ```bash
   pytest tests/ --metrics=my_metric -s
   ```

### Add a New System Prompt

Edit `src/generator.py`:
```python
def get_system_prompt(mode="tutor"):
    prompts = {
        "my_mode": "Your custom prompt...",
        # ... existing modes
    }
    return prompts.get(mode, prompts["tutor"])
```

Use it:
```bash
pytest tests/ --system-prompt=my_mode -s
```

