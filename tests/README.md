# Testing Framework

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
pytest tests/
```
Use the `-s` flag to get the output from the LLM model. Without the `-s` flag, the command only shows the result for the metrics

```bash
# Run with terminal output to see results immediately
pytest tests/ -s

# Run specific benchmark
pytest tests/ --benchmark-ids="test" -s

# Suppress warnings
pytest tests/ -s -W ignore::DeprecationWarning
```

### Common Patterns

```bash
# Test with different system prompts
pytest tests/ --system_prompt_mode=concise -s

# Test without chunks (baseline mode)
pytest tests/ --disable-chunks -s

# Test with golden chunks
pytest tests/ --use-golden-chunks -s

# Test with specific metrics
pytest tests/ --metrics=semantic --metrics=keyword -s

# Test with HTML output (default)
pytest tests/ --output-mode=html
```

## Configuration

### Config File (config/config.yaml)

```yaml
# Embedding Configuration
embed_model: "/path/to/Qwen3-Embedding-4B-Q8_0.gguf"

# Retrieval Configuration
top_k: 5
pool_size: 60
ensemble_method: "rrf"  # or "linear", "weighted"
ranker_weights:
  faiss: 0.6
  bm25: 0.4
rrf_k: 60

# Generator Configuration
model_path: "models/qwen2.5-0.5b-instruct-q5_k_m.gguf"
max_gen_tokens: 400
system_prompt_mode: "tutor"  # Options: baseline, tutor, concise, detailed

# Testing Configuration
use_golden_chunks: false
output_mode: "terminal"  # terminal or html
metrics: ["all"]  # or specific: ["semantic", "keyword", "bleu"]
```

### CLI Arguments Reference

All config options can be overridden via CLI (CLI takes priority):

| Argument | Options | Description |
|----------|---------|-------------|
| `--benchmark-ids` | comma-separated | Filter specific benchmarks (e.g., "test,transactions") |
| `--system_prompt_mode` | baseline, tutor, concise, detailed | System prompt style |
| `--enable-chunks` | flag | Enable chunks in generation (RAG mode) |
| `--disable-chunks` | flag | Disable chunks (baseline mode) |
| `--use-golden-chunks` | flag | Use pre-selected golden chunks |
| `--output-mode` | terminal, html | Output format |
| `--metrics` | metric name | Metrics to use (can specify multiple times) |
| `-s` | flag | Show print statements (pytest flag) |
| `-W ignore::DeprecationWarning` | flag | Suppress FAISS warnings |

## System Prompt Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **baseline** | No system prompt | Testing generator alone |
| **tutor** | Friendly, educational | Default, best for learning |
| **concise** | Brief, direct | Quick answers, summaries |
| **detailed** | Comprehensive | In-depth explanations |

## Benchmarks Format

Benchmarks are defined in `benchmarks.yaml`:

```yaml
benchmarks:
  - id: "unique_id"
    question: "Your question here..."
    expected_answer: "Expected answer text..."
    keywords: ["keyword1", "keyword2"]
    similarity_threshold: 0.7
    golden_chunks: null  # or list of best chunks
```

### Golden Chunks

Golden chunks are pre-selected text snippets most relevant to a question. To use:

1. Add to benchmarks.yaml:
   ```yaml
   golden_chunks:
     - "First most relevant chunk..."
     - "Second most relevant chunk..."
   ```

2. Run with golden chunks:
   ```bash
   pytest tests/ --use-golden-chunks -s
   ```

## Output Modes

### Terminal Mode (Detailed)

```bash
pytest tests/ --output-mode=terminal -s
```

Shows:
- Configuration details
- Per-benchmark progress
- Retrieval info
- Metric breakdowns
- Retrieved answer display
- Pass/fail status

Example output:
```
============================================================
  TokenSmith Benchmark Configuration
============================================================
  Generator Model:    qwen2.5-0.5b-instruct-q5_k_m.gguf
  Embedding Model:    Qwen3-Embedding-4B-Q8_0.gguf
  System Prompt:      tutor
  Chunks Enabled:     True
============================================================

────────────────────────────────────────────────────────────
  Benchmark: test
  Question: What are the contributions...
────────────────────────────────────────────────────────────
  🔍 Retrieving chunks...

  ✅ PASSED
  Final Score:  0.847 (threshold: 0.700)
  Metric Breakdown:
    • semantic    : 0.892
    • text        : 0.834

  📝 Retrieved Answer:
  ----------------------------------------------------------
  The answer text appears here with proper formatting...
  ----------------------------------------------------------
```

### HTML Mode (Reports)

```bash
pytest tests/ --output-mode=html
```

Generates:
- `tests/results/benchmark_summary.html` - Interactive report
- `tests/results/benchmark_results.json` - Detailed data
- `tests/results/failed_tests.log` - Failure details

## Usage Examples

### Experiment with System Prompts

```bash
# Compare all prompt modes
for mode in baseline tutor concise detailed; do
    echo "===== Testing $mode ====="
    pytest tests/ --system_prompt_mode=$mode --benchmark-ids="test" -s
done
```

### Component Isolation (Ablation Study)

```bash
# 1. Pure generator (no RAG, no prompt)
pytest tests/ --disable-chunks --system_prompt_mode=baseline -s

# 2. Generator with prompt (no RAG)
pytest tests/ --disable-chunks --system_prompt_mode=tutor -s

# 3. RAG without prompt
pytest tests/ --enable-chunks --system_prompt_mode=baseline -s

# 4. Full system (RAG + prompt)
pytest tests/ --enable-chunks --system_prompt_mode=tutor -s
```

### Test Specific Benchmarks

```bash
# Run one benchmark for quick iteration
pytest tests/ --benchmark-ids="test" -s

# Run multiple specific benchmarks
pytest tests/ --benchmark-ids="test,transactions" -s
```

### Metrics Selection

```bash
# List available metrics
pytest tests/ --list-metrics

# Use specific metric
pytest tests/ --metrics=semantic -s

# Use multiple metrics
pytest tests/ --metrics=semantic --metrics=keyword -s

# Use all metrics (default)
pytest tests/ --metrics=all -s
```

## Results Files

After running tests, check:

```bash
# View JSON results
cat tests/results/benchmark_results.json | python -m json.tool

# Open HTML report (Linux)
xdg-open tests/results/benchmark_summary.html

# Check failures
cat tests/results/failed_tests.log
```

## Troubleshooting

### No Output Visible

**Problem**: Test passes but no output shown \
**Solution**: Add `-s` flag to see print statements
```bash
pytest tests/ -s
```

### FAISS Warnings

**Problem**: Many deprecation warnings \
**Solution**: Suppress with pytest flag
```bash
pytest tests/ -W ignore::DeprecationWarning
```

## Advanced Usage

### Run Specific Test Function
```bash
pytest tests/test_benchmarks.py::test_tokensmith_benchmarks -s
```

### Generate HTML Report Only
```bash
pytest tests/ --output-mode=html
# No terminal output, only HTML report generated
```

### Verbose trace callback printing
```bash
pytest --log-cli-level=DEBUG tests/ --config=config/test_config.yaml --benchmark-ids='test' -vv -s --pdb --full-trace --showlocals
```

## Metrics Explained

- **Semantic Similarity**: Uses sentence embeddings to measure meaning similarity
- **Keyword Matching**: Checks if important keywords are present
- **NLI Classification**: Find NLI entailment score between answer and expected text

All metrics are weighted and combined into a final score, which is compared against the benchmark's threshold.
