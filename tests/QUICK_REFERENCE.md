# Testing Framework - Quick Reference Card

## Essential Commands

```bash
# List available metrics
pytest tests/ --list-metrics

# Run all tests (HTML report)
pytest tests/

# Run with terminal output
pytest tests/ --output-mode=terminal

# Run specific benchmark
pytest tests/ --benchmark-ids="transactions" --output-mode=terminal
```

## Common Flags

### Output Control
```bash
--output-mode=terminal    # Detailed console output
--output-mode=html        # Generate HTML report (default)
```

### Benchmark Selection
```bash
--benchmark-ids="id1,id2"  # Run specific benchmarks
```

### Models
```bash
--generator-model="path/to/model.gguf"  # Choose generator
--embed-model="model-name-or-path"      # Choose embedder
```

### Retrieval
```bash
--retrieval-method=faiss     # Vector search only
--retrieval-method=bm25      # BM25 only
--retrieval-method=hybrid    # Weighted combination (default)
--faiss-weight=0.5           # Set FAISS weight
--bm25-weight=0.3            # Set BM25 weight
--tag-weight=0.2             # Set tag weight
```

### Prompts
```bash
--system-prompt=baseline     # No system prompt
--system-prompt=tutor        # Friendly tutor (default)
--system-prompt=concise      # Brief answers
--system-prompt=detailed     # Comprehensive explanations
```

### Chunks
```bash
--enable-chunks              # Use RAG (default)
--disable-chunks             # Test generator alone
--use-golden-chunks          # Use pre-selected chunks
```

### Metrics
```bash
--metrics=semantic           # Use specific metric
--metrics=semantic --metrics=keyword  # Multiple metrics
--metrics=all                # All metrics (default)
--threshold=0.75             # Override threshold
```

## Config File Structure

```yaml
# config/config.yaml
embed_model: "path/to/embedding/model"
generator_model: "path/to/generator/model"
retrieval_method: "hybrid"  # faiss|bm25|tag|hybrid
faiss_weight: 0.5
bm25_weight: 0.3
tag_weight: 0.2
system_prompt_mode: "tutor"  # baseline|tutor|concise|detailed
enable_chunks: true
top_k: 5
max_gen_tokens: 400

testing:
  output_mode: "html"  # terminal|html
  index_prefix: "textbook_index"
  metrics: ["all"]
  threshold_override: null
  use_golden_chunks: false
```

## Benchmark File Format

```yaml
# tests/benchmarks.yaml
benchmarks:
  - id: "unique_identifier"
    question: "Your question text..."
    expected_answer: "Expected answer text..."
    keywords: ["keyword1", "keyword2"]
    similarity_threshold: 0.7
    golden_chunks: null  # or list of chunks
```

## Output Files

```
tests/results/
├── benchmark_results.json      # Detailed JSON results
├── benchmark_summary.html      # HTML report
└── failed_tests.log           # Failure details
```

## Compare Configurations
```bash
# Compare prompts
for p in baseline tutor concise detailed; do
  pytest tests/ --system-prompt=$p --benchmark-ids="transactions"
done

# Compare retrieval
for r in faiss bm25 hybrid; do
  pytest tests/ --retrieval-method=$r --benchmark-ids="transactions"
done
```

### Debug Single Test
```bash
pytest tests/ -s \
  --benchmark-ids="transactions" \
  --output-mode=terminal
```