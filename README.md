# TokenSmith

**TokenSmith** is a local-first database system for students to query textbooks, lecture slides, and notes and get fast, cited answers on their own machines using local LLMs. It is based on retrieval augmented generation (RAG) and applies database-inspired principles like indexing, latency-focused querying, caching, and incremental builds, to optimize the ingestion -> retrieval -> generation pipeline.

## Capabilities

* Parse and index PDF documents
* Semantic retrieval with FAISS
* Local inference via `llama.cpp` (GGUF models)
* Acceleration: Metal (Apple Silicon), CUDA (NVIDIA), or CPU
* Configurable chunking (tokens or characters)
* Optional indexing progress visualization
* Table preservation during indexing (flag-based)

## Requirements

* **Python** 3.9+
* **Conda/Miniconda**
* **System prerequisites**:

  * macOS: Xcode Command Line Tools
  * Linux: GCC, make, CMake
  * Windows: Visual Studio Build Tools

## Quick Start

### 1) Clone the repository and Download the models

```shell
git clone https://github.com/georgia-tech-db/TokenSmith.git
cd TokenSmith
```

Create the model directory and put in the appropriate models in it.
```shell
mkdir models
cd models
```

Now, let's say config.yaml has following configs:
```yaml
embed_model: "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
model_path: "models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
```
For above config file, download appropriate files from the below link 
and put them in the `models/` folder with the expected file name.
- https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/tree/main
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main

### 2) Build (creates env, builds llama.cpp, installs deps)

```shell
make build
```

Creates a Conda env `tokensmith`, installs Python deps, and builds/detects `llama.cpp`.

### 3) Activate the environment

```shell
conda activate tokensmith
```

### 4) Prepare documents

```shell
mkdir -p data/chapters
cp your-documents.pdf data/chapters/
```

### 5) Index documents

```shell
make run-index
```

With custom parameters:

```shell
make run-index ARGS="--pdf_range 1-10 --chunk_mode chars --visualize"
```

### 6) Chat

```shell
python -m src.main chat
```

> If you see a missing-model error, download `qwen2.5-0.5b-instruct-q5_k_m.gguf` into `llama.cpp/models`.

### 7) Deactivate

```shell
conda deactivate
```

## Configuration

Priority (highest → lowest):

1. `--config` CLI argument
2. `~/.config/tokensmith/config.yaml`
3. `config/config.yaml`

### Example

```yaml
embed_model: "sentence-transformers/all-MiniLM-L6-v2"
top_k: 5
max_gen_tokens: 400
halo_mode: "none"
seg_filter: null

# Model settings
model_path: "models/qwen2.5-0.5b-instruct-q5_k_m.gguf"

# Indexing settings
chunk_mode: "tokens" # or "chars"
chunk_tokens: 500
chunk_size_char: 20000
```

## Usage

### Basic indexing

```shell
make run-index
```

### Index a specific PDF range

```shell
make run-index ARGS="--pdf_range <start>-<end> --chunk_mode <tokens|chars>"
```

### Index with tables/visualization

```shell
make run-index ARGS="--keep_tables --visualize --chunk_tokens <num_tokens>"
```

### Custom paths/settings

```shell
make run-index ARGS="--pdf_dir <path_to_pdf> --index_prefix book_index --config <path_to_yaml>"
```

### Chat with custom settings

```shell
python -m src.main chat --config <path_to_yaml> --model_path <path_to_gguf>
```

### Use an existing llama.cpp build

```shell
export LLAMA_CPP_BINARY=/usr/local/bin/llama-cli
make build
```

### Environment maintenance

```shell
make update-env
make export-env
make show-deps
```

## Command-Line Arguments

### Core

* `mode`: `index` or `chat`
* `--config`: path to YAML config
* `--pdf_dir`: directory with PDFs
* `--index_prefix`: prefix for index files
* `--model_path`: path to GGUF model

### Indexing

* `--pdf_range`: e.g., `1-10`
* `--chunk_mode`: `tokens` or `chars`
* `--chunk_tokens`: default 500
* `--chunk_size_char`: default 20000
* `--keep_tables`
* `--visualize`

## Development

```shell
make help
make env
make build-llama
make install
make build
make test
make clean
make show-deps
make update-env
make export-env
```

## Testing

```shell
pytest tests/
pytest tests/ -s
pytest tests/ --benchmark-ids="test" -s
```

* Tests call the same `get_answer()` pipeline used by chat
* Metrics: semantic similarity, BLEU, keyword matching, text similarity
* Outputs: terminal logs and HTML report
* System prompts: baseline, tutor, concise, detailed
* Component isolation: run with/without chunks or with golden chunks

Artifacts:

* `tests/results/benchmark_results.json`
* `tests/results/benchmark_summary.html`
* `tests/results/failed_tests.log`

Documentation: see `tests/README.md`.
