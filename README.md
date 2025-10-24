# TokenSmith

**TokenSmith** is a Retrieval-Augmented Generation (RAG) application that enables intelligent document search and question answering using local LLMs. Built with llama.cpp for efficient inference and FAISS for high-performance vector search, TokenSmith allows you to index PDF documents and chat with them using natural language queries.

## 🚀 Features

- **📚 PDF Document Processing**: Extract and index content from PDF documents
- **🔍 Intelligent Retrieval**: Fast semantic search using FAISS vector database
- **🤖 Local LLM Integration**: Powered by llama.cpp for privacy-focused inference
- **⚡ Hardware Acceleration**: Supports Metal (Apple Silicon), CUDA (NVIDIA), and CPU inference
- **🎯 Flexible Chunking**: Token-based or character-based document segmentation
- **📊 Visualization Support**: Optional indexing progress visualization
- **🛠️ Production-Ready**: Conda-based environment management with automated builds
- **🔧 Configurable**: YAML-based configuration system

## 📋 Requirements

- **Python**: 3.9+ 
- **Conda/Miniconda**: For environment management
- **System Requirements**:
  - macOS: Xcode Command Line Tools
  - Linux: GCC, make, cmake
  - Windows: Visual Studio Build Tools (for compilation)

## 🚀 Quick Start

### 1. Clone the Repository
```shell
git clone https://github.com/georgia-tech-db/TokenSmith.git
cd tokensmith
```

### One-command setup: creates conda env, builds llama.cpp, installs dependencies
```shell
make build
```
This will:
- Create a conda environment named `tokensmith`
- Install all Python dependencies
- Detect or build llama.cpp with platform-specific optimizations
- Install TokenSmith in development mode

### 3. Activate the Environment
```shell
conda activate tokensmith
```

### 4. Prepare Your Documents
Place your PDF files in the data directory
```shell
mkdir -p data/chapters
cp your-documents.pdf data/chapters/
```

### 5. Index Your Documents
Index with default settings
```shell
make run-index
```
Or with custom parameters, eg.
```shell
make run-index ARGS="--pdf_range 1-10 --chunk_mode chars --visualize"
```

### 6. Start Chatting
Activate environment first (required for interactive mode)
```shell
conda activate tokensmith
python -m src.main chat
```

> You might have to download `qwen2.5-0.5b-instruct-q5_k_m.gguf` into your `llama.cpp/models` if you get an error about a missing model.

### 7. Deactivate the Environment
```shell
conda deactivate
```

## ⚙️ Configuration

TokenSmith uses YAML configuration files with the following priority order:

1. Command-line `--config` argument
2. User config (`~/.config/tokensmith/config.yaml`)
3. Default config (`config/config.yaml`)

### Sample Configuration
```yaml
# config/config.yaml

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

## 🎮 Usage

### Basic indexing
```shell
make run-index
```

### Index specific PDF range
```shell
make run-index ARGS="--pdf_range <start_page_number>-<end_page_number> --chunk_mode <tokens_or_chars>"
```

### Index with visualization and table preservation
```shell
make run-index ARGS="--keep_tables --visualize --chunk_tokens <number_of_chunk_tokens>"
```

### Custom paths and settings
```shell
make run-index ARGS="--pdf_dir <path_to_pdf> --index_prefix book_index --config <path_to_yaml_config_file>"
```

### Chat with custom settings
```shell
python -m src.main chat --config <path_to_yaml_config_file> --model_path <path_to_llm_model>
```

### Build with existing llama.cpp installation
```shell
export LLAMA_CPP_BINARY=/usr/local/bin/llama-cli
make build
```

### Update environment with new dependencies
```shell
make update-env
```

### Export environment for sharing
```shell
make export-env
```

### Show installed packages
```shell
make show-deps
```


## 📊 Command Line Arguments

### Core Arguments
- `mode`: Operation mode (`index` or `chat`)
- `--config`: Configuration file path
- `--pdf_dir`: Directory containing PDF files
- `--index_prefix`: Prefix for index files
- `--model_path`: Path to GGUF model file

### Indexing Arguments
- `--pdf_range`: Process specific page range (e.g., "1-10")
- `--chunk_mode`: Chunking strategy (`tokens` or `chars`)
- `--chunk_tokens`: Tokens per chunk (default: 500)
- `--chunk_size_char`: Characters per chunk (default: 20000)
- `--keep_tables`: Preserve table formatting
- `--visualize`: Show indexing progress visualization

## 🔨 Development

### Available Make Targets
```shell
make help          # Show all available commands
make env           # Create conda environment
make build-llama   # Build llama.cpp from source
make install        # Install package in development mode
make build          # Full build process
make test # Run tests
make clean # Clean build artifacts
make show-deps # Show installed packages
make update-env # Update environment
make export-env # Export environment with exact versions
```

### Adding Dependencies
```shell
# Add new conda package
conda activate tokensmith
conda install new-package
```
Add to environment.yml for persistence. Edit environment.yml, then:
```shell
make update-env
```

## 🧪 Testing Framework

TokenSmith includes a comprehensive testing framework for evaluating RAG performance. The framework is fully integrated with the main pipeline, ensuring tests use the same code path as production.

### Quick Start

```shell
# Run all benchmarks
pytest tests/

# Run with terminal output to see detailed results
pytest tests/ -s

# Run specific benchmark
pytest tests/ --benchmark-ids="test" -s
```

### Features

- ✅ **Integrated Pipeline**: Tests use the same `get_answer()` function as chat
- ✅ **Multiple Metrics**: Semantic similarity, BLEU, keyword matching, text similarity
- ✅ **Flexible Output**: Terminal (detailed) or HTML (reports)
- ✅ **System Prompts**: Four modes (baseline, tutor, concise, detailed)
- ✅ **Component Isolation**: Test with/without chunks, or use golden chunks
- ✅ **Beautiful Reports**: Interactive HTML reports with metric breakdowns

### Results

Test results are saved in:
- `tests/results/benchmark_results.json` - Detailed JSON data
- `tests/results/benchmark_summary.html` - Interactive HTML report
- `tests/results/failed_tests.log` - Failure details

### Documentation

For complete testing documentation, usage examples, and configuration options:

📖 **[tests/README.md](tests/README.md)** - Complete testing guide with all CLI options and examples