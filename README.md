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

### 7. Deactivate the Environment
```shell
conda deactivate
```

## ⚙️ Configuration

TokenSmith uses YAML configuration files with the following priority order:

1. Command-line `--config` argument
2. Environment-specific config (e.g., `config/production.yaml`)
3. User config (`~/.config/tokensmith/config.yaml`)
4. Default config (`config/config.yaml`)

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

TokenSmith includes a powerful and flexible testing framework for evaluating RAG performance across multiple dimensions. The framework supports:

- **Multiple output modes**: Terminal (detailed) or HTML (reports)
- **Model comparison**: Easy switching between generator and embedding models
- **Retrieval experimentation**: Test different retrieval methods (FAISS, BM25, Tag, Hybrid) with configurable weights
- **System prompts**: Four prompt modes (baseline, tutor, concise, detailed)
- **Component isolation**: Test generator alone, with golden chunks, or full RAG pipeline
- **Comprehensive metrics**: Semantic similarity, BLEU, keyword matching, and more

### Quick Start

```shell
# Run all benchmarks with HTML report
pytest tests/

# Run with detailed terminal output
pytest tests/ --output-mode=terminal

# Run specific benchmark
pytest tests/ --benchmark-ids="transactions" --output-mode=terminal
```

### Common Testing Patterns

```shell
# Test different system prompts
pytest tests/ --system-prompt=concise --benchmark-ids="transactions" --output-mode=terminal

# Test different retrieval methods
pytest tests/ --retrieval-method=faiss --output-mode=terminal

# Test generator alone (no RAG)
pytest tests/ --disable-chunks --output-mode=terminal

# Configure hybrid retrieval weights
pytest tests/ --faiss-weight=0.6 --bm25-weight=0.3 --tag-weight=0.1
```

### Results

Test results are automatically saved in:
- `tests/results/benchmark_results.json` - Detailed JSON data
- `tests/results/benchmark_summary.html` - Interactive HTML report
- `tests/results/failed_tests.log` - Failure details

### Complete Testing Documentation

For comprehensive testing documentation, see:
- **[tests/README.md](tests/README.md)** - Complete testing guide
- **[tests/QUICK_REFERENCE.md](tests/QUICK_REFERENCE.md)** - Quick reference card
- **Configuration**: All testing options in `config/config.yaml` and CLI arguments

### Key Features

- ✅ **Flexible Configuration**: Set defaults in config.yaml, override with CLI
- ✅ **No Timeouts**: Tests run to completion for reliable results
- ✅ **Golden Chunks**: Test with pre-selected optimal chunks
- ✅ **Ablation Studies**: Easy component isolation and comparison
- ✅ **Multiple Metrics**: Semantic, BLEU, keyword, and text similarity
- ✅ **Beautiful Reports**: HTML reports with metric breakdowns