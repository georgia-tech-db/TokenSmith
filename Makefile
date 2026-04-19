.PHONY: help env build-llama clean test run-index run-chat run-index-partial run-add-chapters-partial run-chat-partial install update-env

help:
	@echo "TokenSmith - RAG Application (Conda Dependencies)"
	@echo "Available targets:"
	@echo "  env         - Create conda environment with all dependencies"
	@echo "  update-env  - Update environment from environment.yml"
	@echo "  build-llama - Build llama.cpp (if not found)"  
	@echo "  install     - Install package in development mode"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean build artifacts"
	@echo "  show-deps   - Show installed conda packages"
	@echo "  export-env  - Export current environment"
	@echo "  run-index-partial - Create a partial index (e.g., make run-index-partial CHAPTERS=\"1 2\")"
	@echo "  run-add-chapters-partial - Add chapters to partial index (e.g., make run-add-chapters-partial CHAPTERS=\"3\")"
	@echo "  run-chat-partial - Chat using partial index"

# Environment setup - installs all dependencies via conda
env:
	@echo "Creating TokenSmith conda environment..."
	conda env create -f environment.yml -n tokensmith || conda env update -f environment.yml -n tokensmith
	@echo "Running platform-specific setup..."
	conda run -n tokensmith bash scripts/setup_env.sh

# Update environment from environment.yml
update-env:
	@echo "Updating TokenSmith conda environment..."
	conda env update -f environment.yml -n tokensmith

# Build llama.cpp if needed
build-llama:
	@echo "Checking for existing llama.cpp installation..."
	conda run -n tokensmith python scripts/detect_llama.py || conda run -n tokensmith bash scripts/build_llama.sh

# Install package in development mode (no dependencies, they're from conda)
install:
	conda run -n tokensmith pip install -e . --no-deps

# Full build process
build: env install
	@echo "TokenSmith build complete! Activate environment with: conda activate tokensmith"

# Show installed packages
show-deps:
	@echo "Installed conda packages:"
	conda list -n tokensmith

# Export current environment for sharing
export-env:
	@echo "Exporting environment to environment-lock.yml..."
	conda env export -n tokensmith > environment-lock.yml
	@echo "Environment exported with exact versions."

# Run tests
test:
	conda run -n tokensmith python -m pytest tests/ -v || echo "No tests found"

# Clean
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# PDF to Markdown extraction
run-extract:
	@echo "Extracting PDF to markdown (data/chapters/*.pdf -> data/*.md)"
	conda run --no-capture-output -n tokensmith python -m src.preprocessing.extraction
	
# Run modes
run-index:
	@echo "Running TokenSmith index mode with additional CLI args: $(ARGS)"
	conda run --no-capture-output -n tokensmith python -m src.main index $(ARGS)

run-chat:
	@echo "Running TokenSmith chat mode with additional CLI args: $(ARGS)"
	@echo "Note: Chat mode requires interactive terminal. If this fails, use:"
	@echo "  conda activate tokensmith && python -m src.main chat $(ARGS)"
	conda run --no-capture-output -n tokensmith --no-capture-output python -m src.main chat $(ARGS)

run-index-partial:
	@echo "Running TokenSmith partial index mode with chapters: $(CHAPTERS) $(ARGS)"
	conda run --no-capture-output -n tokensmith python -m src.main index --partial --chapters $(CHAPTERS) $(ARGS)

run-add-chapters-partial:
	@echo "Adding chapters $(CHAPTERS) to partial index with ARGS: $(ARGS)"
	conda run --no-capture-output -n tokensmith python -m src.main add-chapters --partial --chapters $(CHAPTERS) $(ARGS)

run-chat-partial:
	@echo "Running TokenSmith chat mode (partial index) with additional CLI args: $(ARGS)"
	conda run --no-capture-output -n tokensmith --no-capture-output python -m src.main chat --partial $(ARGS)
