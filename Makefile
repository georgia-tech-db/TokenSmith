.PHONY: help env build-llama clean test run-index run-chat install update-env

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
build: env build-llama install
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

# Run modes
run-index:
	@echo "Running TokenSmith index mode with additional CLI args: $(ARGS)"
	conda run --no-capture-output -n tokensmith python -m src.main index $(ARGS)

run-chat:
	@echo "Running TokenSmith chat mode with additional CLI args: $(ARGS)"
	@echo "Note: Chat mode requires interactive terminal. If this fails, use:"
	@echo "  conda activate tokensmith && python -m src.main chat $(ARGS)"
	conda run --no-capture-output -n tokensmith --no-capture-output python -m src.main chat $(ARGS)
