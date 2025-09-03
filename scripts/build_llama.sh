#!/bin/bash
set -e

echo "TokenSmith: Building llama.cpp from source..."

# Create build directory
BUILD_DIR="./build"
LLAMA_DIR="$BUILD_DIR/llama.cpp"

mkdir -p "$BUILD_DIR"

# Clone llama.cpp if not exists
if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
else
    echo "Updating llama.cpp..."
    cd "$LLAMA_DIR"
    git pull
    cd - > /dev/null
fi

cd "$LLAMA_DIR"

# Platform-specific build configuration
OS=$(uname -s)
ARCH=$(uname -m)
CMAKE_OPTS=""

if [[ "$OS" == "Darwin" ]]; then
    echo "Configuring for macOS..."
    if [[ "$ARCH" == "arm64" ]]; then
        echo "Enabling Metal support for Apple Silicon"
        CMAKE_OPTS="-DGGML_METAL=ON -DGGML_ACCELERATE=ON"
    else
        CMAKE_OPTS="-DGGML_ACCELERATE=ON"
    fi
elif [[ "$OS" == "Linux" ]]; then
    echo "Configuring for Linux..."
    # Check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected - enabling CUDA"
        CMAKE_OPTS="-DGGML_CUDA=ON"
    else
        echo "CPU-only build"
        CMAKE_OPTS="-DGGML_ACCELERATE=ON"
    fi
fi

# Build
echo "Building with options: $CMAKE_OPTS"
mkdir -p build
cd build

cmake .. $CMAKE_OPTS -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) llama-cli

# Verify build
if [[ -f "./bin/llama-cli" ]]; then
    BINARY_PATH="$(pwd)/bin/llama-cli"
elif [[ -f "./llama-cli" ]]; then
    BINARY_PATH="$(pwd)/llama-cli"
else
    echo "Error: llama-cli binary not found after build"
    exit 1
fi

echo "✓ TokenSmith: Build successful: $BINARY_PATH"

# Save path for TokenSmith in src/ directory
cd - > /dev/null
cd - > /dev/null  # Back to project root

mkdir -p src
echo "$BINARY_PATH" > src/llama_path.txt

echo "TokenSmith: llama.cpp build complete!"
