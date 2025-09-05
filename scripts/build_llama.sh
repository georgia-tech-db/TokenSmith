#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

log() { printf '%s\n' "$*" >&2; }
die() { log "Error: $*"; exit 1; }

# --- Locate script, project root, and key dirs ---
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
# If this repo is a git repo, trust git; else fall back to script's parent
PROJECT_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/.." && pwd -P))"

BUILD_DIR="$PROJECT_ROOT/build"
LLAMA_DIR="$BUILD_DIR/llama.cpp"
LLAMA_BUILD_DIR="$LLAMA_DIR/build"

log "TokenSmith: Building llama.cpp from source..."
mkdir -p "$BUILD_DIR"

# --- Ensure required tools exist ---
command -v git >/dev/null 2>&1 || die "git not found. Install git."
command -v cmake >/dev/null 2>&1 || die "cmake not found. Install cmake (>=3.21)."

# --- Get or update llama.cpp ---
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  log "Cloning llama.cpp into: $LLAMA_DIR"
  git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
else
  log "Updating llama.cpp..."
  git -C "$LLAMA_DIR" pull --ff-only
fi

# --- Platform-specific build options ---
OS="$(uname -s)"
ARCH="$(uname -m)"
CMAKE_OPTS=()

case "$OS" in
  Darwin)
    log "Configuring for macOS ($ARCH)..."
    if [[ "$ARCH" == "arm64" ]]; then
      log "Enabling Metal + Accelerate for Apple Silicon"
      CMAKE_OPTS+=(-DGGML_METAL=ON -DGGML_ACCELERATE=ON)
    else
      CMAKE_OPTS+=(-DGGML_ACCELERATE=ON)
    fi
    ;;
  Linux)
    log "Configuring for Linux..."
    if command -v nvidia-smi >/dev/null 2>&1; then
      log "NVIDIA GPU detected — enabling CUDA"
      CMAKE_OPTS+=(-DGGML_CUDA=ON)
    else
      log "CPU-only build (no NVIDIA GPU detected)"
      CMAKE_OPTS+=(-DGGML_ACCELERATE=ON)
    fi
    ;;
  *)
    log "Unknown OS '$OS' — proceeding with defaults"
    ;;
esac

# --- Figure out parallelism ---
CORES="$(
  { command -v nproc >/dev/null 2>&1 && nproc; } \
  || { command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu; } \
  || echo 4
)"

# --- Configure & build ---
log "Building with options: ${CMAKE_OPTS[*]}"
cmake -S "$LLAMA_DIR" -B "$LLAMA_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "${CMAKE_OPTS[@]}"
cmake --build "$LLAMA_BUILD_DIR" --target llama-cli -- -j"$CORES"

# --- Locate the resulting binary ---
if [[ -x "$LLAMA_BUILD_DIR/bin/llama-cli" ]]; then
  BINARY_PATH="$(cd "$LLAMA_BUILD_DIR/bin" && pwd -P)/llama-cli"
elif [[ -x "$LLAMA_BUILD_DIR/llama-cli" ]]; then
  BINARY_PATH="$(cd "$LLAMA_BUILD_DIR" && pwd -P)/llama-cli"
else
  die "llama-cli binary not found after build in '$LLAMA_BUILD_DIR'"
fi

log "✓ TokenSmith: Build successful: $BINARY_PATH"

# --- Persist path for TokenSmith consumers ---
mkdir -p "$PROJECT_ROOT/src"
printf '%s\n' "$BINARY_PATH" > "$PROJECT_ROOT/src/llama_path.txt"

# Also export env var for current shell
export LLAMA_CPP_BINARY="$BINARY_PATH"

log "TokenSmith: llama.cpp build complete!"
