#!/usr/bin/env python3
"""
Detect existing llama.cpp installation for TokenSmith.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

def find_llama_binary():
    """Find llama.cpp binary in various locations."""
    
    # Check environment variable first
    if env_path := os.getenv("LLAMA_CPP_BINARY"):
        if Path(env_path).exists():
            print(f"Found llama.cpp via LLAMA_CPP_BINARY: {env_path}")
            return env_path
    
    # Common binary names
    binary_names = ["llama-cli", "llama.cpp", "main"]
    
    # Check PATH
    for name in binary_names:
        if binary_path := shutil.which(name):
            print(f"Found llama.cpp in PATH: {binary_path}")
            return binary_path
    
    # Check common installation locations
    common_paths = [
        "/usr/local/bin/llama-cli",
        "/opt/homebrew/bin/llama-cli",  # Homebrew on Apple Silicon
        "/usr/bin/llama-cli",
        Path.home() / ".local/bin/llama-cli",
        "./llama.cpp/llama-cli",
        "./build/bin/llama-cli",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            print(f"Found llama.cpp at: {path}")
            return str(path)
    
    return None

def test_binary(binary_path):
    """Test if binary works and supports required flags."""
    try:
        result = subprocess.run(
            [binary_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and "--temp" in result.stdout:
            print(f"✓ Binary test passed: {binary_path}")
            return True
        else:
            print(f"✗ Binary test failed: {binary_path}")
            return False
    except Exception as e:
        print(f"✗ Binary test error: {e}")
        return False

def main():
    """Main detection logic for TokenSmith."""
    print("TokenSmith: Detecting llama.cpp installation...")
    
    binary_path = find_llama_binary()
    
    if not binary_path:
        print("No existing llama.cpp installation found.")
        sys.exit(1)
    
    if not test_binary(binary_path):
        print("Found binary but it doesn't work properly.")
        sys.exit(1)
    
    # Write to src/ directory for TokenSmith (where config.yaml will be)
    config_dir = Path("src")
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "llama_path.txt", "w") as f:
        f.write(binary_path)
    
    print(f"✓ TokenSmith: llama.cpp ready: {binary_path}")
    print("Skipping build step.")
    sys.exit(0)

if __name__ == "__main__":
    main()
