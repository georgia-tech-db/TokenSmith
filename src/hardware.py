"""
Runtime hardware detection for TokenSmith.

Detects the platform, available GPU backends, and recommends the appropriate
n_gpu_layers value for llama.cpp. Called once at startup before any models load.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareInfo:
    os_name: str              # "macOS", "Linux", "Windows", "Unknown"
    arch: str                 # "arm64", "x86_64", etc.
    metal_compiled: bool      # True if llama_cpp was built with Metal support
    cuda_available: bool      # True if nvidia-smi is reachable
    recommended_n_gpu_layers: int  # -1 (GPU) or 0 (CPU-only)
    backend_name: str         # Human-readable: "Metal", "CUDA", "CPU"

    def print_summary(self):
        accel = f"{self.backend_name}"
        print(
            f"[TokenSmith] Hardware: {self.os_name} {self.arch} | "
            f"Backend: {accel} | "
            f"n_gpu_layers: {self.recommended_n_gpu_layers}"
        )


def _check_metal_compiled() -> bool:
    """
    Ask llama_cpp whether it was compiled with Metal support.
    llama_print_system_info() returns a string like "... METAL = 1 | ..."
    """
    try:
        import llama_cpp
        info: str = llama_cpp.llama_print_system_info().decode(
            "utf-8", errors="ignore"
        )
        return "METAL = 1" in info
    except Exception:
        return False


def _check_cuda_available() -> bool:
    """
    Check for a reachable nvidia-smi binary, the same way the build scripts do.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def detect_backend() -> HardwareInfo:
    """
    Detect the platform and determine the best inference backend.

    Decision logic:
      - macOS arm64  → Metal (if compiled in)  → n_gpu_layers = -1
      - macOS x86_64 → CPU only                → n_gpu_layers = 0
      - Linux + NVIDIA GPU detected → CUDA     → n_gpu_layers = -1
      - Linux, no NVIDIA            → CPU      → n_gpu_layers = 0
      - Windows                     → CPU      → n_gpu_layers = 0
      - Unknown                     → CPU      → n_gpu_layers = 0
    """
    system = platform.system()   # "Darwin", "Linux", "Windows"
    machine = platform.machine() # "arm64", "x86_64", "AMD64", etc.

    # Normalise OS label
    if system == "Darwin":
        os_name = "macOS"
    elif system == "Linux":
        os_name = "Linux"
    elif system == "Windows":
        os_name = "Windows"
    else:
        os_name = system or "Unknown"

    metal_compiled = _check_metal_compiled()
    cuda_available = _check_cuda_available()

    # --- Decision tree ---
    if os_name == "macOS" and machine == "arm64":
        if metal_compiled:
            return HardwareInfo(
                os_name=os_name,
                arch=machine,
                metal_compiled=True,
                cuda_available=False,
                recommended_n_gpu_layers=-1,
                backend_name="Metal (Apple Silicon)",
            )
        else:
            # arm64 Mac but llama_cpp wasn't compiled with Metal — fall back to CPU
            return HardwareInfo(
                os_name=os_name,
                arch=machine,
                metal_compiled=False,
                cuda_available=False,
                recommended_n_gpu_layers=0,
                backend_name="CPU (Metal not compiled in)",
            )

    elif os_name == "macOS" and machine == "x86_64":
        # Intel Mac — Metal is only for Apple Silicon; AMD/Intel GPUs not supported
        return HardwareInfo(
            os_name=os_name,
            arch=machine,
            metal_compiled=metal_compiled,  # may be True in a bad pre-built wheel
            cuda_available=False,
            recommended_n_gpu_layers=0,
            backend_name="CPU (Intel Mac — Metal unsupported)",
        )

    elif os_name == "Linux" and cuda_available:
        return HardwareInfo(
            os_name=os_name,
            arch=machine,
            metal_compiled=False,
            cuda_available=True,
            recommended_n_gpu_layers=-1,
            backend_name="CUDA",
        )

    else:
        # Linux (no NVIDIA), Windows, or unknown
        return HardwareInfo(
            os_name=os_name,
            arch=machine,
            metal_compiled=False,
            cuda_available=False,
            recommended_n_gpu_layers=0,
            backend_name="CPU",
        )


def apply_hardware_config(cfg) -> None:
    """
    If cfg.device == "auto", override cfg.n_gpu_layers with the value
    recommended by detect_backend() and print a hardware summary.

    If cfg.device is set explicitly ("cpu", "metal", "cuda"), honour it
    and still print a summary so the user knows what's happening.

    Mutates cfg in-place; returns nothing.
    """
    hardware = detect_backend()

    if cfg.device == "auto":
        cfg.n_gpu_layers = hardware.recommended_n_gpu_layers
        hardware.print_summary()
    elif cfg.device == "cpu":
        cfg.n_gpu_layers = 0
        print(
            f"[TokenSmith] Hardware: {hardware.os_name} {hardware.arch} | "
            f"Backend: CPU (forced via config) | n_gpu_layers: 0"
        )
    elif cfg.device == "metal":
        if hardware.metal_compiled and hardware.os_name == "macOS" and hardware.arch == "arm64":
            cfg.n_gpu_layers = -1
            print(
                f"[TokenSmith] Hardware: {hardware.os_name} {hardware.arch} | "
                f"Backend: Metal (forced via config) | n_gpu_layers: -1"
            )
        else:
            print(
                f"[TokenSmith] WARNING: device='metal' requested but Metal is not "
                f"available on this machine ({hardware.os_name} {hardware.arch}). "
                f"Falling back to CPU."
            )
            cfg.n_gpu_layers = 0
    elif cfg.device == "cuda":
        if hardware.cuda_available:
            cfg.n_gpu_layers = -1
            print(
                f"[TokenSmith] Hardware: {hardware.os_name} {hardware.arch} | "
                f"Backend: CUDA (forced via config) | n_gpu_layers: -1"
            )
        else:
            print(
                f"[TokenSmith] WARNING: device='cuda' requested but no NVIDIA GPU "
                f"detected. Falling back to CPU."
            )
            cfg.n_gpu_layers = 0
    else:
        print(
            f"[TokenSmith] WARNING: Unknown device='{cfg.device}'. "
            f"Falling back to CPU."
        )
        cfg.n_gpu_layers = 0
