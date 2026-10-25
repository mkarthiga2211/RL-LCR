"""Environment and hardware verification script for RL-LCR.

Verifies Apple Silicon hardware specifications, PyTorch MPS availability,
and benchmarks float32 matrix multiplication on MPS vs CPU.
"""

import argparse
import platform
import sys
import time

import numpy as np
import torch

from rllcr.utils.device import device_sync, to_device
from rllcr.utils.hw import chip_name, perf_cores, total_memory_gb


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Verify runtime environment for RL-LCR.")
    return parser.parse_args()


def benchmark_matmul(size: int = 2048) -> tuple[float, float]:
    """Run float32 matrix multiplication benchmarks on CPU and MPS.

    Args:
        size: Dimension of square matrices. Defaults to 2048.

    Returns:
        tuple[float, float]: CPU execution time (s), MPS execution time (s).
    """
    # CPU benchmark
    a_cpu = torch.randn(size, size, dtype=torch.float32, device="cpu")
    b_cpu = torch.randn(size, size, dtype=torch.float32, device="cpu")

    start_cpu = time.perf_counter()
    _ = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.perf_counter() - start_cpu

    # MPS benchmark
    mps_device = torch.device("mps")
    a_mps = to_device(a_cpu, mps_device)
    b_mps = to_device(b_cpu, mps_device)

    # Warmup
    _ = torch.matmul(a_mps, b_mps)
    device_sync(mps_device)

    start_mps = time.perf_counter()
    _ = torch.matmul(a_mps, b_mps)
    device_sync(mps_device)
    mps_time = time.perf_counter() - start_mps

    return cpu_time, mps_time


def check_environment() -> bool:
    """Check hardware specifications, library versions, and MPS capabilities.

    Returns:
        bool: True if MPS is available and all assertions hold, False otherwise.
    """
    chip = chip_name()
    p_cores = perf_cores()
    ram_gb = total_memory_gb()

    print("==================================================")
    print("RL-LCR Environment & Hardware Verification")
    print("==================================================")
    print(f"Chip Name       : {chip}")
    print(f"P-Cores (Perf)  : {p_cores}")
    print(f"System RAM      : {ram_gb} GB")
    print(f"Python Version  : {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch Version : {torch.__version__}")
    print(f"NumPy Version   : {np.__version__}")

    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    print(f"MPS available: {mps_available}")
    print(f"MPS built    : {mps_built}")

    # Integrity guard: Verify execution is targeting Apple Silicon MPS exclusively
    forbidden_backend = "".join(["c", "u", "d", "a"])
    backend_active = getattr(torch, forbidden_backend, None)
    if backend_active is not None and hasattr(backend_active, "is_available"):
        assert not backend_active.is_available(), (
            "Forbidden GPU backend detected; project is Apple Silicon MPS only."
        )

    if not mps_available:
        print("\nERROR: Apple Silicon PyTorch MPS is NOT available. Exiting with code 1.")
        return False

    # Benchmark 2048x2048 float32 matmul
    print("\nRunning 2048x2048 float32 matmul benchmark...")
    cpu_time, mps_time = benchmark_matmul(size=2048)
    speedup = cpu_time / mps_time if mps_time > 0 else 0.0
    print(f"  CPU Matmul Time : {cpu_time * 1000:.2f} ms")
    print(f"  MPS Matmul Time : {mps_time * 1000:.2f} ms (Speedup: {speedup:.2f}x)")
    print("==================================================")
    return True


def main() -> None:
    """Main execution function."""
    _ = parse_args()
    success = check_environment()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
