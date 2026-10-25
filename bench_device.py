"""Benchmark device performance for Lagrangian layer operations.

Executes synthetic projected gradient-descent workloads matching the RL-LCR
Lagrangian layer to evaluate CPU vs Apple Silicon MPS execution times across
varying batch sizes and network edge counts.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from rllcr.utils.device import device_sync, to_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for device benchmarking.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark Lagrangian layer workload on CPU vs Apple Silicon MPS."
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 8, 64, 256, 1024],
        help="Batch sizes to evaluate.",
    )
    parser.add_argument(
        "--edges",
        type=int,
        nargs="+",
        default=[50, 200, 800],
        help="Number of network edges to evaluate.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of projected gradient descent iterations.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/device_benchmark.csv",
        help="Output CSV file path.",
    )
    return parser.parse_args()


def run_lagrangian_benchmark(
    batch_size: int,
    n_edges: int,
    iters: int,
    device: torch.device,
) -> float:
    """Run synthetic Lagrangian projected gradient-descent optimization on a device.

    Workload formula:
        loss = (x*c).sum(1)
             + lamE * relu(x @ e - Emax)
             + lamD * relu(x @ dly - Dmax)
             + rho * ((x @ Inc.T - b)**2).sum(1)

    Args:
        batch_size: Route/packet batch dimension B.
        n_edges: Link/edge count dimension E.
        iters: Number of gradient descent steps.
        device: PyTorch device ('cpu' or 'mps').

    Returns:
        float: Elapsed wall time in seconds.
    """
    n_nodes = max(10, n_edges // 5)

    # Synthetic parameters (float32 only)
    c = torch.rand(n_edges, dtype=torch.float32)
    e = torch.rand(n_edges, dtype=torch.float32)
    dly = torch.rand(n_edges, dtype=torch.float32)
    inc = torch.randn(n_nodes, n_edges, dtype=torch.float32)
    b = torch.randn(batch_size, n_nodes, dtype=torch.float32)

    c_dev = to_device(c, device)
    e_dev = to_device(e, device)
    dly_dev = to_device(dly, device)
    inc_t_dev = to_device(inc.T, device)
    b_dev = to_device(b, device)

    lam_e = 1.0
    lam_d = 1.0
    rho = 1.0
    e_max = 0.5 * float(e.sum().item())
    d_max = 0.5 * float(dly.sum().item())
    lr = 0.01

    x = torch.rand(batch_size, n_edges, dtype=torch.float32, device=device, requires_grad=True)

    # Warmup step
    for _ in range(2):
        term1 = (x * c_dev).sum(dim=1)
        term2 = lam_e * torch.relu(torch.matmul(x, e_dev) - e_max)
        term3 = lam_d * torch.relu(torch.matmul(x, dly_dev) - d_max)
        flow_diff = torch.matmul(x, inc_t_dev) - b_dev
        term4 = rho * (flow_diff**2).sum(dim=1)
        step_loss = (term1 + term2 + term3 + term4).sum()
        step_loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
            x.clamp_(0.0, 1.0)
            x.grad.zero_()
    device_sync(device)

    # Timed loop
    device_sync(device)
    start_time = time.perf_counter()
    for _ in range(iters):
        term1 = (x * c_dev).sum(dim=1)
        term2 = lam_e * torch.relu(torch.matmul(x, e_dev) - e_max)
        term3 = lam_d * torch.relu(torch.matmul(x, dly_dev) - d_max)
        flow_diff = torch.matmul(x, inc_t_dev) - b_dev
        term4 = rho * (flow_diff**2).sum(dim=1)
        step_loss = (term1 + term2 + term3 + term4).sum()
        step_loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
            x.clamp_(0.0, 1.0)
            x.grad.zero_()
    device_sync(device)
    return time.perf_counter() - start_time


def main() -> None:
    """Execute device benchmark across parameter grid."""
    args = parse_args()
    has_mps = torch.backends.mps.is_available()

    print("================================================================")
    print("RL-LCR Device Benchmark: Lagrangian Layer (CPU vs MPS)")
    print(f"Batches: {args.batches} | Edges: {args.edges} | Iterations: {args.iters}")
    print("================================================================")

    results: list[dict[str, Any]] = []

    for b in args.batches:
        for e in args.edges:
            print(f"Running (Batch={b:4d}, Edges={e:3d}) ... ", end="", flush=True)

            cpu_time = run_lagrangian_benchmark(b, e, args.iters, torch.device("cpu"))

            if has_mps:
                mps_time = run_lagrangian_benchmark(b, e, args.iters, torch.device("mps"))
                speedup = cpu_time / mps_time if mps_time > 0 else 0.0
                faster = "mps" if mps_time < cpu_time else "cpu"
                print(
                    f"CPU: {cpu_time * 1000:6.1f} ms | MPS: {mps_time * 1000:6.1f} ms | "
                    f"Speedup: {speedup:5.2f}x [{faster.upper()}]"
                )
            else:
                mps_time = float("inf")
                speedup = 0.0
                faster = "cpu"
                print(f"CPU: {cpu_time * 1000:6.1f} ms | MPS: N/A")

            results.append(
                {
                    "batch_size": b,
                    "n_edges": e,
                    "iters": args.iters,
                    "cpu_time_s": cpu_time,
                    "mps_time_s": mps_time,
                    "speedup": speedup,
                    "faster_device": faster,
                }
            )

    df = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("================================================================")
    print(f"Saved benchmark results to {out_path}")

    # Identify smallest (B, E) where MPS is faster
    mps_faster_df = df[df["faster_device"] == "mps"].sort_values(
        by=["batch_size", "n_edges"]
    )
    if not mps_faster_df.empty:
        best_row = mps_faster_df.iloc[0]
        min_b = int(best_row["batch_size"])
        min_e = int(best_row["n_edges"])
        best_speedup = float(best_row["speedup"])
        print(
            f"\nOptimal threshold: Smallest (B, E) where MPS is faster -> "
            f"batch_size={min_b}, n_edges={min_e} (Speedup: {best_speedup:.2f}x)"
        )
        print(
            f"\n[RECOMMENDATION] Suggested config settings for configs/lagrangian.yaml:\n"
            f"  mps_min_batch: {min_b}\n"
            f"  mps_min_edges: {min_e}\n"
        )
        print("Would you like to write these values into configs/lagrangian.yaml? (Ask for confirmation)")
    else:
        print("\nMPS was not faster for tested sizes; recommend CPU fallback for tested ranges.")


if __name__ == "__main__":
    main()
