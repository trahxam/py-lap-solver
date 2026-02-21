"""Benchmark LAP1015 CUDA and break down transfer vs solve time.

Timing keys returned by the CUDA binding:
- transfer_ms: host-table to device upload time
- solve_ms: CUDA solve loop time
- total_ms: full call time (includes setup overhead)
"""

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from get_cost_matrices import get_full_square_matrix


def _get_cuda_backend():
    """Import CUDA backend and verify timed API availability."""
    try:
        from py_lap_solver import _lap1015_cuda
    except ImportError as exc:
        raise RuntimeError(
            "py_lap_solver._lap1015_cuda is unavailable. Rebuild with CUDA enabled."
        ) from exc

    if not getattr(_lap1015_cuda, "HAS_CUDA", False):
        raise RuntimeError("CUDA backend is built but reports HAS_CUDA=False.")

    if not hasattr(_lap1015_cuda, "solve_lap_cuda_float_timed"):
        raise RuntimeError(
            "Timed CUDA API missing. Reinstall the updated extension "
            "(function: solve_lap_cuda_float_timed)."
        )

    return _lap1015_cuda


def _run_once(backend, matrix, use_epsilon: bool, use_pinned_memory: bool, max_devices: int):
    """Run a single timed solve and return timing dict."""
    _, timing = backend.solve_lap_cuda_float_timed(
        matrix,
        use_epsilon=use_epsilon,
        use_pinned_memory=use_pinned_memory,
        max_devices=max_devices,
        devices=[],
        silent=True,
    )
    return {
        "transfer_ms": float(timing["transfer_ms"]),
        "solve_ms": float(timing["solve_ms"]),
        "total_ms": float(timing["total_ms"]),
    }


def benchmark_cuda_breakdown(
    sizes: List[int],
    warmup: int,
    repeats: int,
    use_epsilon: bool,
    use_pinned_memory: bool,
    max_devices: int,
) -> Dict[str, List[float]]:
    """Benchmark LAP1015 CUDA timing breakdown vs matrix size."""
    backend = _get_cuda_backend()
    results = {"transfer_ms": [], "solve_ms": [], "total_ms": []}

    print("\n" + "=" * 70)
    print("LAP1015 CUDA TIMING BREAKDOWN")
    print("=" * 70)
    print(
        f"use_epsilon={use_epsilon}, use_pinned_memory={use_pinned_memory}, max_devices={max_devices}"
    )

    for size in sizes:
        matrix = get_full_square_matrix(size)
        for _ in range(warmup):
            _run_once(backend, matrix, use_epsilon, use_pinned_memory, max_devices)

        transfer_samples = []
        solve_samples = []
        total_samples = []

        for _ in range(repeats):
            timing = _run_once(backend, matrix, use_epsilon, use_pinned_memory, max_devices)
            transfer_samples.append(timing["transfer_ms"])
            solve_samples.append(timing["solve_ms"])
            total_samples.append(timing["total_ms"])

        transfer_ms = statistics.median(transfer_samples)
        solve_ms = statistics.median(solve_samples)
        total_ms = statistics.median(total_samples)

        results["transfer_ms"].append(transfer_ms)
        results["solve_ms"].append(solve_ms)
        results["total_ms"].append(total_ms)

        print(
            f"N={size:5d} | transfer={transfer_ms:9.3f} ms | "
            f"solve={solve_ms:9.3f} ms | total={total_ms:9.3f} ms"
        )

    return results


def write_csv(output_path: Path, sizes: List[int], results: Dict[str, List[float]]) -> None:
    """Write benchmark breakdown to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["matrix_size", "transfer_ms", "solve_ms", "total_ms"])
        for idx, size in enumerate(sizes):
            writer.writerow(
                [
                    size,
                    results["transfer_ms"][idx],
                    results["solve_ms"][idx],
                    results["total_ms"][idx],
                ]
            )


def plot_breakdown(output_path: Path, sizes: List[int], results: Dict[str, List[float]]) -> None:
    """Create linear and log-log subplots for transfer vs solve timing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting benchmark results.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    linear_ax, log_ax = axes

    series: List[Tuple[str, str]] = [
        ("transfer_ms", "Transfer to device"),
        ("solve_ms", "Solve on device"),
        ("total_ms", "Total call"),
    ]

    for key, label in series:
        y = results[key]
        linear_ax.plot(sizes, y, marker="o", linewidth=2, label=label)
        log_ax.plot(sizes, y, marker="o", linewidth=2, label=label)

    linear_ax.set_title("LAP1015 CUDA Timing Breakdown")
    linear_ax.set_xlabel("Matrix size N (for NxN)")
    linear_ax.set_ylabel("Runtime (ms)")
    linear_ax.grid(True, linestyle="--", alpha=0.4)
    linear_ax.legend()

    log_ax.set_title("LAP1015 CUDA Timing Breakdown (log-log)")
    log_ax.set_xlabel("Matrix size N (for NxN)")
    log_ax.set_ylabel("Runtime (ms)")
    log_ax.set_xscale("log")
    log_ax.set_yscale("log")
    log_ax.grid(True, which="both", linestyle="--", alpha=0.4)
    log_ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark LAP1015 CUDA and break down transfer vs solve timing."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Square matrix sizes to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per size.")
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed repeats per size (median used)."
    )
    parser.add_argument(
        "--no-use-epsilon",
        dest="use_epsilon",
        action="store_false",
        default=True,
        help="Disable epsilon scaling.",
    )
    parser.add_argument(
        "--no-use-pinned-memory",
        dest="use_pinned_memory",
        action="store_false",
        default=True,
        help="Disable pinned host memory in CUDA table setup.",
    )
    parser.add_argument("--max-devices", type=int, default=1, help="Maximum CUDA devices to use.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory for CSV/PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run CUDA timing breakdown benchmark."""
    args = parse_args()
    results = benchmark_cuda_breakdown(
        sizes=args.sizes,
        warmup=args.warmup,
        repeats=args.repeats,
        use_epsilon=args.use_epsilon,
        use_pinned_memory=args.use_pinned_memory,
        max_devices=args.max_devices,
    )

    csv_path = args.output_dir / "lap1015_cuda_timing_breakdown.csv"
    png_path = args.output_dir / "lap1015_cuda_timing_breakdown.png"

    write_csv(csv_path, args.sizes, results)
    plot_breakdown(png_path, args.sizes, results)

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
