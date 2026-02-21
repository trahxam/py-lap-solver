"""Benchmark LAP solvers and generate runtime-vs-matrix-size graphs.

This script benchmarks:
- single problem solving via ``solve_single``
- batched solving via ``batch_solve`` (CUDA solvers excluded)

Only square matrices are benchmarked.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List

from get_cost_matrices import get_full_square_matrix

from py_lap_solver.solvers import Solvers


def _time_call(func: Callable[[], object], warmup: int, repeats: int) -> float:
    """Return median runtime in milliseconds for a callable."""
    for _ in range(warmup):
        func()

    samples_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        samples_ms.append((end - start) * 1000.0)

    return statistics.median(samples_ms)


def _get_single_solvers():
    """Get solvers to benchmark for solve_single()."""
    available = Solvers.get_available_solvers()
    ordered_names = ["Scipy", "Lap1015", "Lap1015CUDA"]
    return {name: available[name] for name in ordered_names if name in available}


def _get_batched_solvers():
    """Get solvers to benchmark for batch_solve()."""
    available = Solvers.get_available_solvers()
    ordered_names = ["BatchedScipyOMP", "BatchedScipySequential", "ScipyMP8"]
    return {name: available[name] for name in ordered_names if name in available}


def benchmark_single(matrix_sizes: List[int], warmup: int, repeats: int) -> Dict[str, List[float]]:
    """Benchmark solve_single runtime (ms) versus matrix size."""
    solvers = _get_single_solvers()
    results = {name: [] for name in solvers}

    print("\n" + "=" * 70)
    print("SINGLE PROBLEM BENCHMARKS (solve_single)")
    print("=" * 70)
    print(f"Solvers: {', '.join(solvers.keys())}")

    for size in matrix_sizes:
        matrix = get_full_square_matrix(size)
        print(f"\nMatrix size: ({size}, {size})")
        for name, solver in solvers.items():
            runtime_ms = _time_call(
                lambda s=solver, m=matrix: s.solve_single(m),
                warmup=warmup,
                repeats=repeats,
            )
            results[name].append(runtime_ms)
            print(f"{name:25s}: {runtime_ms:9.3f} ms")

    return results


def benchmark_batched(
    matrix_sizes: List[int], batch_size: int, warmup: int, repeats: int
) -> Dict[str, List[float]]:
    """Benchmark batch_solve runtime (ms) versus matrix size for a fixed batch size."""
    solvers = _get_batched_solvers()
    results = {name: [] for name in solvers}

    print("\n" + "=" * 70)
    print("BATCHED PROBLEM BENCHMARKS (batch_solve, non-CUDA)")
    print("=" * 70)
    print(f"Batch size: {batch_size}")
    print(f"Solvers: {', '.join(solvers.keys())}")

    for size in matrix_sizes:
        matrices = get_full_square_matrix(size, batch_size=batch_size)
        print(f"\nMatrix size: ({size}, {size}), batch size: {batch_size}")
        for name, solver in solvers.items():
            runtime_ms = _time_call(
                lambda s=solver, mats=matrices: s.batch_solve(mats),
                warmup=warmup,
                repeats=repeats,
            )
            results[name].append(runtime_ms)
            print(f"{name:25s}: {runtime_ms:9.3f} ms")

    return results


def write_csv(output_path: Path, matrix_sizes: List[int], results: Dict[str, List[float]]) -> None:
    """Write benchmark results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    solver_names = sorted(results.keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["matrix_size", *solver_names])
        for idx, size in enumerate(matrix_sizes):
            writer.writerow([size, *[results[name][idx] for name in solver_names]])


def plot_runtime_vs_size(
    output_path: Path,
    matrix_sizes: List[int],
    results: Dict[str, List[float]],
    title: str,
) -> None:
    """Save runtime-vs-size plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for benchmark plotting. "
            "Install it in your environment and rerun."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for solver_name in sorted(results.keys()):
        plt.plot(matrix_sizes, results[solver_name], marker="o", linewidth=2, label=solver_name)

    plt.title(title)
    plt.xlabel("Matrix size N (for NxN)")
    plt.ylabel("Runtime (ms)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_single_runtime_with_log_subplot(
    output_path: Path, matrix_sizes: List[int], results: Dict[str, List[float]]
) -> None:
    """Save single-solve runtime plot with linear and log-log subplots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for benchmark plotting. "
            "Install it in your environment and rerun."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    linear_ax, log_ax = axes

    for solver_name in sorted(results.keys()):
        runtimes = results[solver_name]
        linear_ax.plot(matrix_sizes, runtimes, marker="o", linewidth=2, label=solver_name)
        log_ax.plot(matrix_sizes, runtimes, marker="o", linewidth=2, label=solver_name)

    linear_ax.set_title("Single Solve Runtime vs Matrix Size")
    linear_ax.set_xlabel("Matrix size N (for NxN)")
    linear_ax.set_ylabel("Runtime (ms)")
    linear_ax.grid(True, linestyle="--", alpha=0.4)
    linear_ax.legend()

    log_ax.set_title("Single Solve Runtime vs Matrix Size (log-log)")
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
    parser = argparse.ArgumentParser(description="Benchmark LAP solvers and generate plots.")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "all"],
        default="all",
        help="Which benchmark(s) to run.",
    )
    parser.add_argument(
        "--single-sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Square matrix sizes for single-problem benchmarks.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024],
        help="Square matrix sizes for batched benchmarks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for batched benchmarks.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup calls per solver/size.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per solver/size (median is reported).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory where CSV and PNG outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run benchmark workflow."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("LAP Solver Benchmark")
    print("=" * 70)
    Solvers.print_available_solvers()

    if args.mode in {"single", "all"}:
        single_results = benchmark_single(
            args.single_sizes, warmup=args.warmup, repeats=args.repeats
        )
        single_csv = args.output_dir / "single_runtime_vs_size.csv"
        single_png = args.output_dir / "single_runtime_vs_size.png"
        write_csv(single_csv, args.single_sizes, single_results)
        plot_single_runtime_with_log_subplot(single_png, args.single_sizes, single_results)
        print(f"\nWrote: {single_csv}")
        print(f"Wrote: {single_png}")

    if args.mode in {"batch", "all"}:
        batch_results = benchmark_batched(
            args.batch_sizes,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        batch_csv = args.output_dir / f"batch_runtime_vs_size_bs{args.batch_size}.csv"
        batch_png = args.output_dir / f"batch_runtime_vs_size_bs{args.batch_size}.png"
        write_csv(batch_csv, args.batch_sizes, batch_results)
        plot_runtime_vs_size(
            batch_png,
            args.batch_sizes,
            batch_results,
            f"Batched Solve Runtime vs Matrix Size (batch size={args.batch_size})",
        )
        print(f"\nWrote: {batch_csv}")
        print(f"Wrote: {batch_png}")


if __name__ == "__main__":
    main()
