"""Benchmark batched solving runtime vs batch size for fixed problem size."""

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


def _get_batched_solvers():
    """Get batch-capable solvers used in this benchmark."""
    available = Solvers.get_available_solvers()
    ordered_names = ["BatchedScipyOMP", "BatchedScipySequential", "ScipyMP8"]
    return {name: available[name] for name in ordered_names if name in available}


def _default_batch_sizes(max_batch_size: int) -> List[int]:
    """Generate batch sizes 1,2,4,... up to max_batch_size."""
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    sizes = []
    bs = 1
    while bs <= max_batch_size:
        sizes.append(bs)
        bs *= 2
    return sizes


def benchmark_batch_size_scaling(
    problem_size: int,
    batch_sizes: List[int],
    warmup: int,
    repeats: int,
) -> Dict[str, List[float]]:
    """Benchmark batch_solve runtime vs batch size for fixed square problem size."""
    solvers = _get_batched_solvers()
    results = {name: [] for name in solvers}

    print("\n" + "=" * 70)
    print("BATCHED SCALING BENCHMARK (RUNTIME VS BATCH SIZE)")
    print("=" * 70)
    print(f"Problem size: ({problem_size}, {problem_size})")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Solvers: {', '.join(solvers.keys())}")

    for batch_size in batch_sizes:
        matrices = get_full_square_matrix(problem_size, batch_size=batch_size)
        print(f"\nBatch size: {batch_size}")
        for name, solver in solvers.items():
            runtime_ms = _time_call(
                lambda: solver.batch_solve(matrices),
                warmup=warmup,
                repeats=repeats,
            )
            results[name].append(runtime_ms)
            print(f"{name:25s}: {runtime_ms:9.3f} ms")

    return results


def write_csv(output_path: Path, batch_sizes: List[int], results: Dict[str, List[float]]) -> None:
    """Write scaling results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    solver_names = sorted(results.keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", *solver_names])
        for idx, batch_size in enumerate(batch_sizes):
            writer.writerow([batch_size, *[results[name][idx] for name in solver_names]])


def plot_runtime_vs_batch_size(
    output_path: Path,
    batch_sizes: List[int],
    results: Dict[str, List[float]],
    problem_size: int,
) -> None:
    """Plot runtime-vs-batch-size with linear and log-log subplots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for benchmark plotting.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    linear_ax, log_ax = axes

    for solver_name in sorted(results.keys()):
        runtimes = results[solver_name]
        linear_ax.plot(batch_sizes, runtimes, marker="o", linewidth=2, label=solver_name)
        log_ax.plot(batch_sizes, runtimes, marker="o", linewidth=2, label=solver_name)

    linear_ax.set_title(f"Batched Runtime vs Batch Size (N={problem_size})")
    linear_ax.set_xlabel("Batch size")
    linear_ax.set_ylabel("Runtime (ms)")
    linear_ax.grid(True, linestyle="--", alpha=0.4)
    linear_ax.legend()

    log_ax.set_title(f"Batched Runtime vs Batch Size (log-log, N={problem_size})")
    log_ax.set_xlabel("Batch size")
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
        description="Benchmark batched runtime vs batch size for fixed problem size."
    )
    parser.add_argument(
        "--problem-size",
        type=int,
        default=256,
        help="Square matrix size N for NxN problems.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1024,
        help="Maximum batch size when --batch-sizes is not provided.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit batch sizes. If omitted: 1,2,4,... up to --max-batch-size.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per point.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per point (median used).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory for CSV/PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run batch-size scaling benchmark workflow."""
    args = parse_args()
    batch_sizes = args.batch_sizes if args.batch_sizes is not None else _default_batch_sizes(args.max_batch_size)

    results = benchmark_batch_size_scaling(
        problem_size=args.problem_size,
        batch_sizes=batch_sizes,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    csv_path = args.output_dir / f"batch_runtime_vs_batch_size_n{args.problem_size}.csv"
    png_path = args.output_dir / f"batch_runtime_vs_batch_size_n{args.problem_size}.png"
    write_csv(csv_path, batch_sizes, results)
    plot_runtime_vs_batch_size(png_path, batch_sizes, results, problem_size=args.problem_size)

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
