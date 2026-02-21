# py-lap-solver

A unified Python framework for Linear Assignment Problem (LAP) solvers.

## Overview

`py-lap-solver` provides a common interface for multiple LAP solver implementations, ranging from pure Python (scipy) to optimized C++ implementations with OpenMP and CUDA support.

The Linear Assignment Problem seeks to find an optimal assignment between two sets given a cost matrix, minimizing (or maximizing) the total cost of the assignment.

## Installation

Install from pypi
```
pip install py-lap-solver
```

Or install from source

```bash

git clone git@github.com:nikitapond/py-lap-solver.git

# Install the package in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Install with Pixi

For a reproducible environment (Python, C/C++ toolchain, CMake, lint/test tools), use Pixi:

```bash
# Install/update the environment from pixi.toml.
# This also installs py-lap-solver in editable mode.
pixi install
```

On Linux, the Pixi config installs `cuda-nvcc` so CUDA bindings can be built.
If you still get CPU-only builds, verify:

```bash
pixi run which nvcc
pixi run nvcc --version
```

Useful Pixi tasks:

```bash
pixi run format
pixi run lint
pixi run lint-fix
pixi run test
pixi run build-wheel
```

## Features

- **Unified Interface**: Common API across all solver implementations
- **Multiple Backends**:
  - **ScipySolver**: Pure Python implementation using scipy's Hungarian algorithm
  - **BatchedScipySolver**: C++ implementation with OpenMP parallelization for batch processing
  - **Lap1015Solver**: Highly optimized C++ implementation (shortest augmenting path algorithm)
- **Batch Processing**: Solve multiple LAP instances efficiently with OpenMP parallelization
- **Flexible Input**: Support for square and rectangular cost matrices
- **Float32 Execution**: Cost matrices are coerced to `float32` in all solver wrappers
- **Optional GPU Support**: CUDA-backed LAP1015 solver via optional `_lap1015_cuda` extension

## Quick Start

### Using the Solver Registry (Recommended)

The easiest way to use the solvers is through the pre-configured `Solvers` registry:

```python
from py_lap_solver.solvers import Solvers
import numpy as np

# Create a batch of cost matrices
batch_matrices = np.random.rand(100, 500, 500).astype(np.float32)

# Use the fastest available solver with OpenMP parallelization
# This will give you ~6x speedup over sequential processing
assignments = Solvers.BatchedScipyOMP.batch_solve(batch_matrices)

# For single problems, use the standard scipy solver
cost_matrix = np.random.rand(500, 500).astype(np.float32)
single_assignment = Solvers.Scipy.solve_single(cost_matrix)
```

Available solvers in the registry:
- `Solvers.Scipy` - Pure Python scipy implementation (always available)
- `Solvers.BatchedScipyOMP` - C++ scipy with OpenMP batch parallelization
- `Solvers.BatchedScipySequential` - C++ scipy without parallelization
- `Solvers.Lap1015OMP` - LAP1015 algorithm with OpenMP (limited benefit)
- `Solvers.Lap1015Sequential` - LAP1015 algorithm without OpenMP
- `Solvers.Lap1015CUDA` - LAP1015 CUDA backend (only when CUDA extension is built)

### Manual Configuration

You can also instantiate solvers directly with custom parameters:

```python
from py_lap_solver.solvers import ScipySolver, BatchedScipySolver, Lap1015Solver
import numpy as np

# Use scipy solver (always available)
scipy_solver = ScipySolver()
assignments = scipy_solver.solve_single(cost_matrix)

# Use batched scipy solver with runtime OpenMP control
if BatchedScipySolver.is_available():
    # Create solver with OpenMP enabled (default)
    batch_solver_omp = BatchedScipySolver(use_openmp=True)

    # Create solver without OpenMP for comparison
    batch_solver_seq = BatchedScipySolver(use_openmp=False)

    batch_matrices = np.random.rand(10, 100, 100).astype(np.float32)
    fast_assignments = batch_solver_omp.batch_solve(batch_matrices)  # ~6x faster
    slow_assignments = batch_solver_seq.batch_solve(batch_matrices)

# Use LAP1015 solver
if Lap1015Solver.is_available():
    # Note: OpenMP provides minimal benefit for LAP1015 due to algorithm structure
    lap_solver = Lap1015Solver(use_openmp=False)
    assignments = lap_solver.solve_single(cost_matrix)

# Use LAP1015 CUDA solver
if Lap1015Solver.has_cuda():
    lap_cuda_solver = Lap1015Solver(use_cuda=True, use_openmp=False)
    assignments_cuda = lap_cuda_solver.solve_single(cost_matrix)
```

`Lap1015CUDA` preloads the cost matrix to GPU memory before solving, so it does not
stream rows from host memory during the solve loop.

### Return Format

All solvers return assignments in a consistent format:
- **Single problem**: 1D array of shape `(N,)` where `result[i]` is the column assigned to row `i`
- **Batch problem**: 2D array of shape `(B, N)` where `result[b, i]` is the column assigned to row `i` in batch `b`
- Unassigned rows are marked with `-1` (or custom `unassigned_value`)

```python
import numpy as np
from py_lap_solver.solvers import Solvers

cost_matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
assignments = Solvers.Scipy.solve_single(cost_matrix)
# assignments = [1, 0]  (row 0 -> col 1, row 1 -> col 0)
```

### Building with C++ Extensions

To enable the optimized C++ solvers, you need CMake and build tools:

```bash
# Install build dependencies
pip install scikit-build-core pybind11

# Build and install with C++ extensions
pip install -e . --no-build-isolation

# On macOS, you may need to install libomp for OpenMP support
brew install libomp
```

With Pixi, those build dependencies are managed by `pixi.toml`, so usually:

```bash
pixi install
```

After changing C++ binding/build files, you can force a rebuild/reinstall with:

```bash
pixi run install
```

If CUDA Toolkit (with `nvcc`) is available at build time, the optional
`py_lap_solver._lap1015_cuda` module is built automatically. You can then use:

```python
from py_lap_solver.solvers import Lap1015Solver

solver = Lap1015Solver(use_cuda=True, cuda_max_devices=1)
result = solver.solve_single(cost_matrix)
```

Check CUDA backend availability after install:

```bash
pixi run python -c "from py_lap_solver.solvers import Lap1015Solver; print(Lap1015Solver.has_cuda(), Lap1015Solver.cuda_device_count())"
```

If CMake still reports "Looking for a CUDA compiler - NOTFOUND", force the compiler path:

```bash
CUDACXX="$(pixi run which nvcc)" pixi run install
```

If CUDA configure fails with `Unsupported gpu architecture 'compute_50'`, force modern arch detection:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native" pixi run install
```

If build fails with `Unsupported gpu architecture 'compute_70'`, your toolkit is too new
for that GPU architecture (common with CUDA 13+). Use CUDA 12.x in Pixi and reinstall:

```bash
pixi install
pixi run nvcc --version
pixi run install
```

### OpenMP Runtime Control

All C++ solvers support runtime OpenMP control through the `use_openmp` parameter:

```python
from py_lap_solver.solvers import BatchedScipySolver
import numpy as np

# Create solver with OpenMP enabled
solver_parallel = BatchedScipySolver(use_openmp=True)

# Create solver without OpenMP
solver_sequential = BatchedScipySolver(use_openmp=False)

batch = np.random.rand(100, 500, 500).astype(np.float32)

# Parallel: ~126ms for 100 matrices
assignments_fast = solver_parallel.batch_solve(batch)

# Sequential: ~762ms for 100 matrices
assignments_slow = solver_sequential.batch_solve(batch)
```

**Why OpenMP Helps for Batched Scipy but Not LAP1015**:
- **BatchedScipySolver**: Each matrix in the batch is independent → perfect parallelization with `#pragma omp parallel for`
- **LAP1015Solver**: Complex intra-matrix data dependencies → synchronization barriers dominate, killing performance

### Recommended Usage Patterns

```python
from py_lap_solver.solvers import Solvers
import numpy as np

# Pattern 1: Batch processing (FAST - use OpenMP)
batch_matrices = np.random.rand(1000, 100, 100).astype(np.float32)
assignments = Solvers.BatchedScipyOMP.batch_solve(batch_matrices)

# Pattern 2: Single large problem (no parallelization benefit)
single_matrix = np.random.rand(5000, 5000).astype(np.float32)
assignment = Solvers.Scipy.solve_single(single_matrix)  # or BatchedScipySequential

# Pattern 3: Many small problems in a loop
for i in range(1000):
    matrix = generate_matrix()
    # BAD: Calling solve_single in a loop
    result = Solvers.Scipy.solve_single(matrix)

# Better: Batch them together
all_matrices = np.array([generate_matrix() for _ in range(1000)])
results = Solvers.BatchedScipyOMP.batch_solve(all_matrices)  # 6x faster!
```

## Development

### Installation

```bash
# Install with development dependencies (includes black, ruff, pytest)
pip install -e ".[dev]"

# Or with Pixi-managed environment and toolchain
pixi install
```

### Code Formatting and Linting

The project uses `black` for code formatting and `ruff` for linting. A Makefile is provided for convenience:

```bash
# Format code with black
make format

# Or with Pixi
pixi run format

# Lint code with ruff
make lint

# Or with Pixi
pixi run lint

# Auto-fix linting issues
make lint-fix

# Or with Pixi
pixi run lint-fix

# Run all checks
make check

# Or with Pixi
pixi run check

# Format, lint-fix, check, and test in one command
make all

# Or with Pixi
pixi run all
```

Or use the tools directly:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/
```

### Testing

```bash
# Run tests with pytest
pytest tests/

# Or with Pixi
pixi run test

# Or use make
make test
```

### Benchmarking

Benchmarks are split into single-problem and batched-problem modes.
Both modes generate runtime-vs-matrix-size graphs (`.png`) and raw data (`.csv`)
under `benchmark_results/`.

```bash
# Single-problem benchmarks (includes CUDA solver if available)
pixi run benchmark-single

# Batched benchmarks (non-CUDA solvers only)
pixi run benchmark-batch

# Run both
pixi run benchmark-all

# CUDA-only timing breakdown (transfer vs solve)
pixi run benchmark-cuda-breakdown

# Batch-size scaling at fixed problem size (default N=256, batch=1..1024 powers of two)
pixi run benchmark-batch-scaling
```

Direct script usage:

```bash
python tests/benchmarks.py --mode single
python tests/benchmarks.py --mode batch --batch-size 32
python tests/benchmarks.py --mode all
python tests/benchmark_lap1015_cuda_breakdown.py
python tests/benchmark_batched_batch_size_scaling.py
```

Plot outputs are written to `benchmark_results/`:

- Runtime vs matrix size (single solve): `single_runtime_vs_size.png` (linear + log-log)
- Runtime vs matrix size (batched solve): `batch_runtime_vs_size_bs{BATCH_SIZE}.png`
- Runtime vs batch size at fixed problem size: `batch_runtime_vs_batch_size_n{PROBLEM_SIZE}.png` (linear + log-log)
- CUDA transfer/solve breakdown vs matrix size: `lap1015_cuda_timing_breakdown.png` (linear + log-log)

## License

MIT
