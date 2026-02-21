"""Float32 policy checks for solver inputs and test matrix generators."""

import numpy as np
import pytest
from get_cost_matrices import (
    get_full_rect_matrix,
    get_full_square_matrix,
    get_masked_rect_matrix,
    get_masked_square_matrix,
    get_padded_square_to_rect_matrix,
)
from scipy.optimize import linear_sum_assignment

from py_lap_solver.solvers import Solvers


def _total_cost(cost_matrix, row_to_col):
    """Compute assignment cost, skipping unassigned rows."""
    total_cost = np.float32(0.0)
    for row, col in enumerate(row_to_col):
        if col >= 0:
            total_cost += cost_matrix[row, col]
    return float(total_cost)


@pytest.mark.parametrize(
    "generator,kwargs",
    [
        (get_full_square_matrix, {"size": 16}),
        (get_full_rect_matrix, {"n_rows": 10, "n_cols": 20}),
        (get_masked_square_matrix, {"size": 20}),
        (get_masked_rect_matrix, {"n_rows": 8, "n_cols": 16}),
        (get_padded_square_to_rect_matrix, {"full_size": 20, "num_valid_rows": 8}),
    ],
)
def test_matrix_generators_emit_float32(generator, kwargs):
    """All matrix generation utilities should emit float32 tensors."""
    result = generator(**kwargs)
    matrix = result[0] if isinstance(result, tuple) else result
    assert matrix.dtype == np.float32

    result_batched = generator(batch_size=3, **kwargs)
    matrix_batched = result_batched[0] if isinstance(result_batched, tuple) else result_batched
    assert matrix_batched.dtype == np.float32


@pytest.mark.parametrize(
    "solver_name,solver_instance", list(Solvers.get_available_solvers().items())
)
def test_solvers_match_float32_reference_single(solver_name, solver_instance):
    """Single solve should match scipy's float32 reference."""
    matrix = get_full_square_matrix(32)

    row_ind, col_ind = linear_sum_assignment(matrix)
    ref = np.full(matrix.shape[0], -1, dtype=np.int32)
    ref[row_ind] = col_ind

    result = solver_instance.solve_single(matrix)

    ref_cost = _total_cost(matrix, ref)
    result_cost = _total_cost(matrix, result)
    assert np.isclose(
        result_cost, ref_cost, atol=1e-5
    ), f"{solver_name}: single float32 cost mismatch"


@pytest.mark.parametrize(
    "solver_name,solver_instance", list(Solvers.get_available_solvers().items())
)
def test_solvers_match_float32_reference_batch(solver_name, solver_instance):
    """Batch solve should match scipy's float32 reference."""
    batch = get_full_square_matrix(24, batch_size=4)

    results = solver_instance.batch_solve(batch)

    for idx in range(batch.shape[0]):
        row_ind, col_ind = linear_sum_assignment(batch[idx])
        ref = np.full(batch.shape[1], -1, dtype=np.int32)
        ref[row_ind] = col_ind

        ref_cost = _total_cost(batch[idx], ref)
        result_cost = _total_cost(batch[idx], results[idx])
        assert np.isclose(
            result_cost, ref_cost, atol=1e-5
        ), f"{solver_name}: batch {idx} float32 mismatch"
