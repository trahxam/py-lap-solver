from multiprocessing.pool import ThreadPool as Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..base import LapSolver


def solve_single(cost_matrix, unassigned_value, maximize, num_valid):
    """Helper function to solve a single LAP instance.

    This is defined outside the class to facilitate multiprocessing.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (N, M).
    unassigned_value : int
        Value to use for unassigned rows/columns in the output arrays.
    maximize : bool
        If True, solve the maximization problem instead of minimization.
    num_valid : int or None
        Number of valid rows if matrix is padded. If None, uses the full matrix row size.
    Returns
    -------
    row_to_col : np.ndarray
        Array of shape (N,) where row_to_col[i] gives the column assigned to row i.
        Unassigned rows have value `unassigned_value`.

    """
    # Keep Python/scipy path consistent with the C++ wrappers: float32 inputs only.
    cost_matrix = np.asarray(cost_matrix, dtype=np.float32)
    n_rows = cost_matrix.shape[0]

    # Slice to valid region if needed (this creates a view, not a copy)
    cost_to_solve = cost_matrix[:num_valid, :] if num_valid is not None else cost_matrix

    # Scipy minimizes by default, so negate for maximization
    if maximize:
        cost_to_solve = -cost_to_solve

    # Scipy returns (row_ind, col_ind) pairs
    row_ind, col_ind = linear_sum_assignment(cost_to_solve)

    # Convert to full-size array matching input row dimension
    row_to_col = np.full(n_rows, unassigned_value, dtype=np.int32)
    row_to_col[row_ind] = col_ind

    return row_to_col


class ScipySolver(LapSolver):
    """Linear Assignment Problem solver using scipy.optimize.linear_sum_assignment.

    This solver uses the Hungarian algorithm implementation from scipy.
    It's reliable and well-tested, suitable for small to medium-sized problems.

    Parameters
    ----------
    maximize : bool, optional
        If True, solve the maximization problem instead of minimization.
        Default is False (minimization).
    unassigned_value : int, optional
        Value to use for unassigned rows/columns in the output arrays.
        Default is -1.
    use_python_mp : bool, optional
        Whether to use Python multiprocessing for batch solving. Default is False.
    n_jobs : int, optional
        Number of worker processes to use when use_python_mp is True.
        Default is 8. Ignored if use_python_mp is False.
    """

    def __init__(
        self, maximize=False, unassigned_value=-1, use_python_mp=False, n_jobs=8, **kwargs
    ):
        super().__init__()
        self.maximize = maximize
        self.unassigned_value = unassigned_value
        self.use_python_mp = use_python_mp
        self.n_jobs = n_jobs

    def solve_single(self, cost_matrix, num_valid=None):
        """Solve a single linear assignment problem.

        Parameters
        ----------
        cost_matrix : np.ndarray
            Cost matrix of shape (N, M).
        num_valid : int, optional
            Number of valid rows/cols if matrix is padded.
            If None, uses the full matrix size.

        Returns
        -------
        row_to_col : np.ndarray
            Array of shape (N,) where row_to_col[i] gives the column assigned to row i.
            Unassigned rows have value `unassigned_value`.
        """
        return solve_single(
            cost_matrix,
            self.unassigned_value,
            self.maximize,
            num_valid,
        )

    def batch_solve(self, batch_cost_matrices, num_valid=None):
        """Solve multiple linear assignment problems.

        Parameters
        ----------
        batch_cost_matrices : np.ndarray
            Batch of cost matrices of shape (B, N, M).
        num_valid : np.ndarray or int, optional
            Number of valid rows/cols for each matrix.
            Can be a scalar (same for all) or array of shape (B,).

        Returns
        -------
        np.ndarray
            Array of shape (B, N) where element [b, i] gives the column assigned
            to row i in batch element b. Unassigned rows have value `unassigned_value`.
        """
        # Keep Python/scipy path consistent with the C++ wrappers: float32 inputs only.
        batch_cost_matrices = np.asarray(batch_cost_matrices, dtype=np.float32)
        batch_size, n_rows, _ = batch_cost_matrices.shape

        # Handle num_valid as scalar or array
        if num_valid is None:
            num_valid_array = [None] * batch_size
        elif isinstance(num_valid, (int, np.integer)):
            num_valid_array = [num_valid] * batch_size
        else:
            num_valid_array = num_valid

        if self.use_python_mp:
            # Use multiprocessing pool to solve in parallel
            args = [
                (batch_cost_matrices[i], self.unassigned_value, self.maximize, num_valid_array[i])
                for i in range(batch_size)
            ]
            chunk_size = (batch_size + self.n_jobs - 1) // self.n_jobs

            with Pool(processes=self.n_jobs) as pool:
                results = pool.starmap(solve_single, args, chunksize=chunk_size)

            return np.stack(results)
        else:
            # Sequential solving
            results = np.full((batch_size, n_rows), self.unassigned_value, dtype=np.int32)

            for i in range(batch_size):
                results[i] = self.solve_single(batch_cost_matrices[i], num_valid_array[i])

            return results
