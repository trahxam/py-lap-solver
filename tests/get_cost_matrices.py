"""Utilities for generating test cost matrices."""

import numpy as np

FLOAT_DTYPE = np.float32


def get_full_square_matrix(size, batch_size=None, seed=42):
    """Get a full square cost matrix with all entries valid.

    Parameters
    ----------
    size : int
        Dimension of the square matrix (NxN).
    batch_size : int, optional
        If provided, return a batch of matrices with shape (batch_size, size, size).
        If None, return a single matrix with shape (size, size).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    np.ndarray
        Cost matrix of shape (size, size) or (batch_size, size, size).
    """
    rng = np.random.RandomState(seed)

    if batch_size is None:
        return rng.rand(size, size).astype(FLOAT_DTYPE, copy=False)
    else:
        return rng.rand(batch_size, size, size).astype(FLOAT_DTYPE, copy=False)


def get_masked_square_matrix(size, batch_size=None, seed=42):
    """Get a masked square cost matrix where only the first num_valid entries are valid.

    The matrix is square but padded, where num_valid < size. This simulates
    a scenario where the matrix has been pre-padded.

    Parameters
    ----------
    size : int
        Dimension of the padded square matrix (NxN).
    batch_size : int, optional
        If provided, return a batch of matrices with shape (batch_size, size, size).
        If None, return a single matrix with shape (size, size).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    cost_matrix : np.ndarray
        Padded cost matrix of shape (size, size) or (batch_size, size, size).
    num_valid : int or np.ndarray
        Number of valid rows/cols. If batch_size is None, returns int.
        Otherwise returns array of shape (batch_size,).
    """
    rng = np.random.RandomState(seed)

    # Randomly choose num_valid to be 60-80% of size
    if batch_size is None:
        num_valid = int(size * (0.6 + 0.2 * rng.rand()))
        cost_matrix = rng.rand(size, size).astype(FLOAT_DTYPE, copy=False)
        # Make the padded region have large costs
        cost_matrix[num_valid:, :] = 1000.0
        cost_matrix[:, num_valid:] = 1000.0
        return cost_matrix, num_valid
    else:
        num_valid = np.array([int(size * (0.6 + 0.2 * rng.rand())) for _ in range(batch_size)])
        cost_matrices = rng.rand(batch_size, size, size).astype(FLOAT_DTYPE, copy=False)
        # Pad each matrix individually
        for i in range(batch_size):
            nv = num_valid[i]
            cost_matrices[i, nv:, :] = 1000.0
            cost_matrices[i, :, nv:] = 1000.0
        return cost_matrices, num_valid


def get_full_rect_matrix(n_rows, n_cols, batch_size=None, seed=42):
    """Get a full rectangular cost matrix with all entries valid.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    batch_size : int, optional
        If provided, return a batch of matrices with shape (batch_size, n_rows, n_cols).
        If None, return a single matrix with shape (n_rows, n_cols).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    np.ndarray
        Cost matrix of shape (n_rows, n_cols) or (batch_size, n_rows, n_cols).
    """
    rng = np.random.RandomState(seed)

    if batch_size is None:
        return rng.rand(n_rows, n_cols).astype(FLOAT_DTYPE, copy=False)
    else:
        return rng.rand(batch_size, n_rows, n_cols).astype(FLOAT_DTYPE, copy=False)


def get_masked_rect_matrix(n_rows, n_cols, batch_size=None, seed=42):
    """Get a masked rectangular matrix padded to square.

    The original rectangular matrix (n_rows x n_cols) is padded to a square
    matrix of size max(n_rows, n_cols) with large cost values in the padded region.

    Parameters
    ----------
    n_rows : int
        Number of rows in the original (unpadded) matrix.
    n_cols : int
        Number of columns in the original (unpadded) matrix.
    batch_size : int, optional
        If provided, return a batch of matrices.
        If None, return a single matrix.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    cost_matrix : np.ndarray
        Padded square cost matrix of shape (max_dim, max_dim) or
        (batch_size, max_dim, max_dim).
    num_valid_rows : int
        Number of valid rows (equal to n_rows).
    num_valid_cols : int
        Number of valid columns (equal to n_cols).
    """
    rng = np.random.RandomState(seed)
    max_dim = max(n_rows, n_cols)

    if batch_size is None:
        cost_matrix = np.full((max_dim, max_dim), 1000.0, dtype=FLOAT_DTYPE)
        cost_matrix[:n_rows, :n_cols] = rng.rand(n_rows, n_cols).astype(FLOAT_DTYPE, copy=False)
        return cost_matrix, n_rows, n_cols
    else:
        cost_matrices = np.full((batch_size, max_dim, max_dim), 1000.0, dtype=FLOAT_DTYPE)
        for i in range(batch_size):
            cost_matrices[i, :n_rows, :n_cols] = rng.rand(n_rows, n_cols).astype(
                FLOAT_DTYPE, copy=False
            )
        return cost_matrices, n_rows, n_cols


def get_padded_square_to_rect_matrix(full_size, num_valid_rows, batch_size=None, seed=42):
    """Get a padded square matrix where only first num_valid_rows rows are valid.

    This simulates the scenario where you have N ground truth objects (rows) but the
    matrix is padded to (M, M) where M > N. The solver uses num_valid to limit to
    first N rows, solving the (N, M) assignment problem.

    Parameters
    ----------
    full_size : int
        Full size of the square matrix (M x M).
    num_valid_rows : int
        Number of valid rows (N). Must be < full_size.
    batch_size : int, optional
        If provided, return a batch of matrices.
        If None, return a single matrix.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    cost_matrix : np.ndarray
        Padded square cost matrix of shape (full_size, full_size) or
        (batch_size, full_size, full_size).
    num_valid_rows : int
        Number of valid rows to use.
    """
    rng = np.random.RandomState(seed)

    if batch_size is None:
        cost_matrix = np.full((full_size, full_size), 1000.0, dtype=FLOAT_DTYPE)
        cost_matrix[:num_valid_rows, :] = rng.rand(num_valid_rows, full_size).astype(
            FLOAT_DTYPE, copy=False
        )
        return cost_matrix, num_valid_rows
    else:
        cost_matrices = np.full((batch_size, full_size, full_size), 1000.0, dtype=FLOAT_DTYPE)
        for i in range(batch_size):
            cost_matrices[i, :num_valid_rows, :] = rng.rand(num_valid_rows, full_size).astype(
                FLOAT_DTYPE, copy=False
            )
        return cost_matrices, num_valid_rows
