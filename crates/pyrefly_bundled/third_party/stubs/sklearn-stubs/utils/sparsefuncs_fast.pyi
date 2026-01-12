import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

def csr_row_norms(X: np.ndarray) -> np.ndarray:
    """Squared L2 norm of each row in CSR matrix X."""

def csr_mean_variance_axis0(
    X: csr_matrix, weights: np.ndarray | None = None, return_sum_weights: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and variance along axis 0 on a CSR matrix
    Uses a np.float64 accumulator.
    Parameters
    ----------
    X : CSR sparse matrix, shape (n_samples, n_features)
        Input data.
    weights : ndarray of shape (n_samples,), dtype=floating, default=None
        If it is set to None samples will be equally weighted.
        .. versionadded:: 0.24
    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature.
        .. versionadded:: 0.24
    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means
    variances : float array with shape (n_features,)
        Feature-wise variances
    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if return_sum_weights is True.
    """

def csc_mean_variance_axis0(
    X: csc_matrix, weights: np.ndarray | None = None, return_sum_weights: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and variance along axis 0 on a CSC matrix
    Uses a np.float64 accumulator.
    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.
    weights : ndarray of shape (n_samples,), dtype=floating, default=None
        If it is set to None samples will be equally weighted.
        .. versionadded:: 0.24
    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature.
        .. versionadded:: 0.24
    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means
    variances : float array with shape (n_features,)
        Feature-wise variances
    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if return_sum_weights is True.
    """

def incr_mean_variance_axis0(
    X: csr_matrix | csc_matrix, last_mean: np.ndarray, last_var: np.ndarray, last_n: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and variance along axis 0 on a CSR or CSC matrix.
    last_mean, last_var are the statistics computed at the last step by this
    function. Both must be initialized to 0.0. last_n is the
    number of samples encountered until now and is initialized at 0.
    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
      Input data.
    last_mean : float array with shape (n_features,)
      Array of feature-wise means to update with the new data X.
    last_var : float array with shape (n_features,)
      Array of feature-wise var to update with the new data X.
    last_n : float array with shape (n_features,)
      Sum of the weights seen so far (if weights are all set to 1
      this will be the same as number of samples seen so far, before X).
    weights : float array with shape (n_samples,) or None. If it is set
      to None samples will be equall weighted.
    Returns
    -------
    updated_mean : float array with shape (n_features,)
      Feature-wise means
    updated_variance : float array with shape (n_features,)
      Feature-wise variances
    updated_n : int array with shape (n_features,)
      Updated number of samples seen
    """

def inplace_csr_row_normalize_l1(X: np.ndarray) -> None:
    """Inplace row normalize using the l1 norm"""

def inplace_csr_row_normalize_l2(X: np.ndarray) -> None:
    """Inplace row normalize using the l2 norm"""

def assign_rows_csr(
    X: csr_matrix,
    X_rows: np.ndarray,
    out_rows: np.ndarray,
    out: np.ndarray,
) -> None:
    """Densify selected rows of a CSR matrix into a preallocated array.
    Like out[out_rows] = X[X_rows].toarray() but without copying.
    No-copy supported for both dtype=np.float32 and dtype=np.float64.
    Parameters
    ----------
    X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)
    X_rows : array, dtype=np.intp, shape=n_rows
    out_rows : array, dtype=np.intp, shape=n_rows
    out : array, shape=(arbitrary, n_features)
    """
