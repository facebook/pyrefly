import numpy as np

def lloyd_iter_chunked_dense(
    X: np.ndarray,
    sample_weight: np.ndarray,
    centers_old: np.ndarray,
    centers_new: np.ndarray,
    weight_in_clusters: np.ndarray,
    labels: np.ndarray,
    center_shift: np.ndarray,
    n_threads: int,
    update_centers: bool = True,
) -> None:
    """Single iteration of K-means lloyd algorithm with dense input.
    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.
    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.
    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.
    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.
    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.
    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.
    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.
    n_threads : int
        The number of threads to be used by openmp.
    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """

def lloyd_iter_chunked_sparse(
    X: np.ndarray,
    sample_weight: np.ndarray,
    centers_old: np.ndarray,
    centers_new: np.ndarray,
    weight_in_clusters: np.ndarray,
    labels: np.ndarray,
    center_shift: np.ndarray,
    n_threads: int,
    update_centers: bool = True,
) -> None:
    """Single iteration of K-means lloyd algorithm with sparse input.
    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.
    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The observations to cluster. Must be in CSR format.
    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.
    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.
    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.
    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.
    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.
    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.
    n_threads : int
        The number of threads to be used by openmp.
    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
