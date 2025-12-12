import numpy as np
from scipy.sparse import spmatrix

def init_bounds_dense(
    X: np.ndarray,
    centers: np.ndarray,
    center_half_distances: np.ndarray,
    labels: np.ndarray,
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
    n_threads: int,
) -> None:
    """Initialize upper and lower bounds for each sample for dense input data.
    Given X, centers and the pairwise distances divided by 2.0 between the
    centers this calculates the upper bounds and lower bounds for each sample.
    The upper bound for each sample is set to the distance between the sample
    and the closest center.
    The lower bound for each sample is a one-dimensional array of n_clusters.
    For each sample i assume that the previously assigned cluster is c1 and the
    previous closest distance is dist, for a new cluster c2, the
    lower_bound[i][c2] is set to distance between the sample and this new
    cluster, if and only if dist > center_half_distances[c1][c2]. This prevents
    computation of unnecessary distances for each sample to the clusters that
    it is unlikely to be assigned to.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The input data.
    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers.
    center_half_distances : ndarray of shape (n_clusters, n_clusters),             dtype=floating
        The half of the distance between any 2 clusters centers.
    labels : ndarray of shape(n_samples), dtype=int
        The label for each sample. This array is modified in place.
    upper_bounds : ndarray of shape(n_samples,), dtype=floating
        The upper bound on the distance between each sample and its closest
        cluster center. This array is modified in place.
    lower_bounds : ndarray, of shape(n_samples, n_clusters), dtype=floating
        The lower bound on the distance between each sample and each cluster
        center. This array is modified in place.
    n_threads : int
        The number of threads to be used by openmp.
    """

def init_bounds_sparse(
    X: spmatrix,
    centers: np.ndarray,
    center_half_distances: np.ndarray,
    labels: np.ndarray,
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
    n_threads: int,
) -> None:
    """Initialize upper and lower bounds for each sample for sparse input data.
    Given X, centers and the pairwise distances divided by 2.0 between the
    centers this calculates the upper bounds and lower bounds for each sample.
    The upper bound for each sample is set to the distance between the sample
    and the closest center.
    The lower bound for each sample is a one-dimensional array of n_clusters.
    For each sample i assume that the previously assigned cluster is c1 and the
    previous closest distance is dist, for a new cluster c2, the
    lower_bound[i][c2] is set to distance between the sample and this new
    cluster, if and only if dist > center_half_distances[c1][c2]. This prevents
    computation of unnecessary distances for each sample to the clusters that
    it is unlikely to be assigned to.
    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The input data. Must be in CSR format.
    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers.
    center_half_distances : ndarray of shape (n_clusters, n_clusters),             dtype=floating
        The half of the distance between any 2 clusters centers.
    labels : ndarray of shape(n_samples), dtype=int
        The label for each sample. This array is modified in place.
    upper_bounds : ndarray of shape(n_samples,), dtype=floating
        The upper bound on the distance between each sample and its closest
        cluster center. This array is modified in place.
    lower_bounds : ndarray of shape(n_samples, n_clusters), dtype=floating
        The lower bound on the distance between each sample and each cluster
        center. This array is modified in place.
    n_threads : int
        The number of threads to be used by openmp.
    """

def elkan_iter_chunked_dense(
    X: np.ndarray,
    sample_weight: np.ndarray,
    centers_old: np.ndarray,
    centers_new: np.ndarray,
    weight_in_clusters: np.ndarray,
    center_half_distances: np.ndarray,
    distance_next_center: np.ndarray,
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
    labels: np.ndarray,
    center_shift: np.ndarray,
    n_threads: int,
    update_centers: bool = True,
) -> None:
    """Single iteration of K-means Elkan algorithm with dense input.
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
        computed during this iteration.
    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.
    center_half_distances : ndarray of shape (n_clusters, n_clusters),             dtype=floating
        Half pairwise distances between centers.
    distance_next_center : ndarray of shape (n_clusters,), dtype=floating
        Distance between each center its closest center.
    upper_bounds : ndarray of shape (n_samples,), dtype=floating
        Upper bound for the distance between each sample and its center,
        updated inplace.
    lower_bounds : ndarray of shape (n_samples, n_clusters), dtype=floating
        Lower bound for the distance between each sample and each center,
        updated inplace.
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

def elkan_iter_chunked_sparse(
    X: spmatrix,
    sample_weight: np.ndarray,
    centers_old: np.ndarray,
    centers_new: np.ndarray,
    weight_in_clusters: np.ndarray,
    center_half_distances: np.ndarray,
    distance_next_center: np.ndarray,
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
    labels: np.ndarray,
    center_shift: np.ndarray,
    n_threads: int,
    update_centers: bool = True,
) -> None:
    """Single iteration of K-means Elkan algorithm with sparse input.
    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.
    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        The observations to cluster. Must be in CSR format.
    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.
    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.
    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.
    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.
    center_half_distances : ndarray of shape (n_clusters, n_clusters),             dtype=floating
        Half pairwise distances between centers.
    distance_next_center : ndarray of shape (n_clusters,), dtype=floating
        Distance between each center its closest center.
    upper_bounds : ndarray of shape (n_samples,), dtype=floating
        Upper bound for the distance between each sample and its center,
        updated inplace.
    lower_bounds : ndarray of shape (n_samples, n_clusters), dtype=floating
        Lower bound for the distance between each sample and each center,
        updated inplace.
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
