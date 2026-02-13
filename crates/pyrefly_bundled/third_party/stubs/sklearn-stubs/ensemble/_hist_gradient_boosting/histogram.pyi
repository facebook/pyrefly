import numpy as np

class HistogramBuilder:
    """A Histogram builder... used to build histograms.
    A histogram is an array with n_bins entries of type HISTOGRAM_DTYPE. Each
    feature has its own histogram. A histogram contains the sum of gradients
    and hessians of all the samples belonging to each bin.
    There are different ways to build a histogram:
    - by subtraction: hist(child) = hist(parent) - hist(sibling)
    - from scratch. In this case we have routines that update the hessians
      or not (not useful when hessians are constant for some losses e.g.
      least squares). Also, there's a special case for the root which
      contains all the samples, leading to some possible optimizations.
      Overall all the implementations look the same, and are optimized for
      cache hit.
    Parameters
    ----------
    X_binned : ndarray of int, shape (n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    n_bins : int
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
    gradients : ndarray, shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians : ndarray, shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians_are_constant : bool
        Whether hessians are constant.
    """

    def __init__(
        self,
        X_binned: np.ndarray,
        n_bins: int,
        gradients: np.ndarray,
        hessians: np.ndarray,
        hessians_are_constant: bool,
        n_threads: int,
    ) -> None: ...
    def compute_histograms_brute(
        self,
        sample_indices: np.ndarray,
        allowed_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the histograms of the node by scanning through all the data.
        For a given feature, the complexity is O(n_samples)
        Parameters
        ----------
        sample_indices : array of int, shape (n_samples_at_node,)
            The indices of the samples at the node to split.
        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.
        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape (n_features, n_bins)
            The computed histograms of the current node.
        """

    def compute_histograms_subtraction(
        self,
        parent_histograms: np.ndarray,
        sibling_histograms: np.ndarray,
        allowed_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the histograms of the node using the subtraction trick.
        hist(parent) = hist(left_child) + hist(right_child)
        For a given feature, the complexity is O(n_bins). This is much more
        efficient than compute_histograms_brute, but it's only possible for one
        of the siblings.
        Parameters
        ----------
        parent_histograms : ndarray of HISTOGRAM_DTYPE,                 shape (n_features, n_bins)
            The histograms of the parent.
        sibling_histograms : ndarray of HISTOGRAM_DTYPE,                 shape (n_features, n_bins)
            The histograms of the sibling.
        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.
        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape(n_features, n_bins)
            The computed histograms of the current node.
        """
