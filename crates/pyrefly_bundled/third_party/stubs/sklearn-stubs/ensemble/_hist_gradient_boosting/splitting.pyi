import numpy as np

class SplitInfo:
    """Pure data class to store information about a potential split.
    Parameters
    ----------
    gain : float
        The gain of the split.
    feature_idx : int
        The index of the feature to be split.
    bin_idx : int
        The index of the bin on which the split is made. Should be ignored if
        `is_categorical` is True: `left_cat_bitset` will be used to determine
        the split.
    missing_go_to_left : bool
        Whether missing values should go to the left child. This is used
        whether the split is categorical or not.
    sum_gradient_left : float
        The sum of the gradients of all the samples in the left child.
    sum_hessian_left : float
        The sum of the hessians of all the samples in the left child.
    sum_gradient_right : float
        The sum of the gradients of all the samples in the right child.
    sum_hessian_right : float
        The sum of the hessians of all the samples in the right child.
    n_samples_left : int, default=0
        The number of samples in the left child.
    n_samples_right : int
        The number of samples in the right child.
    is_categorical : bool
        Whether the split is done on a categorical feature.
    left_cat_bitset : ndarray of shape=(8,), dtype=uint32 or None
        Bitset representing the categories that go to the left. This is used
        only when `is_categorical` is True.
        Note that missing values are part of that bitset if there are missing
        values in the training data. For missing values, we rely on that
        bitset for splitting, but at prediction time, we rely on
        missing_go_to_left.
    """

    def __init__(
        self,
        gain: float,
        feature_idx: int,
        bin_idx: int,
        missing_go_to_left: bool,
        sum_gradient_left: float,
        sum_hessian_left: float,
        sum_gradient_right: float,
        sum_hessian_right: float,
        n_samples_left: int,
        n_samples_right: int,
        value_left,
        value_right,
        is_categorical: bool,
        left_cat_bitset: np.ndarray | None = None,
    ): ...

class Splitter:
    """Splitter used to find the best possible split at each node.
    A split (see SplitInfo) is characterized by a feature and a bin.
    The Splitter is also responsible for partitioning the samples among the
    leaves of the tree (see split_indices() and the partition attribute).
    Parameters
    ----------
    X_binned : ndarray of int, shape (n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    n_bins_non_missing : ndarray, shape (n_features,)
        For each feature, gives the number of bins actually used for
        non-missing values.
    missing_values_bin_idx : uint8
        Index of the bin that is used for missing values. This is the index of
        the last bin and is always equal to max_bins (as passed to the GBDT
        classes), or equivalently to n_bins - 1.
    has_missing_values : ndarray, shape (n_features,)
        Whether missing values were observed in the training data, for each
        feature.
    is_categorical : ndarray of bool of shape (n_features,)
        Indicates categorical features.
    monotonic_cst : ndarray of int of shape (n_features,), dtype=int
        Indicates the monotonic constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease
        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
    l2_regularization : float
        The L2 regularization parameter.
    min_hessian_to_split : float, default=1e-3
        The minimum sum of hessians needed in each node. Splits that result in
        at least one child having a sum of hessians less than
        min_hessian_to_split are discarded.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf.
    min_gain_to_split : float, default=0.0
        The minimum gain needed to split a node. Splits with lower gain will
        be ignored.
    hessians_are_constant: bool, default is False
        Whether hessians are constant.
    n_threads : int, default=1
        Number of OpenMP threads to use.
    """

    X_binned: np.ndarray
    n_features: int
    n_bins_non_missing: np.ndarray
    missing_values_bin_idx: int
    has_missing_values: np.ndarray
    is_categorical: np.ndarray
    monotonic_cst: np.ndarray
    hessians_are_constant: int
    l2_regularization: float
    min_hessian_to_split: float
    min_samples_leaf: int
    min_gain_to_split: float

    partition: np.ndarray
    left_indices_buffer: np.ndarray
    right_indices_buffer: np.ndarray
    n_threads: int

    def __init__(
        self,
        X_binned: np.ndarray,
        n_bins_non_missing: np.ndarray,
        missing_values_bin_idx: int,
        has_missing_values: np.ndarray,
        is_categorical: np.ndarray,
        monotonic_cst: np.ndarray,
        l2_regularization: float,
        min_hessian_to_split: float = 1e-3,
        min_samples_leaf: int = 20,
        min_gain_to_split: float = 0.0,
        hessians_are_constant: bool = False,
        n_threads: int = 1,
    ): ...
    def split_indices(self, split_info: SplitInfo, sample_indices: np.ndarray):
        """Split samples into left and right arrays.
        The split is performed according to the best possible split
        (split_info).
        Ultimately, this is nothing but a partition of the sample_indices
        array with a given pivot, exactly like a quicksort subroutine.
        Parameters
        ----------
        split_info : SplitInfo
            The SplitInfo of the node to split.
        sample_indices : ndarray of unsigned int, shape (n_samples_at_node,)
            The indices of the samples at the node to split. This is a view
            on self.partition, and it is modified inplace by placing the
            indices of the left child at the beginning, and the indices of
            the right child at the end.
        Returns
        -------
        left_indices : ndarray of int, shape (n_left_samples,)
            The indices of the samples in the left child. This is a view on
            self.partition.
        right_indices : ndarray of int, shape (n_right_samples,)
            The indices of the samples in the right child. This is a view on
            self.partition.
        right_child_position : int
            The position of the right child in ``sample_indices``.
        """

    def find_node_split(
        self,
        n_samples: int,
        histograms,  # IN
        sum_gradients: float,
        sum_hessians: float,
        value: float,
        lower_bound: float = ...,
        upper_bound: float = ...,
        allowed_features: np.ndarray | None = None,
    ):
        """For each feature, find the best bin to split on at a given node.
        Return the best split info among all features.
        Parameters
        ----------
        n_samples : int
            The number of samples at the node.
        histograms : ndarray of HISTOGRAM_DTYPE of                 shape (n_features, max_bins)
            The histograms of the current node.
        sum_gradients : float
            The sum of the gradients for each sample at the node.
        sum_hessians : float
            The sum of the hessians for each sample at the node.
        value : float
            The bounded value of the current node. We directly pass the value
            instead of re-computing it from sum_gradients and sum_hessians,
            because we need to compute the loss and the gain based on the
            *bounded* value: computing the value from
            sum_gradients / sum_hessians would give the unbounded value, and
            the interaction with min_gain_to_split would not be correct
            anymore. Side note: we can't use the lower_bound / upper_bound
            parameters either because these refer to the bounds of the
            children, not the bounds of the current node.
        lower_bound : float
            Lower bound for the children values for respecting the monotonic
            constraints.
        upper_bound : float
            Upper bound for the children values for respecting the monotonic
            constraints.
        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.
        Returns
        -------
        best_split_info : SplitInfo
            The info about the best possible split among all features.
        """
