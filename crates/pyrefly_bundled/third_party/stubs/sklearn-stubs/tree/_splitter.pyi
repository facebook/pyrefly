import numpy as np

from ._criterion import Criterion

class SplitRecord:
    # Data to track sample split
    feature: int  # Which feature to split on.
    pos: int  # Split samples array at the given position,
    # i.e. count of samples below threshold for feature.
    # pos is >= end if the node is a leaf.
    threshold: float  # Threshold to split at.
    improvement: float  # Impurity improvement given parent node.
    impurity_left: float  # Impurity of the left split.
    impurity_right: float  # Impurity of the right split.

class Splitter:
    """The splitter searches in the input space for a feature and a threshold
    to split the samples samples[start:end]."""

    criterion: Criterion  # Impurity criterion
    max_features: int  # Number of features to test
    min_samples_leaf: int  # Min samples in a leaf
    min_weight_leaf: float  # Minimum weight in a leaf

    random_state: object  # Random state
    rand_r_state: int  # sklearn_rand_r random number state

    samples: np.ndarray
    n_samples: int
    weighted_n_samples: float
    features: np.ndarray
    constant_features: np.ndarray
    n_features: int
    feature_values: np.ndarray

    start: int
    end: int

    y: np.ndarray
    sample_weight: np.ndarray

    def init(self, X: object, y: np.ndarray, sample_weight: np.ndarray) -> int: ...
    def node_reset(self, start: int, end: int, weighted_n_node_samples: list[float]) -> int: ...
    def node_split(self, impurity: float, split: SplitRecord, n_constant_features: list[int]) -> int: ...
    def node_value(self, dest: list[float]) -> None: ...
    def node_impurity(self) -> float: ...
