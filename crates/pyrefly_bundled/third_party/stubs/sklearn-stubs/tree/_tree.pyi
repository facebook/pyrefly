from typing import Any

import numpy as np

from ._splitter import Splitter

TREE_LEAF: int

class Node:
    left_child: int
    right_child: int
    feature: int
    threshold: float
    impurity: float
    n_node_samples: int
    weighted_n_node_samples: float

class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __init__(
        self,
        splitter: Splitter,
        min_samples_split: int,
        min_samples_leaf: int,
        min_weight_leaf: float,
        max_depth: int,
        min_impurity_decrease: float,
    ) -> None: ...
    def build(
        self,
        tree: Tree,
        X: object,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None: ...

class BestFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in best-first fashion.
    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.
    """

    max_leaf_nodes: int

    def __init__(
        self,
        splitter: Splitter,
        min_samples_split: int,
        min_samples_leaf: int,
        min_weight_leaf: float,
        max_depth: int,
        max_leaf_nodes: int,
        min_impurity_decrease: float,
    ) -> None: ...
    def build(
        self,
        tree: Tree,
        X: object,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None: ...

class Tree:
    n_features: int
    n_classes: int
    n_outputs: int
    max_n_classes: int

    max_dept: int
    node_count: int
    capacity: int
    nodes: np.ndarray
    value: np.ndarray
    value_stride: int

    def predict(self, X) -> np.ndarray: ...
    def apply(self, X) -> np.ndarray: ...
    def decision_path(self, X) -> Any: ...
    def compute_node_depths(self): ...
    def compute_feature_importances(self, normalize): ...

class TreeBuilder:
    splitter: Splitter
    min_samples_split: int
    min_samples_leaf: int
    min_weight_leaf: float
    max_depth: int
    min_impurity_decrease: float

    def build(
        self,
        tree: Tree,
        X,
        y,
        sample_weight,
    ): ...

def ccp_pruning_path(orig_tree: Tree) -> dict: ...
