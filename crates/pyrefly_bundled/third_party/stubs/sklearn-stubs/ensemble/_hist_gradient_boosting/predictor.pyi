from numpy import ndarray

from ..._typing import ArrayLike, Int, MatrixLike

# Author: Nicolas Hug

class TreePredictor:
    def __init__(
        self,
        nodes: ArrayLike,
        binned_left_cat_bitsets: MatrixLike,
        raw_left_cat_bitsets: MatrixLike,
    ) -> None: ...
    def get_n_leaf_nodes(self): ...
    def get_max_depth(self): ...
    def predict(
        self,
        X: MatrixLike,
        known_cat_bitsets: MatrixLike,
        f_idx_map: ArrayLike,
        n_threads: Int,
    ) -> ndarray: ...
    def predict_binned(self, X: MatrixLike, missing_values_bin_idx: int, n_threads: Int) -> ndarray: ...
    def compute_partial_dependence(self, grid: MatrixLike, target_features: ArrayLike, out: ArrayLike) -> None: ...
