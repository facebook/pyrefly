from collections.abc import Sequence
from typing_extensions import Self

from numpy import ndarray, uint8
from numpy.random import RandomState

from ..._typing import ArrayLike, Int, MatrixLike
from ...base import BaseEstimator, TransformerMixin

# Author: Nicolas Hug

class _BinMapper(TransformerMixin, BaseEstimator):
    missing_values_bin_idx_: uint8 = ...
    is_categorical_: ndarray = ...
    n_bins_non_missing_: ndarray = ...
    bin_thresholds_: list[ndarray] = ...

    def __init__(
        self,
        n_bins: Int = 256,
        subsample: None | int = ...,
        is_categorical: None | ArrayLike = None,
        known_categories: Sequence[None | ArrayLike] | None | ArrayLike = None,
        random_state: RandomState | None | Int = None,
        n_threads: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y=None) -> Self: ...
    def transform(self, X: MatrixLike) -> ndarray: ...
    def make_known_categories_bitsets(self) -> tuple[ndarray, ndarray]: ...
