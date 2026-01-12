from collections.abc import Sequence
from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from pandas.core.series import Series
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Int, MatrixLike
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

# Authors: Andreas Mueller <amueller@ais.uni-bonn.de>
#          Joris Van den Bossche <jorisvandenbossche@gmail.com>
# License: BSD 3 clause

__all__ = ["OneHotEncoder", "OrdinalEncoder"]

class _BaseEncoder(TransformerMixin, BaseEstimator): ...

class OneHotEncoder(_BaseEncoder):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    drop_idx_: ndarray = ...
    categories_: list[ArrayLike] = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        categories: Sequence[ArrayLike] | Literal["auto"] = "auto",
        drop: None | ArrayLike | Literal["first", "if_binary"] = None,
        sparse: str | bool = "deprecated",
        sparse_output: bool = True,
        dtype=...,
        handle_unknown: Literal["error", "ignore", "infrequent_if_exist"] = "error",
        min_frequency: float | None = None,
        max_categories: None | Int = None,
    ) -> None: ...
    @property
    def infrequent_categories_(self) -> list[ndarray]: ...
    def fit(
        self,
        X: MatrixLike,
        y: Series | None | ndarray | list[int] = None,
    ) -> Self: ...
    def transform(self, X: MatrixLike) -> ndarray | spmatrix: ...
    def inverse_transform(self, X: MatrixLike) -> ndarray: ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...

class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    categories_: list[ArrayLike] = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        categories: Sequence[ArrayLike] | Literal["auto"] = "auto",
        dtype=...,
        handle_unknown: Literal["error", "use_encoded_value"] = "error",
        unknown_value: float | None = None,
        encoded_missing_value: float = ...,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Series | None = None) -> Self: ...
    def transform(self, X: MatrixLike) -> ndarray: ...
    def inverse_transform(self, X: MatrixLike) -> ndarray: ...
