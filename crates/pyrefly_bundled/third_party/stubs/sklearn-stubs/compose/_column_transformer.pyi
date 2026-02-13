from collections.abc import Sequence
from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import dtype, ndarray
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, TransformerMixin
from ..utils._bunch import Bunch
from ..utils.metaestimators import _BaseComposition

__all__ = ["ColumnTransformer", "make_column_transformer", "make_column_selector"]

_ERR_MSG_1DCOLUMN: str = ...

class ColumnTransformer(TransformerMixin, _BaseComposition):
    n_features_in_: int = ...
    output_indices_: dict = ...
    sparse_output_: bool = ...
    transformers_: list = ...

    _required_parameters: ClassVar[list] = ...

    def __init__(
        self,
        transformers: Sequence[tuple],
        *,
        remainder: Literal["drop", "passthrough"] | BaseEstimator = "drop",
        sparse_threshold: Float = 0.3,
        n_jobs: None | Int = None,
        transformer_weights: None | dict = None,
        verbose: bool = False,
        verbose_feature_names_out: bool = True,
    ) -> None: ...
    def set_output(self, *, transform: None | Literal["default", "pandas"] = None) -> BaseEstimator: ...
    def get_params(self, deep: bool = True) -> dict: ...
    def set_params(self, **kwargs) -> Self: ...
    @property
    def named_transformers_(self) -> Bunch: ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
    def fit(self, X: MatrixLike, y: None | ArrayLike = None) -> Self: ...
    def fit_transform(self, X: MatrixLike, y: None | ArrayLike = None) -> ndarray | spmatrix: ...
    def transform(self, X: MatrixLike) -> ndarray | spmatrix: ...

def make_column_transformer(
    *transformers,
    remainder: Literal["drop", "passthrough"] | BaseEstimator = "drop",
    sparse_threshold: Float = 0.3,
    n_jobs: None | Int = None,
    verbose: bool = False,
    verbose_feature_names_out: bool = True,
) -> ColumnTransformer: ...

class make_column_selector:
    def __init__(
        self,
        pattern: None | str = None,
        *,
        dtype_include: Sequence[dtype] | None | dtype = None,
        dtype_exclude: Sequence[dtype] | None | dtype = None,
    ) -> None: ...
    def __call__(self, df: MatrixLike) -> list[str]: ...
