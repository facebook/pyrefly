from typing import Any, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from pandas.core.series import Series
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Int, MatrixLike
from ..base import BaseEstimator, TransformerMixin

__all__ = [
    "PolynomialFeatures",
    "SplineTransformer",
]

class PolynomialFeatures(TransformerMixin, BaseEstimator):
    n_output_features_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        degree: int | tuple[int, int] = 2,
        *,
        interaction_only: bool = False,
        include_bias: bool = True,
        order: Literal["C", "F"] = "C",
    ) -> None: ...
    @property
    def powers_(self) -> ndarray: ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
    def fit(self, X: MatrixLike | ArrayLike, y: Any = None) -> Self: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray | spmatrix: ...

class SplineTransformer(TransformerMixin, BaseEstimator):
    n_features_out_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    bsplines_: list = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_knots: Int = 5,
        degree: Int = 3,
        *,
        knots: Literal["uniform", "quantile"] | MatrixLike = "uniform",
        extrapolation: Literal["error", "constant", "linear", "continue", "periodic"] = "constant",
        include_bias: bool = True,
        order: Literal["C", "F"] = "C",
        sparse_output: bool = False,
    ) -> None: ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
    def fit(
        self,
        X: MatrixLike,
        y: Series | None | ndarray = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def transform(self, X: MatrixLike) -> ndarray: ...
