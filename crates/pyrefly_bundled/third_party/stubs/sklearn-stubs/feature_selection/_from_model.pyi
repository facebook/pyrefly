from typing import Any, Callable, ClassVar
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, MatrixLike
from ..base import BaseEstimator, MetaEstimatorMixin
from ._base import SelectorMixin

# Authors: Gilles Louppe, Mathieu Blondel, Maheshakya Wijewardena
# License: BSD 3 clause

class SelectFromModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    max_features_: int = ...
    estimator_: BaseEstimator = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        estimator: Any,
        *,
        threshold: float | None | str = None,
        prefit: bool = False,
        norm_order: float = 1,
        max_features: None | Callable | int = None,
        importance_getter: str | Callable = "auto",
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike,
        y: None | ArrayLike = None,
        **fit_params,
    ) -> Self: ...
    @property
    def threshold_(self) -> float: ...
    def partial_fit(
        self,
        X: MatrixLike,
        y: None | ArrayLike = None,
        **fit_params,
    ) -> Self: ...
    @property
    def n_features_in_(self) -> int: ...
