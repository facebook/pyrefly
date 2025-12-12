from typing import Any, ClassVar
from typing_extensions import Self

from numpy import ndarray

from .._typing import Float, MatrixLike
from ..base import BaseEstimator
from ._base import SelectorMixin

# Author: Lars Buitinck
# License: 3-clause BSD

class VarianceThreshold(SelectorMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    variances_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(self, threshold: Float = 0.0) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
