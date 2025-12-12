from typing import Any, Callable, ClassVar
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, MatrixLike
from ..base import BaseEstimator, RegressorMixin

# Authors: Andreas Mueller <andreas.mueller@columbia.edu>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
# License: BSD 3 clause

__all__ = ["TransformedTargetRegressor"]

class TransformedTargetRegressor(RegressorMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    transformer_: Any = ...
    regressor_: Any = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        regressor: Any = None,
        *,
        transformer: Any = None,
        func: None | Callable = None,
        inverse_func: None | Callable = None,
        check_inverse: bool = True,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        **fit_params,
    ) -> Self: ...
    def predict(self, X: MatrixLike | ArrayLike, **predict_params) -> ndarray: ...
    @property
    def n_features_in_(self) -> int: ...
