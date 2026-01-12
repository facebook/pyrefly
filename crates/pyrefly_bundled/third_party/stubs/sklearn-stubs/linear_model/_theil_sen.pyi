from typing import ClassVar
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int
from ..base import RegressorMixin
from ._base import LinearModel

# Author: Florian Wilhelm <florian.wilhelm@gmail.com>
#
# License: BSD 3 clause

_EPSILON = ...

class TheilSenRegressor(RegressorMixin, LinearModel):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_subpopulation_: int = ...
    n_iter_: int = ...
    breakdown_: float = ...
    intercept_: float = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_subpopulation: float | Int = 1e4,
        n_subsamples: None | Int = None,
        max_iter: Int = 300,
        tol: Float = 1.0e-3,
        random_state: RandomState | None | Int = None,
        n_jobs: None | Int = None,
        verbose: bool = False,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
