from abc import ABCMeta
from typing import Any, ClassVar
from typing_extensions import Self

from numpy import ndarray
from numpy.random.mtrand import RandomState

from .._typing import Float, Int, MatrixLike
from ..base import BaseEstimator, DensityMixin

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

class BaseMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: int,
        tol: float,
        reg_covar: float,
        max_iter: int,
        n_init: int,
        init_params: str,
        random_state: None | RandomState | int,
        warm_start: bool,
        verbose: int,
        verbose_interval: int,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
    def fit_predict(self, X: MatrixLike, y: Any = None) -> ndarray: ...
    def score_samples(self, X: MatrixLike) -> ndarray: ...
    def score(self, X: MatrixLike, y: Any = None) -> Float: ...
    def predict(self, X: MatrixLike) -> ndarray: ...
    def predict_proba(self, X: MatrixLike) -> ndarray: ...
    def sample(self, n_samples: Int = 1) -> tuple[ndarray, ndarray]: ...
