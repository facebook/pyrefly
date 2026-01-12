from abc import ABCMeta, abstractmethod
from typing import ClassVar
from typing_extensions import Self

from numpy import ndarray
from numpy.random.mtrand import RandomState

from .._typing import ArrayLike, Int, MatrixLike
from ..base import BaseEstimator as BaseEstimator, ClassifierMixin, MultiOutputMixin, RegressorMixin
from ..utils._seq_dataset import ArrayDataset64, CSRDataset64
from ._stochastic_gradient import SGDClassifier

SPARSE_INTERCEPT_DECAY: float = ...

def make_dataset(
    X: MatrixLike,
    y: ArrayLike,
    sample_weight: ArrayLike,
    random_state: None | Int | RandomState = None,
) -> tuple[ArrayDataset64 | CSRDataset64, float]: ...

class LinearModel(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y): ...
    def predict(self, X: MatrixLike) -> ndarray: ...

class LinearClassifierMixin(ClassifierMixin):
    def decision_function(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...

class SparseCoefMixin:
    def densify(self) -> Self: ...
    def sparsify(self) -> SGDClassifier | Self: ...

class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    intercept_: float | ndarray = ...
    singular_: ndarray = ...
    rank_: int = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: None | Int = None,
        positive: bool = False,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
