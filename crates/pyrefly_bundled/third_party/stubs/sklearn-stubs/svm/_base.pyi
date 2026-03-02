from abc import ABCMeta, abstractmethod
from typing import Callable, ClassVar
from typing_extensions import Self

from numpy import ndarray
from numpy.random.mtrand import RandomState

from .._typing import ArrayLike, Float, MatrixLike
from ..base import BaseEstimator, ClassifierMixin

LIBSVM_IMPL: list = ...

class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    _sparse_kernels: ClassVar[list] = ...

    @abstractmethod
    def __init__(
        self,
        kernel: str | Callable,
        degree: int,
        gamma: str | Float,
        coef0: float,
        tol: float,
        C: Float,
        nu: float,
        epsilon: float,
        shrinking: bool,
        probability: bool,
        cache_size: int,
        class_weight: dict[int, int] | None | str,
        verbose: bool,
        max_iter: int,
        random_state: None | RandomState | int,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    @property
    def coef_(self) -> ndarray: ...
    def n_support_(self) -> ndarray: ...

class BaseSVC(ClassifierMixin, BaseLibSVM, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        kernel: str | Callable,
        degree: int,
        gamma: str | Float,
        coef0: float,
        tol: float,
        C: Float,
        nu: float,
        shrinking: bool,
        probability: bool,
        cache_size: int,
        class_weight: dict[int, int] | None | str,
        verbose: bool,
        max_iter: int,
        decision_function_shape: str,
        random_state: None | RandomState | int,
        break_ties: bool,
    ) -> None: ...
    def decision_function(self, X: MatrixLike) -> ndarray: ...
    def predict(self, X: MatrixLike) -> ndarray: ...
    def predict_proba(self, X: MatrixLike) -> ndarray: ...
    def predict_log_proba(self, X: MatrixLike) -> ndarray: ...
    @property
    def probA_(self) -> ndarray: ...
    @property
    def probB_(self) -> ndarray: ...
