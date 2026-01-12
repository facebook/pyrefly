from abc import ABC
from typing import Any, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin

EPSILON = ...

def norm(x: ArrayLike) -> float: ...
def trace_dot(X: ArrayLike, Y: ArrayLike) -> Float: ...
def non_negative_factorization(
    X: MatrixLike | ArrayLike,
    W: None | MatrixLike = None,
    H: None | ArrayLike = None,
    n_components: None | Int = None,
    *,
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"] | None = None,
    update_H: bool = True,
    solver: Literal["cd", "mu"] = "cd",
    beta_loss: float | Literal["frobenius", "kullback-leibler", "itakura-saito"] = "frobenius",
    tol: Float = 1e-4,
    max_iter: Int = 200,
    alpha_W: Float = 0.0,
    alpha_H: float | Literal["same"] = "same",
    l1_ratio: Float = 0.0,
    random_state: RandomState | None | Int = None,
    verbose: Int = 0,
    shuffle: bool = False,
) -> tuple[ndarray, ndarray, int]: ...

class _BaseNMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, ABC):
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: None | int = None,
        *,
        init=None,
        beta_loss: str = "frobenius",
        tol: float = 1e-4,
        max_iter: int = 200,
        random_state=None,
        alpha_W: float = 0.0,
        alpha_H: str = "same",
        l1_ratio: float = 0.0,
        verbose: int = 0,
    ) -> None: ...
    def fit(self, X: MatrixLike | ArrayLike, y: Any = None, **params) -> Self: ...
    def inverse_transform(self, W: MatrixLike) -> ndarray | spmatrix: ...

class NMF(_BaseNMF):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    reconstruction_err_: float = ...
    n_components_: int = ...
    components_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: None | Int = None,
        *,
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"] | None = None,
        solver: Literal["cd", "mu"] = "cd",
        beta_loss: float | Literal["frobenius", "kullback-leibler", "itakura-saito"] = "frobenius",
        tol: Float = 1e-4,
        max_iter: Int = 200,
        random_state: RandomState | None | Int = None,
        alpha_W: Float = 0.0,
        alpha_H: float | Literal["same"] = "same",
        l1_ratio: Float = 0.0,
        verbose: Int = 0,
        shuffle: bool = False,
    ) -> None: ...
    def fit_transform(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        W: None | MatrixLike = None,
        H: None | ArrayLike = None,
    ) -> ndarray: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray: ...

class MiniBatchNMF(_BaseNMF):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_steps_: int = ...
    n_iter_: int = ...
    reconstruction_err_: float = ...
    n_components_: int = ...
    components_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: None | Int = None,
        *,
        init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"] | None = None,
        batch_size: Int = 1024,
        beta_loss: float | Literal["frobenius", "kullback-leibler", "itakura-saito"] = "frobenius",
        tol: Float = 1e-4,
        max_no_improvement: Int = 10,
        max_iter: Int = 200,
        alpha_W: Float = 0.0,
        alpha_H: float | Literal["same"] = "same",
        l1_ratio: Float = 0.0,
        forget_factor: Float = 0.7,
        fresh_restarts: bool = False,
        fresh_restarts_max_iter: Int = 30,
        transform_max_iter: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: int | bool = 0,
    ) -> None: ...
    def fit_transform(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        W: None | MatrixLike = None,
        H: None | ArrayLike = None,
    ) -> ndarray: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def partial_fit(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        W: None | MatrixLike = None,
        H: None | ArrayLike = None,
    ) -> Self: ...
