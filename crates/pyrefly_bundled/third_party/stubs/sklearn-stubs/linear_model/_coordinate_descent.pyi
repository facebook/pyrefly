from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse import spmatrix
from scipy.sparse._coo import coo_matrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import MultiOutputMixin, RegressorMixin
from ..model_selection import BaseCrossValidator
from ._base import LinearModel

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Gael Varoquaux <gael.varoquaux@inria.fr>
#
# License: BSD 3 clause

def lasso_path(
    X: MatrixLike | ArrayLike,
    y: MatrixLike | ArrayLike,
    *,
    eps: Float = 1e-3,
    n_alphas: Int = 100,
    alphas: None | ArrayLike = None,
    precompute: Literal["auto"] | MatrixLike | bool = "auto",
    Xy: None | MatrixLike | ArrayLike = None,
    copy_X: bool = True,
    coef_init: None | ArrayLike = None,
    verbose: int | bool = False,
    return_n_iter: bool = False,
    positive: bool = False,
    **params,
) -> tuple[ndarray, ndarray, ndarray] | tuple[ndarray, ndarray, ndarray, list[int]]: ...
def enet_path(
    X: MatrixLike | ArrayLike,
    y: MatrixLike | ArrayLike,
    *,
    l1_ratio: Float = 0.5,
    eps: Float = 1e-3,
    n_alphas: Int = 100,
    alphas: None | ArrayLike = None,
    precompute: Literal["auto"] | MatrixLike | bool = "auto",
    Xy: None | MatrixLike | ArrayLike = None,
    copy_X: bool = True,
    coef_init: None | ArrayLike = None,
    verbose: int | bool = False,
    return_n_iter: bool = False,
    positive: bool = False,
    check_input: bool = True,
    **params,
) -> tuple[ndarray, ndarray, ndarray, list[int]]: ...

###############################################################################
# ElasticNet model

class ElasticNet(MultiOutputMixin, RegressorMixin, LinearModel):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    dual_gap_: float | ndarray = ...
    n_iter_: list[int] = ...
    intercept_: float | ndarray = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    path = ...

    def __init__(
        self,
        alpha: Float = 1.0,
        *,
        l1_ratio: Float = 0.5,
        fit_intercept: bool = True,
        precompute: MatrixLike | bool = False,
        max_iter: Int = 1000,
        copy_X: bool = True,
        tol: Float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        random_state: RandomState | None | Int = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ) -> None: ...
    def fit(
        self,
        X: coo_matrix | MatrixLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
        check_input: bool = True,
    ) -> Self: ...
    @property
    def sparse_coef_(self) -> spmatrix: ...

###############################################################################
# Lasso model

class Lasso(ElasticNet):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int | list[int] = ...
    intercept_: float | ndarray = ...
    sparse_coef_: spmatrix = ...
    dual_gap_: float | ndarray = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    path = ...

    def __init__(
        self,
        alpha: Float = 1.0,
        *,
        fit_intercept: bool = True,
        precompute: MatrixLike | bool = False,
        copy_X: bool = True,
        max_iter: Int = 1000,
        tol: Float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        random_state: RandomState | None | Int = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ) -> None: ...

class LinearModelCV(MultiOutputMixin, LinearModel, ABC):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: None | ndarray = None,
        fit_intercept: bool = True,
        precompute: str = "auto",
        max_iter: int = 1000,
        tol: float = 1e-4,
        copy_X: bool = True,
        cv: None | int = None,
        verbose: bool = False,
        n_jobs=None,
        positive: bool = False,
        random_state: None | int = None,
        selection: str = "cyclic",
    ) -> None: ...
    @staticmethod
    @abstractmethod
    def path(X, y, **kwargs): ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self | LassoCV: ...

class LassoCV(RegressorMixin, LinearModelCV):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    dual_gap_: float | ndarray = ...
    alphas_: ndarray = ...
    mse_path_: ndarray = ...
    intercept_: float | ndarray = ...
    coef_: ndarray = ...
    alpha_: float = ...

    path = ...

    def __init__(
        self,
        *,
        eps: Float = 1e-3,
        n_alphas: Int = 100,
        alphas: None | ArrayLike = None,
        fit_intercept: bool = True,
        precompute: Literal["auto"] | MatrixLike | bool = "auto",
        max_iter: Int = 1000,
        tol: Float = 1e-4,
        copy_X: bool = True,
        cv: int | BaseCrossValidator | Iterable | None = None,
        verbose: int | bool = False,
        n_jobs: None | Int = None,
        positive: bool = False,
        random_state: RandomState | None | Int = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ) -> None: ...

class ElasticNetCV(RegressorMixin, LinearModelCV):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    dual_gap_: float = ...
    alphas_: ndarray = ...
    mse_path_: ndarray = ...
    intercept_: float | ndarray = ...
    coef_: ndarray = ...
    l1_ratio_: float = ...
    alpha_: float = ...

    _parameter_constraints: ClassVar[dict] = ...

    path = ...

    def __init__(
        self,
        *,
        l1_ratio: float | Sequence[float] = 0.5,
        eps: Float = 1e-3,
        n_alphas: Int = 100,
        alphas: None | ArrayLike = None,
        fit_intercept: bool = True,
        precompute: Literal["auto"] | MatrixLike | bool = "auto",
        max_iter: Int = 1000,
        tol: Float = 1e-4,
        cv: int | BaseCrossValidator | Iterable | None = None,
        copy_X: bool = True,
        verbose: int | bool = 0,
        n_jobs: None | Int = None,
        positive: bool = False,
        random_state: RandomState | None | Int = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ) -> None: ...

###############################################################################
# Multi Task ElasticNet and Lasso models (with joint feature selection)

class MultiTaskElasticNet(Lasso):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    sparse_coef_: spmatrix = ...
    eps_: float = ...
    dual_gap_: float = ...
    n_iter_: int = ...
    coef_: ndarray = ...
    intercept_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        alpha: Float = 1.0,
        *,
        l1_ratio: Float = 0.5,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: Int = 1000,
        tol: Float = 1e-4,
        warm_start: bool = False,
        random_state: RandomState | None | Int = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ) -> None: ...
    def fit(self, X: ArrayLike, y: MatrixLike) -> Self | MultiTaskLasso: ...

class MultiTaskLasso(MultiTaskElasticNet):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    sparse_coef_: spmatrix = ...
    eps_: float = ...
    dual_gap_: ndarray = ...
    n_iter_: int = ...
    intercept_: ndarray = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        alpha: Float = 1.0,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: Int = 1000,
        tol: Float = 1e-4,
        warm_start: bool = False,
        random_state: RandomState | None | Int = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ) -> None: ...

class MultiTaskElasticNetCV(RegressorMixin, LinearModelCV):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    dual_gap_: float = ...
    n_iter_: int = ...
    l1_ratio_: float = ...
    alphas_: ndarray = ...
    mse_path_: ndarray = ...
    alpha_: float = ...
    coef_: ndarray = ...
    intercept_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    path = ...

    def __init__(
        self,
        *,
        l1_ratio: float | Sequence[float] = 0.5,
        eps: Float = 1e-3,
        n_alphas: Int = 100,
        alphas: None | ArrayLike = None,
        fit_intercept: bool = True,
        max_iter: Int = 1000,
        tol: Float = 1e-4,
        cv: int | BaseCrossValidator | Iterable | None = None,
        copy_X: bool = True,
        verbose: int | bool = 0,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ) -> None: ...

    # This is necessary as LinearModelCV now supports sample_weight while
    # MultiTaskElasticNet does not (yet).
    def fit(self, X: ArrayLike, y: MatrixLike) -> Self: ...

class MultiTaskLassoCV(RegressorMixin, LinearModelCV):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    dual_gap_: float = ...
    n_iter_: int = ...
    alphas_: ndarray = ...
    mse_path_: ndarray = ...
    alpha_: float = ...
    coef_: ndarray = ...
    intercept_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    path = ...

    def __init__(
        self,
        *,
        eps: Float = 1e-3,
        n_alphas: Int = 100,
        alphas: None | ArrayLike = None,
        fit_intercept: bool = True,
        max_iter: Int = 1000,
        tol: Float = 1e-4,
        copy_X: bool = True,
        cv: int | BaseCrossValidator | Iterable | None = None,
        verbose: int | bool = False,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ) -> None: ...

    # This is necessary as LinearModelCV now supports sample_weight while
    # MultiTaskElasticNet does not (yet).
    def fit(self, X: ArrayLike, y: MatrixLike) -> Self: ...
