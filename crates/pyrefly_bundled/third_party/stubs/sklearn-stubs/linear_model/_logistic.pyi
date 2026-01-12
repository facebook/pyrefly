from collections.abc import Mapping, Sequence
from typing import Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..model_selection import BaseCrossValidator
from ._base import BaseEstimator, LinearClassifierMixin, SparseCoefMixin

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Fabian Pedregosa <f@bianp.net>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#         Lars Buitinck
#         Simon Wu <s8wu@uwaterloo.ca>
#         Arthur Mensch <arthur.mensch@m4x.org

_LOGISTIC_SOLVER_CONVERGENCE_MSG: str = ...

class LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    n_iter_: ndarray = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    intercept_: ndarray = ...
    coef_: ndarray = ...
    classes_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        penalty: Literal["l1", "l2", "elasticnet"] | None = "l2",
        *,
        dual: bool = False,
        tol: Float = 1e-4,
        C: Float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: Float = 1,
        class_weight: None | Mapping | str = None,
        random_state: RandomState | None | Int = None,
        solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs",
        max_iter: Int = 100,
        multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
        verbose: Int = 0,
        warm_start: bool = False,
        n_jobs: None | Int = None,
        l1_ratio: None | Float = None,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict_proba(self, X: MatrixLike) -> ndarray: ...
    def predict_log_proba(self, X: MatrixLike) -> ndarray: ...

class LogisticRegressionCV(LogisticRegression, LinearClassifierMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: ndarray = ...
    l1_ratio_: ndarray = ...
    C_: ndarray = ...
    scores_: dict = ...
    coefs_paths_: ndarray = ...
    l1_ratios_: ndarray = ...
    Cs_: ndarray = ...
    intercept_: ndarray = ...
    coef_: ndarray = ...
    classes_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        Cs: Sequence[float] | int = 10,
        fit_intercept: bool = True,
        cv: int | None | BaseCrossValidator = None,
        dual: bool = False,
        penalty: Literal["l1", "l2", "elasticnet"] = "l2",
        scoring: None | str | Callable = None,
        solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs",
        tol: Float = 1e-4,
        max_iter: Int = 100,
        class_weight: None | Mapping | str = None,
        n_jobs: None | Int = None,
        verbose: Int = 0,
        refit: bool = True,
        intercept_scaling: Float = 1.0,
        multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
        random_state: RandomState | None | Int = None,
        l1_ratios: None | Sequence[float] = None,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def score(self, X: MatrixLike, y: ArrayLike, sample_weight: None | ArrayLike = None) -> float: ...
