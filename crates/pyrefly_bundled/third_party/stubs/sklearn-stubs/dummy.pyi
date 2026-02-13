import warnings
from numbers import Integral as Integral, Real as Real
from typing import ClassVar, Literal
from typing_extensions import Self

import numpy as np
import scipy.sparse as sp
from numpy import ndarray
from numpy.random import RandomState

from ._typing import ArrayLike, Float, Int, MatrixLike
from .base import BaseEstimator, ClassifierMixin, MultiOutputMixin, RegressorMixin
from .utils import check_random_state as check_random_state
from .utils._param_validation import Interval as Interval, StrOptions as StrOptions
from .utils.multiclass import class_distribution as class_distribution
from .utils.validation import (
    check_array as check_array,
    check_consistent_length as check_consistent_length,
    check_is_fitted as check_is_fitted,
)

# Author: Mathieu Blondel <mathieu@mblondel.org>
#         Arnaud Joly <a.joly@ulg.ac.be>
#         Maheshakya Wijewardena <maheshakya.10@cse.mrt.ac.lk>
# License: BSD 3 clause

class DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):
    sparse_output_: bool = ...
    n_outputs_: int = ...
    class_prior_: ndarray | list[ArrayLike] = ...
    n_classes_: int | list[int] = ...
    classes_: ndarray | list[ArrayLike] = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        strategy: Literal["most_frequent", "prior", "stratified", "uniform", "constant"] = "prior",
        random_state: RandomState | None | Int = None,
        constant: None | str | ArrayLike | int = None,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, X: MatrixLike) -> ndarray: ...
    def predict_proba(self, X: MatrixLike) -> ndarray | list[ndarray]: ...
    def predict_log_proba(self, X: ArrayLike) -> ndarray | list[ndarray]: ...
    def score(
        self,
        X: None | MatrixLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> float: ...

class DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    n_outputs_: int = ...
    constant_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        strategy: Literal["mean", "median", "quantile", "constant"] = "mean",
        constant: float | None | ArrayLike = None,
        quantile: float | None = None,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, X: MatrixLike, return_std: bool = False) -> ndarray | tuple[ndarray, ndarray]: ...
    def score(
        self,
        X: None | MatrixLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Float: ...
