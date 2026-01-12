import warnings
from collections.abc import Iterable
from functools import partial as partial
from inspect import signature as signature
from math import log as log
from numbers import Integral as Integral
from typing import ClassVar, Literal
from typing_extensions import Self

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy.optimize import fmin_bfgs as fmin_bfgs
from scipy.special import expit as expit, xlogy as xlogy

from ._typing import ArrayLike, Int, MatrixLike
from .base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    clone as clone,
    is_classifier as is_classifier,
)
from .isotonic import IsotonicRegression
from .model_selection import BaseCrossValidator, check_cv as check_cv, cross_val_predict as cross_val_predict
from .preprocessing import LabelEncoder as LabelEncoder, label_binarize as label_binarize
from .svm import LinearSVC as LinearSVC
from .utils import check_matplotlib_support as check_matplotlib_support, column_or_1d as column_or_1d, indexable as indexable
from .utils._param_validation import HasMethods as HasMethods, Hidden as Hidden, StrOptions as StrOptions
from .utils.multiclass import check_classification_targets as check_classification_targets
from .utils.parallel import Parallel as Parallel, delayed as delayed
from .utils.validation import check_consistent_length as check_consistent_length, check_is_fitted as check_is_fitted

class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    calibrated_classifiers_: list = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    classes_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        estimator: None | BaseEstimator = None,
        *,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        cv: int | BaseCrossValidator | Iterable | None | str = None,
        n_jobs: None | Int = None,
        ensemble: bool = True,
        base_estimator: str | BaseEstimator = "deprecated",
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
        **fit_params,
    ) -> Self: ...
    def predict_proba(self, X: MatrixLike) -> ndarray: ...
    def predict(self, X: MatrixLike) -> ndarray: ...

class _CalibratedClassifier:
    def __init__(
        self,
        estimator: BaseEstimator,
        calibrators: list[IsotonicRegression | _SigmoidCalibration] | list[BaseEstimator],
        *,
        classes: ArrayLike,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
    ) -> None: ...
    def predict_proba(self, X: ArrayLike) -> ndarray: ...

class _SigmoidCalibration(RegressorMixin, BaseEstimator):
    b_: float = ...
    a_: float = ...

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, T: ArrayLike) -> ndarray: ...

def calibration_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: None | str | Int = None,
    normalize: str | bool = "deprecated",
    n_bins: Int = 5,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> tuple[ndarray, ndarray]: ...

class CalibrationDisplay:
    figure_: Figure = ...
    ax_: Axes = ...
    line_: Artist = ...

    def __init__(
        self,
        prob_true: ArrayLike,
        prob_pred: ArrayLike,
        y_prob: ArrayLike,
        *,
        estimator_name: None | str = None,
        pos_label: None | str | int = None,
    ) -> None: ...
    def plot(
        self,
        *,
        ax: None | Axes = None,
        name: None | str = None,
        ref_line: bool = True,
        **kwargs,
    ) -> CalibrationDisplay: ...
    @classmethod
    def from_estimator(
        cls,
        estimator: BaseEstimator,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        *,
        n_bins: Int = 5,
        strategy: Literal["uniform", "quantile"] = "uniform",
        pos_label: None | str | int = None,
        name: None | str = None,
        ref_line: bool = True,
        ax: None | Axes = None,
        **kwargs,
    ) -> CalibrationDisplay: ...
    @classmethod
    def from_predictions(
        cls,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        *,
        n_bins: Int = 5,
        strategy: Literal["uniform", "quantile"] = "uniform",
        pos_label: None | str | int = None,
        name: None | str = None,
        ref_line: bool = True,
        ax: None | Axes = None,
        **kwargs,
    ) -> CalibrationDisplay: ...
