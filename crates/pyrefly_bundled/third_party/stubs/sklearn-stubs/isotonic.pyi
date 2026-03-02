import math
import warnings
from numbers import Real as Real
from typing import Callable, ClassVar, Literal
from typing_extensions import Self

import numpy as np
from numpy import ndarray
from scipy import interpolate as interpolate
from scipy.stats import spearmanr as spearmanr

from ._typing import ArrayLike, Float
from .base import BaseEstimator, RegressorMixin, TransformerMixin
from .utils import check_array as check_array, check_consistent_length as check_consistent_length
from .utils._param_validation import Interval as Interval, StrOptions as StrOptions

# Authors: Fabian Pedregosa <fabian@fseoane.net>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nelle Varoquaux <nelle.varoquaux@gmail.com>
# License: BSD 3 clause

__all__ = ["check_increasing", "isotonic_regression", "IsotonicRegression"]

def check_increasing(x: ArrayLike, y: ArrayLike) -> bool: ...
def isotonic_regression(
    y: ArrayLike,
    *,
    sample_weight: None | ArrayLike = None,
    y_min: None | Float = None,
    y_max: None | Float = None,
    increasing: bool = True,
) -> ndarray | list[float]: ...

class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    increasing_: bool = ...
    f_: Callable = ...
    y_thresholds_: ndarray = ...
    X_thresholds_: ndarray = ...
    X_max_: float = ...
    X_min_: float = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        y_min: None | Float = None,
        y_max: None | Float = None,
        increasing: str | bool = True,
        out_of_bounds: Literal["nan", "clip", "raise"] = "nan",
    ) -> None: ...
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def transform(self, T: ArrayLike) -> ndarray: ...
    def predict(self, T: ArrayLike) -> ndarray: ...

    # We implement get_feature_names_out here instead of using
    # `ClassNamePrefixFeaturesOutMixin`` because `input_features` are ignored.
    # `input_features` are ignored because `IsotonicRegression` accepts 1d
    # arrays and the semantics of `feature_names_in_` are not clear for 1d arrays.
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
