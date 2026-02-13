import numbers
import operator
import warnings
from collections.abc import Sequence
from contextlib import suppress as suppress
from functools import wraps as wraps
from inspect import Parameter as Parameter, isclass as isclass, signature as signature
from numbers import Integral, Number, Real
from typing import Any, Callable, Literal

import joblib
import numpy as np
import scipy.sparse as sp
from joblib.memory import Memory
from numpy import ndarray
from numpy.core.numeric import ComplexWarning as ComplexWarning
from numpy.random.mtrand import RandomState
from pandas import DataFrame
from scipy.sparse import spmatrix
from scipy.sparse._coo import coo_matrix

from .. import get_config as _get_config
from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator
from ..exceptions import (
    DataConversionWarning as DataConversionWarning,
    NotFittedError as NotFittedError,
    PositiveSpectrumWarning as PositiveSpectrumWarning,
)
from ..utils._array_api import get_namespace as get_namespace
from ._isfinite import FiniteStatus as FiniteStatus, cy_isfinite as cy_isfinite

FLOAT_DTYPES = ...

def assert_all_finite(
    X: MatrixLike | ArrayLike,
    *,
    allow_nan: bool = False,
    estimator_name: None | str = None,
    input_name: str = "",
) -> None: ...
def as_float_array(
    X: MatrixLike | ArrayLike, *, copy: bool = True, force_all_finite: str | bool = True
) -> ndarray | spmatrix: ...
def check_memory(memory: None | Memory | str) -> Memory: ...
def check_consistent_length(*arrays) -> None: ...
def indexable(*iterables) -> None | list[ndarray | spmatrix | DataFrame]: ...
def check_array(
    array: Any,
    accept_sparse: Sequence[str] | tuple[str, str] | list[str] | str | bool = False,
    *,
    accept_large_sparse: bool = True,
    dtype: None | Sequence[type] | Literal["numeric"] | type = "numeric",
    order: Literal["F", "C"] | None = None,
    copy: bool = False,
    force_all_finite: str | bool = True,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    ensure_min_samples: Int = 1,
    ensure_min_features: Int = 1,
    estimator: None | str | BaseEstimator = None,
    input_name: str = "",
) -> Any: ...
def check_X_y(
    X: MatrixLike | ArrayLike,
    y: MatrixLike | ArrayLike,
    accept_sparse: Sequence[str] | tuple[str, str] | list[str] | str | bool = False,
    *,
    accept_large_sparse: bool = True,
    dtype: None | Sequence[type] | Literal["numeric"] | type = "numeric",
    order: Literal["F", "C"] | None = None,
    copy: bool = False,
    force_all_finite: str | bool = True,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    multi_output: bool = False,
    ensure_min_samples: Int = 1,
    ensure_min_features: Int = 1,
    y_numeric: bool = False,
    estimator: None | str | BaseEstimator = None,
) -> tuple[Any, Any]: ...
def column_or_1d(y: ArrayLike, *, dtype=None, warn: bool = False) -> ndarray: ...
def check_random_state(seed: None | Int | RandomState) -> RandomState: ...
def has_fit_parameter(estimator: Any, parameter: str) -> bool: ...
def check_symmetric(
    array: coo_matrix | MatrixLike | ArrayLike,
    *,
    tol: Float = 1e-10,
    raise_warning: bool = True,
    raise_exception: bool = False,
) -> ndarray | spmatrix: ...
def check_is_fitted(
    estimator: BaseEstimator,
    attributes: tuple[str, ...] | None | Sequence | list[str] | str = None,
    *,
    msg: None | str = None,
    all_or_any: Callable = ...,
) -> None: ...
def check_non_negative(X: MatrixLike | ArrayLike, whom: str) -> None: ...
def check_scalar(
    x: Any,
    name: str,
    target_type: type[Integral | Real] | tuple | type,
    *,
    min_val: None | Float = None,
    max_val: None | Float = None,
    include_boundaries: Literal["left", "right", "both", "neither"] = "both",
) -> Number | Float: ...
