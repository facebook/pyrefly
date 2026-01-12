import math as math
import numbers as numbers
import platform as platform
import struct as struct
import timeit as timeit
import warnings as warnings
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager as contextmanager, suppress as suppress
from itertools import compress as compress, islice as islice
from typing import Any, SupportsIndex

import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse import issparse as issparse

from .. import get_config as get_config
from .._typing import ArrayLike, Int, MatrixLike
from ..exceptions import DataConversionWarning as DataConversionWarning
from . import metadata_routing
from ._bunch import Bunch as Bunch
from ._estimator_html_repr import estimator_html_repr as estimator_html_repr
from .class_weight import compute_class_weight as compute_class_weight, compute_sample_weight as compute_sample_weight
from .deprecation import deprecated as deprecated
from .discovery import all_estimators as all_estimators
from .murmurhash import murmurhash3_32 as murmurhash3_32
from .validation import (
    as_float_array as as_float_array,
    assert_all_finite as assert_all_finite,
    check_array as check_array,
    check_consistent_length as check_consistent_length,
    check_random_state as check_random_state,
    check_scalar as check_scalar,
    check_symmetric as check_symmetric,
    check_X_y as check_X_y,
    column_or_1d as column_or_1d,
    indexable as indexable,
)

# Do not deprecate parallel_backend and register_parallel_backend as they are
# needed to tune `scikit-learn` behavior and have different effect if called
# from the vendored version or or the site-package version. The other are
# utilities that are independent of scikit-learn so they are not part of
# scikit-learn public API.
parallel_backend = ...
register_parallel_backend = ...

__all__ = [
    "murmurhash3_32",
    "as_float_array",
    "assert_all_finite",
    "check_array",
    "check_random_state",
    "compute_class_weight",
    "compute_sample_weight",
    "column_or_1d",
    "check_consistent_length",
    "check_X_y",
    "check_scalar",
    "indexable",
    "check_symmetric",
    "deprecated",
    "parallel_backend",
    "register_parallel_backend",
    "resample",
    "shuffle",
    "all_estimators",
    "DataConversionWarning",
    "estimator_html_repr",
    "Bunch",
    "metadata_routing",
    "safe_sqr",
    "safe_mask",
    "gen_batches",
    "gen_even_slices",
]

IS_PYPY = ...
_IS_32BIT = ...

def safe_mask(X: MatrixLike | ArrayLike, mask: ArrayLike) -> ndarray: ...
def axis0_safe_slice(X: MatrixLike | ArrayLike, mask: ArrayLike, len_mask: Int) -> ndarray: ...
def resample(
    *arrays,
    replace: bool = True,
    n_samples: None | Int = None,
    random_state: RandomState | None | Int = None,
    stratify: None | MatrixLike | ArrayLike = None,
) -> list[ndarray] | None: ...
def shuffle(
    *arrays, random_state: RandomState | None | Int = None, n_samples: None | Int = None
) -> list[SupportsIndex] | None: ...
def safe_sqr(X: MatrixLike | ArrayLike, *, copy: bool = True) -> ndarray: ...
def gen_batches(n: Int, batch_size: Int, *, min_batch_size: Int = 0) -> Iterator[slice]: ...
def gen_even_slices(n: Int, n_packs: Int, *, n_samples: None | Int = None) -> Iterator[slice]: ...
def tosequence(x: Iterable) -> list: ...
def indices_to_mask(indices: Sequence | ndarray, mask_length: Int) -> ndarray: ...
def get_chunk_n_rows(row_bytes: Int, *, max_n_rows: None | Int = None, working_memory: float | None = None) -> int: ...
def is_scalar_nan(x: Any) -> bool: ...
def check_matplotlib_support(caller_name: str) -> None: ...
def check_pandas_support(caller_name: str): ...
