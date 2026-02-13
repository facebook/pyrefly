# Author: Arnaud Joly, Joel Nothman, Hamzeh Alsalhi
#
# License: BSD 3 clause
import warnings
from collections.abc import Sequence as Sequence
from itertools import chain as chain

import numpy as np
from numpy import ndarray
from scipy.sparse import dok_matrix as dok_matrix, issparse as issparse, lil_matrix as lil_matrix

from .._typing import ArrayLike, MatrixLike
from ..utils._array_api import get_namespace as get_namespace
from .validation import check_array as check_array

_FN_UNIQUE_LABELS: dict = ...

def unique_labels(*ys) -> ndarray: ...
def is_multilabel(y: ArrayLike) -> bool: ...
def check_classification_targets(y: ArrayLike) -> None: ...
def type_of_target(y: MatrixLike | ArrayLike, input_name: str = "") -> str: ...
def class_distribution(
    y: MatrixLike, sample_weight: None | ArrayLike = None
) -> tuple[list[ndarray], list[int], list[ndarray]]: ...
