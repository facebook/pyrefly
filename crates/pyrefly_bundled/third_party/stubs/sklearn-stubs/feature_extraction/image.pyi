from itertools import product as product
from numbers import Integral as Integral, Number as Number, Real as Real
from typing import Any, ClassVar
from typing_extensions import Self

import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import as_strided as as_strided
from numpy.random import RandomState
from scipy import sparse as sparse
from scipy.sparse import spmatrix
from scipy.sparse._coo import coo_matrix

from .._typing import ArrayLike, Int, MatrixLike
from ..base import BaseEstimator
from ..utils import check_array as check_array, check_random_state as check_random_state
from ..utils._param_validation import Interval as Interval

__all__ = [
    "PatchExtractor",
    "extract_patches_2d",
    "grid_to_graph",
    "img_to_graph",
    "reconstruct_from_patches_2d",
]

def img_to_graph(
    img: MatrixLike, *, mask: None | MatrixLike = None, return_as: MatrixLike | ArrayLike = ..., dtype=None
) -> coo_matrix | ndarray: ...
def grid_to_graph(
    n_x: Int, n_y: Int, n_z: Int = 1, *, mask: None | MatrixLike = None, return_as: MatrixLike | ArrayLike = ..., dtype=...
) -> ndarray | spmatrix: ...
def extract_patches_2d(
    image: MatrixLike,
    patch_size: tuple[int, int],
    *,
    max_patches: float | None = None,
    random_state: RandomState | None | Int = None,
) -> ndarray: ...
def reconstruct_from_patches_2d(patches: MatrixLike, image_size: tuple[int, int, int] | tuple[int, int]) -> ndarray: ...

class PatchExtractor(BaseEstimator):
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        patch_size: None | tuple[int, int] = None,
        max_patches: float | None = None,
        random_state: RandomState | None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
    def transform(self, X: MatrixLike) -> ndarray: ...
