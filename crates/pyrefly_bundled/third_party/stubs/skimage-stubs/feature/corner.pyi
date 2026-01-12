from itertools import combinations_with_replacement
from typing import Literal
from warnings import warn

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi, spatial, stats

from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int
from ..transform import integral_image
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD

def _compute_derivatives(image, mode="constant", cval=0): ...
def structure_tensor(
    image: NDArray,
    sigma: float | ArrayLike = 1,
    mode: Literal["constant", "reflect", "wrap", "nearest", "mirror"] = "constant",
    cval: float = 0,
    order: Literal["rc", "xy"] | None = None,
) -> ArrayLike: ...
def hessian_matrix(
    image: NDArray,
    sigma: float = 1,
    mode: Literal["constant", "reflect", "wrap", "nearest", "mirror"] = "constant",
    cval: float = 0,
    order: Literal["rc", "xy"] = "rc",
) -> ArrayLike: ...
def hessian_matrix_det(image: NDArray, sigma: float = 1, approximate: bool = True) -> ArrayLike: ...
def _image_orthogonal_matrix22_eigvals(M00, M01, M11): ...
def _symmetric_compute_eigenvalues(S_elems): ...
def _symmetric_image(S_elems): ...
def structure_tensor_eigenvalues(A_elems: ArrayLike) -> NDArray: ...
def structure_tensor_eigvals(Axx: NDArray, Axy: NDArray, Ayy: NDArray) -> tuple[NDArray, NDArray]: ...
def hessian_matrix_eigvals(H_elems: ArrayLike) -> NDArray: ...
def shape_index(
    image,
    sigma: float = 1,
    mode: Literal["constant", "reflect", "wrap", "nearest", "mirror"] = "constant",
    cval: float = 0,
) -> NDArray: ...
def corner_kitchen_rosenfeld(
    image,
    mode: Literal["constant", "reflect", "wrap", "nearest", "mirror"] = "constant",
    cval: float = 0,
) -> NDArray: ...
def corner_harris(
    image,
    method: Literal["k", "eps"] = "k",
    k: float = 0.05,
    eps: float = 1e-6,
    sigma: float = 1,
) -> NDArray: ...
def corner_shi_tomasi(image, sigma: float = 1) -> NDArray: ...
def corner_foerstner(image, sigma: float = 1) -> tuple[NDArray, NDArray]: ...
def corner_fast(image, n: int = 12, threshold: float = 0.15) -> NDArray: ...
def corner_subpix(image, corners, window_size: int = 11, alpha: float = 0.99): ...
def corner_peaks(
    image,
    min_distance: int = 1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    indices=True,
    num_peaks=...,
    footprint=None,
    labels=None,
    *,
    num_peaks_per_label=...,
    p_norm: float = ...,
) -> np.ndarray: ...
def corner_moravec(image, window_size: int = 1) -> NDArray: ...
def corner_orientations(image, corners, mask): ...
