from typing import Literal

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from .._shared.utils import check_nD

def graycomatrix(
    image,
    distances,
    angles,
    levels: int | None = None,
    symmetric: bool = False,
    normed: bool = False,
): ...
def graycoprops(
    P: NDArray,
    prop: Literal["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"] = "contrast",
): ...
def local_binary_pattern(
    image,
    P: int,
    R: float,
    method: Literal["default", "ror", "uniform", "var"] = "default",
): ...
def multiblock_lbp(int_image, r: int, c: int, width: int, height: int) -> int: ...
def draw_multiblock_lbp(
    image,
    r: int,
    c: int,
    width: int,
    height: int,
    lbp_code: int = 0,
    color_greater_block: tuple[float, float, float] = ...,
    color_less_block=...,
    alpha: float = 0.5,
) -> np.ndarray: ...
