import math

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray
from scipy import spatial

from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD
from ..transform import integral_image
from .peak import peak_local_max

# This basic blob detection algorithm is based on:
# http://www.cs.utah.edu/~jfishbau/advimproc/project1/ (04.04.2013)
# Theory behind: https://en.wikipedia.org/wiki/Blob_detection (04.04.2013)

def _compute_disk_overlap(d, r1, r2): ...
def _compute_sphere_overlap(d, r1, r2): ...
def _blob_overlap(blob1, blob2, *, sigma_dim=1): ...
def _prune_blobs(blobs_array, overlap, *, sigma_dim=1): ...
def _format_exclude_border(img_ndim, exclude_border): ...
def blob_dog(
    image: NDArray,
    min_sigma=1,
    max_sigma=50,
    sigma_ratio: float = 1.6,
    threshold: float | None = 0.5,
    overlap: float = 0.5,
    *,
    threshold_rel: float | None = None,
    exclude_border=False,
): ...
def blob_log(
    image: NDArray,
    min_sigma=1,
    max_sigma=50,
    num_sigma: int = 10,
    threshold: float | None = 0.2,
    overlap: float = 0.5,
    log_scale: bool = False,
    *,
    threshold_rel: float | None = None,
    exclude_border=False,
): ...
def blob_doh(
    image,
    min_sigma: float = 1,
    max_sigma: float = 30,
    num_sigma: int = 10,
    threshold: float | None = 0.01,
    overlap: float = 0.5,
    log_scale: bool = False,
    *,
    threshold_rel: float | None = None,
): ...
