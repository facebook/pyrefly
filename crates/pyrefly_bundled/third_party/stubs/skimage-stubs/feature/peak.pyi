from warnings import warn

import numpy as np
import scipy.ndimage as ndi
from numpy import ndarray
from numpy.typing import NDArray

from .. import measure
from .._shared.coord import ensure_spacing
from .._shared.utils import remove_arg

def _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm): ...
def _get_peak_mask(image, footprint, threshold, mask=None): ...
def _exclude_border(label, border_width): ...
def _get_threshold(image, threshold_abs, threshold_rel): ...
def _get_excluded_border_width(image, min_distance, exclude_border): ...
@remove_arg("indices", changed_version="0.20")
def peak_local_max(
    image: NDArray,
    min_distance: int = 1,
    threshold_abs: float | None = None,
    threshold_rel: float | None = None,
    exclude_border=True,
    indices: bool = True,
    num_peaks: int = ...,
    footprint: NDArray | None = None,
    labels: NDArray | None = None,
    num_peaks_per_label: int = ...,
    p_norm: float = ...,
) -> np.ndarray: ...
def _prominent_peaks(image, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=np.inf): ...
