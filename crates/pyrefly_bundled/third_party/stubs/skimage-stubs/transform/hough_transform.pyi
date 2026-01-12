import numpy as np
from numpy.typing import ArrayLike, NDArray

def hough_line_peaks(
    hspace,
    angles,
    dists,
    min_distance: int = 9,
    min_angle: int = 10,
    threshold: float | None = None,
    num_peaks: int = ...,
) -> tuple[NDArray, ...]: ...
def hough_circle(image, radius, normalize=True, full_output=False): ...
def hough_ellipse(
    image,
    threshold: int = 4,
    accuracy=1,
    min_size: int = 4,
    max_size: int | None = None,
): ...
def hough_line(image, theta=None): ...
def probabilistic_hough_line(
    image,
    threshold: int = 10,
    line_length: int = 50,
    line_gap: int = 10,
    theta=None,
    seed: int | None = None,
) -> ArrayLike: ...
def hough_circle_peaks(
    hspaces,
    radii,
    min_xdistance: int = 1,
    min_ydistance: int = 1,
    threshold: float | None = None,
    num_peaks: int = ...,
    total_num_peaks: int = ...,
    normalize: bool = False,
) -> tuple[NDArray, ...]: ...
def label_distant_points(
    xs: ArrayLike,
    ys: ArrayLike,
    min_xdistance: int,
    min_ydistance: int,
    max_points: int,
) -> ArrayLike: ...
