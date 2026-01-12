def _preprocess(image, mask, sigma, mode, cval): ...
def _set_local_maxima(magnitude, pts, w_num, w_denum, row_slices, col_slices, out): ...
def _get_local_maxima(isobel, jsobel, magnitude, eroded_mask): ...
def canny(
    image,
    sigma: float = 1.0,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    mask=None,
    use_quantiles: bool = False,
    *,
    mode="constant",
    cval: float = 0.0,
): ...
