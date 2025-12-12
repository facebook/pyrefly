from collections.abc import Callable, Sequence
from typing import Concatenate, Literal, TypeAlias, TypeVar, TypedDict, overload, type_check_only

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["argrelextrema", "argrelmax", "argrelmin", "find_peaks", "find_peaks_cwt", "peak_prominences", "peak_widths"]

###

_SCT = TypeVar("_SCT", bound=np.generic)

_Int1D: TypeAlias = onp.Array1D[np.intp]
_IntND: TypeAlias = onp.ArrayND[np.intp]
_Float1D: TypeAlias = onp.Array1D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]

_Mode: TypeAlias = Literal["clip", "wrap"]

_ArgRel: TypeAlias = tuple[_IntND, ...]
_PeakProminences: TypeAlias = tuple[_FloatND, _IntND, _IntND]
_PeakWidths: TypeAlias = tuple[_FloatND, _FloatND, _FloatND, _FloatND]

_PeakCondition: TypeAlias = onp.ToFloat | onp.ToFloatND | Sequence[onp.ToFloat | None]

_WaveletFunc: TypeAlias = (
    Callable[Concatenate[int, float, ...], onp.ToComplex1D] | Callable[Concatenate[np.intp, np.float64, ...], onp.ToComplex1D]
)

###

def argrelmin(data: onp.Array, axis: op.CanIndex = 0, order: onp.ToInt = 1, mode: _Mode = "clip") -> _ArgRel: ...
def argrelmax(data: onp.Array, axis: op.CanIndex = 0, order: onp.ToInt = 1, mode: _Mode = "clip") -> _ArgRel: ...
def argrelextrema(
    data: onp.ArrayND[_SCT],
    comparator: Callable[[onp.ArrayND[_SCT], onp.ArrayND[_SCT]], onp.ToBoolND],
    axis: op.CanIndex = 0,
    order: onp.ToInt = 1,
    mode: _Mode = "clip",
) -> _ArgRel: ...

#
def peak_prominences(x: onp.ToArray1D, peaks: onp.ToIntND, wlen: onp.ToFloat | None = None) -> _PeakProminences: ...
def peak_widths(
    x: onp.ToArray1D,
    peaks: onp.ToIntND,
    rel_height: onp.ToFloat = 0.5,
    prominence_data: _PeakProminences | None = None,
    wlen: onp.ToFloat | None = None,
) -> _PeakWidths: ...

#
def find_peaks_cwt(
    vector: onp.Array,
    widths: onp.ToFloat | onp.ToFloatND,
    wavelet: _WaveletFunc | None = None,
    max_distances: onp.ArrayND[npc.floating | npc.integer] | None = None,
    gap_thresh: onp.ToFloat | None = None,
    min_length: onp.ToInt | None = None,
    min_snr: onp.ToFloat = 1,
    noise_perc: onp.ToFloat = 10,
    window_size: onp.ToInt | None = None,
) -> _Int1D: ...

# We need these 2^5=32 (combinations of) TypedDicts for each combination of 5 optional find_peaks parameters
# https://github.com/scipy/scipy-stubs/issues/944#issuecomment-3413406314

# 0 (5 choose 0 = 1)

@type_check_only  # {}
class _PeakProperties_0(TypedDict): ...

# 1 (5 choose 1 = 5)

@type_check_only  # {height}
class _PeakProperties_h(TypedDict):
    peak_heights: _Float1D

@type_check_only  # {threshold}
class _PeakProperties_t(TypedDict):
    left_thresholds: _Float1D
    right_thresholds: _Float1D

@type_check_only  # {prominence}
class _PeakProperties_p(TypedDict):
    prominences: _Float1D
    left_bases: _Int1D
    right_bases: _Int1D

@type_check_only  # {width}
class _PeakProperties_w(TypedDict):
    widths: _Float1D
    width_heights: _Float1D
    left_ips: _Float1D
    right_ips: _Float1D

@type_check_only  # {plateau_size}
class _PeakProperties_s(TypedDict):
    plateau_sizes: _Int1D
    left_edges: _Int1D
    right_edges: _Int1D

# 2 (5 choose 2 = 10)

@type_check_only  # {height, threshold}
class _PeakProperties_ht(_PeakProperties_h, _PeakProperties_t): ...

@type_check_only  # {height, prominence}
class _PeakProperties_hp(_PeakProperties_h, _PeakProperties_p): ...

@type_check_only  # {height, width}
class _PeakProperties_hw(_PeakProperties_h, _PeakProperties_w): ...

@type_check_only  # {height, plateau_size}
class _PeakProperties_hs(_PeakProperties_h, _PeakProperties_s): ...

@type_check_only  # {threshold, prominence}
class _PeakProperties_tp(_PeakProperties_t, _PeakProperties_p): ...

@type_check_only  # {threshold, width}
class _PeakProperties_tw(_PeakProperties_t, _PeakProperties_w): ...

@type_check_only  # {threshold, plateau_size}
class _PeakProperties_ts(_PeakProperties_t, _PeakProperties_s): ...

@type_check_only  # {prominence, width}
class _PeakProperties_pw(_PeakProperties_p, _PeakProperties_w): ...

@type_check_only  # {prominence, plateau_size}
class _PeakProperties_ps(_PeakProperties_p, _PeakProperties_s): ...

@type_check_only  # {width, plateau_size}
class _PeakProperties_ws(_PeakProperties_w, _PeakProperties_s): ...

# 3 (5 choose 3 = 10)

@type_check_only  # {height, threshold, prominence}
class _PeakProperties_htp(_PeakProperties_ht, _PeakProperties_p): ...

@type_check_only  # {height, threshold, width}
class _PeakProperties_htw(_PeakProperties_ht, _PeakProperties_w): ...

@type_check_only  # {height, threshold, plateau_size}
class _PeakProperties_hts(_PeakProperties_ht, _PeakProperties_s): ...

@type_check_only  # {height, prominence, width}
class _PeakProperties_hpw(_PeakProperties_hp, _PeakProperties_w): ...

@type_check_only  # {height, prominence, plateau_size}
class _PeakProperties_hps(_PeakProperties_hp, _PeakProperties_s): ...

@type_check_only  # {height, width, plateau_size}
class _PeakProperties_hws(_PeakProperties_hw, _PeakProperties_s): ...

@type_check_only  # {threshold, prominence, width}
class _PeakProperties_tpw(_PeakProperties_t, _PeakProperties_pw): ...

@type_check_only  # {threshold, prominence, plateau_size}
class _PeakProperties_tps(_PeakProperties_t, _PeakProperties_ps): ...

@type_check_only  # {threshold, width, plateau_size}
class _PeakProperties_tws(_PeakProperties_t, _PeakProperties_ws): ...

@type_check_only  # {prominence, width, plateau_size}
class _PeakProperties_pws(_PeakProperties_p, _PeakProperties_ws): ...

# 4 (5 choose 4 = 5)

@type_check_only  # {height, threshold, prominence, width}
class _PeakProperties_htpw(_PeakProperties_ht, _PeakProperties_pw): ...

@type_check_only  # {height, threshold, prominence, plateau_size}
class _PeakProperties_htps(_PeakProperties_ht, _PeakProperties_ps): ...

@type_check_only  # {height, threshold, width, plateau_size}
class _PeakProperties_htws(_PeakProperties_ht, _PeakProperties_ws): ...

@type_check_only  # {height, prominence, width, plateau_size}
class _PeakProperties_hpws(_PeakProperties_hp, _PeakProperties_ws): ...

@type_check_only  # {threshold, prominence, width, plateau_size}
class _PeakProperties_tpws(_PeakProperties_t, _PeakProperties_pws): ...

# 5 (5 choose 5 = 1)

@type_check_only  # {height, threshold, prominence, width, plateau_size}
class _PeakProperties_htpws(_PeakProperties_htp, _PeakProperties_ws): ...

# 0
@overload  # {}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    threshold: None = None,
    distance: float | None = None,
    prominence: None = None,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_0]: ...

# 1
@overload  # {height}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: None = None,
    distance: float | None = None,
    prominence: None = None,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_h]: ...
@overload  # {threshold}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: None = None,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_t]: ...
@overload  # {prominence}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    threshold: None = None,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_p]: ...
@overload  # {width}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    threshold: None = None,
    distance: float | None = None,
    prominence: None = None,
    *,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_w]: ...
@overload  # {plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    threshold: None = None,
    distance: float | None = None,
    prominence: None = None,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    *,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_s]: ...

# 2
@overload  # {height, threshold}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: None = None,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_ht]: ...
@overload  # {height, prominence}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: None = None,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_hp]: ...
@overload  # {height, width}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: None = None,
    distance: float | None = None,
    prominence: None = None,
    *,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_hw]: ...
@overload  # {height, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: None = None,
    distance: float | None = None,
    prominence: None = None,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    *,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_hs]: ...
@overload  # {threshold, prominence}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: _PeakCondition,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_tp]: ...
@overload  # {threshold, width}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: None = None,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_tw]: ...
@overload  # {threshold, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: None = None,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_ts]: ...
@overload  # {prominence, width}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    threshold: None = None,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_pw]: ...
@overload  # {prominence, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    threshold: None = None,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_ps]: ...
@overload  # {width, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    threshold: None = None,
    distance: float | None = None,
    prominence: None = None,
    *,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_ws]: ...

# 3
@overload  # {height, threshold, prominence}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: _PeakCondition,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_htp]: ...
@overload  # {height, threshold, width}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: None = None,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_htw]: ...
@overload  # {height, threshold, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: None = None,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_hts]: ...
@overload  # {height, prominence, width}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: None = None,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_hpw]: ...
@overload  # {height, prominence, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: None = None,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_hps]: ...
@overload  # {height, width, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: None = None,
    distance: float | None = None,
    prominence: None = None,
    *,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_hws]: ...
@overload  # {threshold, prominence, width}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: _PeakCondition,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_tpw]: ...
@overload  # {threshold, prominence, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: _PeakCondition,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_tps]: ...
@overload  # {threshold, width, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: None = None,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_tws]: ...
@overload  # {prominence, width, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    threshold: None = None,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_pws]: ...

# 4
@overload  # {height, threshold, prominence, width}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: _PeakCondition,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: None = None,
) -> tuple[_Int1D, _PeakProperties_htpw]: ...
@overload  # {height, threshold, prominence, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: _PeakCondition,
    width: None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_htps]: ...
@overload  # {height, threshold, width, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: None = None,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_htws]: ...
@overload  # {height, prominence, width, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: None = None,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_hpws]: ...
@overload  # {threshold, prominence, width, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: None = None,
    *,
    threshold: _PeakCondition,
    distance: float | None = None,
    prominence: _PeakCondition,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_tpws]: ...

# 5
@overload  # {height, threshold, prominence, width, plateau_size}
def find_peaks(
    x: onp.ToFloat1D,
    height: _PeakCondition,
    threshold: _PeakCondition,
    distance: float | None = None,
    *,
    prominence: _PeakCondition,
    width: _PeakCondition,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: _PeakCondition,
) -> tuple[_Int1D, _PeakProperties_htpws]: ...
