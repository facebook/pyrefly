from collections.abc import Sequence
from typing import (
    Literal,
    overload,
)

import numpy as np
from pandas import (
    Categorical,
    CategoricalDtype,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    Timestamp,
)
from pandas.core.series import Series

from pandas._typing import (
    IntervalT,
    Label,
    np_1darray_float,
    np_1darray_intp,
    np_ndarray_anyint,
    np_ndarray_float,
)

@overload
def cut(
    x: Sequence[float] | np_ndarray_anyint | np_ndarray_float | Index,
    bins: int | Index[int] | Index[float] | Sequence[float] | Series,
    right: bool = ...,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[np_1darray_intp, np_1darray_float]: ...
@overload
def cut(
    x: Sequence[float] | np_ndarray_anyint | np_ndarray_float | Index,
    bins: IntervalIndex[IntervalT],
    right: bool = ...,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[np_1darray_intp, IntervalIndex[IntervalT]]: ...
@overload
def cut(  # pyright: ignore[reportOverlappingOverload]
    x: Series[Timestamp],
    bins: int | Series[Timestamp] | DatetimeIndex | Sequence[np.datetime64 | Timestamp],
    right: bool = ...,
    labels: Literal[False] | Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Series, DatetimeIndex]: ...
@overload
def cut(
    x: Series[Timestamp],
    bins: IntervalIndex[Interval[Timestamp]],
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Series, DatetimeIndex]: ...
@overload
def cut(
    x: Series,
    bins: int | Index[int] | Index[float] | Sequence[float] | Series,
    right: bool = ...,
    labels: Literal[False] | Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Series, np_1darray_float]: ...
@overload
def cut(
    x: Series,
    bins: IntervalIndex[Interval[int]] | IntervalIndex[Interval[float]],
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Series, IntervalIndex]: ...
@overload
def cut(
    x: Sequence[float] | np_ndarray_anyint | np_ndarray_float | Index,
    bins: int | Index[int] | Index[float] | Sequence[float] | Series,
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Categorical, np_1darray_float]: ...
@overload
def cut(
    x: Sequence[float] | np_ndarray_anyint | np_ndarray_float | Index,
    bins: IntervalIndex[IntervalT],
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Categorical, IntervalIndex[IntervalT]]: ...
@overload
def cut(
    x: Sequence[float] | np_ndarray_anyint | np_ndarray_float | Index,
    bins: int | Sequence[float] | Index[int] | Index[float] | IntervalIndex | Series,
    right: bool = ...,
    *,
    labels: Literal[False],
    retbins: Literal[False] = False,
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> np_1darray_intp: ...
@overload
def cut(
    x: Series[Timestamp],
    bins: (
        int
        | Sequence[np.datetime64 | Timestamp]
        | IntervalIndex[Interval[Timestamp]]
        | DatetimeIndex
        | Series[Timestamp]
    ),
    right: bool = ...,
    labels: Literal[False] | Sequence[Label] | None = ...,
    retbins: Literal[False] = False,
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> Series[CategoricalDtype]: ...
@overload
def cut(
    x: Series,
    bins: int | Sequence[float] | Index[int] | Index[float] | IntervalIndex | Series,
    right: bool = ...,
    labels: Literal[False] | Sequence[Label] | None = ...,
    retbins: Literal[False] = False,
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> Series: ...
@overload
def cut(
    x: Sequence[float] | np_ndarray_anyint | np_ndarray_float | Index,
    bins: int | Sequence[float] | Index[int] | Index[float] | IntervalIndex | Series,
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    retbins: Literal[False] = False,
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> Categorical: ...
@overload
def qcut(
    x: Sequence[float] | np_ndarray_anyint | np_ndarray_float | Index,
    q: int | Sequence[float] | np_ndarray_float | Index[float] | Series[float],
    labels: Literal[False],
    retbins: Literal[False] = False,
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = "raise",
) -> np_1darray_intp: ...
@overload
def qcut(
    x: Sequence[float] | np_ndarray_anyint | np_ndarray_float | Index,
    q: int | Sequence[float] | np_ndarray_float | Index[float] | Series[float],
    labels: Sequence[Label] | None = None,
    retbins: Literal[False] = False,
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = "raise",
) -> Categorical: ...
@overload
def qcut(
    x: Series,
    q: int | Sequence[float] | np_ndarray_float | Index[float] | Series[float],
    labels: Literal[False] | Sequence[Label] | None = ...,
    retbins: Literal[False] = False,
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = "raise",
) -> Series: ...
@overload
def qcut(
    x: (
        Sequence[float]
        | np_ndarray_anyint
        | np_ndarray_float
        | Index[int]
        | Index[float]
    ),
    q: int | Sequence[float] | np_ndarray_float | Index[float] | Series[float],
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = "raise",
    # float when there are nan's
) -> tuple[np_1darray_intp | np_1darray_float, np_1darray_float]: ...
@overload
def qcut(
    x: Series[int] | Series[float],
    q: int | Sequence[float] | np_ndarray_float | Index[float] | Series[float],
    labels: Literal[False] | Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = "raise",
) -> tuple[Series, np_1darray_float]: ...
@overload
def qcut(
    x: (
        Sequence[float]
        | np_ndarray_anyint
        | np_ndarray_float
        | Index[int]
        | Index[float]
    ),
    q: int | Sequence[float] | np_ndarray_float | Index[float] | Series[float],
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = "raise",
) -> tuple[Categorical, np_1darray_float]: ...
