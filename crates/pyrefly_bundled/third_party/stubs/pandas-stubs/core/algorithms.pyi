from typing import (
    Any,
    Literal,
    overload,
)

from numpy import typing as npt
from pandas.api.extensions import ExtensionArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.indexes.base import Index
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series

from pandas._typing import (
    T_EXTENSION_ARRAY,
    AnyArrayLike,
    GenericT,
    IntervalT,
    TakeIndexer,
    np_1darray,
    np_1darray_dt,
    np_1darray_int64,
    np_1darray_td,
    np_ndarray,
)

# These are type: ignored because the Index types overlap due to inheritance but indices
# with extension types return the same type while standard type return ndarray
@overload
def unique(values: CategoricalIndex) -> CategoricalIndex: ...
@overload
def unique(values: IntervalIndex[IntervalT]) -> IntervalIndex[IntervalT]: ...
@overload
def unique(values: PeriodIndex) -> PeriodIndex: ...
@overload
# switch to DatetimeIndex after Pandas 3.0
def unique(values: DatetimeIndex) -> np_1darray_dt | DatetimeIndex: ...
@overload
# switch to TimedeltaIndex after Pandas 3.0
def unique(values: TimedeltaIndex) -> np_1darray_td: ...
@overload
# switch to Index[int] after Pandas 3.0
def unique(values: RangeIndex) -> np_1darray_int64: ...
@overload
def unique(values: MultiIndex) -> np_ndarray: ...
@overload
def unique(values: Index) -> np_1darray | Index: ...  # switch to Index after Pandas 3.0
@overload
def unique(values: Categorical) -> Categorical: ...

# @overload
# def unique(values: Series[Never]) -> np_1darray | ExtensionArray: ...
# TODO: DatetimeArray python/mypy#19952
# @overload
# def unique(values: Series[Timestamp]) -> np_ndarray_dt | ExtensionArray: ...
# @overload
# def unique(values: Series[int]) -> np_1darray_anyint | ExtensionArray: ...
@overload
def unique(values: Series) -> np_1darray | ExtensionArray: ...
@overload
def unique(values: npt.NDArray[GenericT]) -> np_1darray[GenericT]: ...
@overload
def unique(values: T_EXTENSION_ARRAY) -> T_EXTENSION_ARRAY: ...
@overload
def factorize(
    values: npt.NDArray[GenericT],
    sort: bool = ...,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np_1darray_int64, np_1darray[GenericT]]: ...
@overload
def factorize(
    values: Index | Series,
    sort: bool = ...,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np_1darray_int64, Index]: ...
@overload
def factorize(
    values: Categorical,
    sort: bool = ...,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np_1darray_int64, Categorical]: ...
def value_counts(
    values: AnyArrayLike | list | tuple,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins: int | None = None,
    dropna: bool = True,
) -> Series: ...
def take(
    arr: np_ndarray[Any] | ExtensionArray | Index | Series,
    indices: TakeIndexer,
    axis: Literal[0, 1] = 0,
    allow_fill: bool = False,
    fill_value: Any = None,
) -> np_1darray | ExtensionArray: ...
