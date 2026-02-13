from collections.abc import (
    Callable,
    Hashable,
    Mapping,
)
from typing import (
    Any,
    Literal,
    TypeAlias,
    final,
    overload,
)

import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.core.groupby.grouper import Grouper
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._typing import (
    S1,
    Axis,
    InterpolateOptions,
    NDFrameT,
    Scalar,
    TimeGrouperOrigin,
    TimestampConvention,
    np_ndarray_float,
)

_FrameGroupByFunc: TypeAlias = (
    Callable[[DataFrame], Scalar]
    | Callable[[DataFrame], Series]
    | Callable[[DataFrame], DataFrame]
    | np.ufunc
)
_FrameGroupByFuncTypes: TypeAlias = (
    _FrameGroupByFunc | str | list[_FrameGroupByFunc | str]
)
_FrameGroupByFuncArgs: TypeAlias = (
    _FrameGroupByFuncTypes | Mapping[Hashable, _FrameGroupByFuncTypes]
)

_SeriesGroupByFunc: TypeAlias = (
    Callable[[Series], Scalar] | Callable[[Series], Series] | np.ufunc
)
_SeriesGroupByFuncTypes: TypeAlias = (
    _SeriesGroupByFunc | str | list[_SeriesGroupByFunc | str]
)
_SeriesGroupByFuncArgs: TypeAlias = (
    _SeriesGroupByFuncTypes | Mapping[Hashable, _SeriesGroupByFunc | str]
)

class Resampler(BaseGroupBy[NDFrameT]):
    def __getattr__(self, attr: str) -> SeriesGroupBy: ...
    @overload
    def aggregate(
        self: Resampler[DataFrame],
        func: _FrameGroupByFuncArgs | None = ...,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self: Resampler[Series],
        func: _SeriesGroupByFuncArgs | None = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series | DataFrame: ...
    agg = aggregate
    apply = aggregate
    @overload
    def transform(
        self: Resampler[Series],
        arg: Callable[[Series], Series[S1]],
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @overload
    def transform(
        self: Resampler[DataFrame],
        arg: Callable[[Series], Series[S1]],
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @final
    def ffill(self, limit: int | None = ...) -> NDFrameT: ...
    @final
    def nearest(self, limit: int | None = ...) -> NDFrameT: ...
    @final
    def bfill(self, limit: int | None = ...) -> NDFrameT: ...
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: Axis = ...,
        limit: int | None = ...,
        inplace: Literal[False] = False,
        limit_direction: Literal["forward", "backward", "both"] = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        **kwargs: Any,
    ) -> NDFrameT: ...
    @final
    def asfreq(self, fill_value: Scalar | None = ...) -> NDFrameT: ...
    @final
    def sum(self, numeric_only: bool = False, min_count: int = 0) -> NDFrameT: ...
    @final
    def prod(self, numeric_only: bool = False, min_count: int = 0) -> NDFrameT: ...
    @final
    def min(self, numeric_only: bool = ..., min_count: int = ...) -> NDFrameT: ...
    @final
    def max(self, numeric_only: bool = ..., min_count: int = ...) -> NDFrameT: ...
    @final
    def first(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True,
    ) -> NDFrameT: ...
    @final
    def last(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True,
    ) -> NDFrameT: ...
    @final
    def median(self, numeric_only: bool = False) -> NDFrameT: ...
    @final
    def mean(self, numeric_only: bool = False) -> NDFrameT: ...
    @final
    def std(self, ddof: int = 1, numeric_only: bool = False) -> NDFrameT: ...
    @final
    def var(self, ddof: int = 1, numeric_only: bool = False) -> NDFrameT: ...
    @final
    def sem(self, ddof: int = 1, numeric_only: bool = False) -> NDFrameT: ...
    @final
    def ohlc(self) -> DataFrame: ...
    @overload
    def nunique(self: Resampler[Series]) -> Series[int]: ...
    @overload
    def nunique(self: Resampler[DataFrame]) -> DataFrame: ...
    @final
    def size(self) -> Series[int]: ...
    @overload
    def count(self: Resampler[Series]) -> Series[int]: ...
    @overload
    def count(self: Resampler[DataFrame]) -> DataFrame: ...
    @final
    def quantile(
        self,
        q: float | list[float] | np_ndarray_float | Series[float] = 0.5,
        **kwargs: Any,
    ) -> NDFrameT: ...

# We lie about inheriting from Resampler because at runtime inherits all Resampler
# attributes via setattr
class _GroupByMixin(Resampler[NDFrameT]):
    key: str | list[str] | None
    def __getitem__(self, key: str | list[str] | None) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class DatetimeIndexResampler(Resampler[NDFrameT]): ...

class DatetimeIndexResamplerGroupby(
    _GroupByMixin[NDFrameT], DatetimeIndexResampler[NDFrameT]
):
    @final
    def __getattr__(self, attr: str) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class PeriodIndexResampler(DatetimeIndexResampler[NDFrameT]): ...

class PeriodIndexResamplerGroupby(
    _GroupByMixin[NDFrameT], PeriodIndexResampler[NDFrameT]
):
    @final
    def __getattr__(self, attr: str) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class TimedeltaIndexResampler(DatetimeIndexResampler[NDFrameT]): ...

class TimedeltaIndexResamplerGroupby(
    _GroupByMixin[NDFrameT], TimedeltaIndexResampler[NDFrameT]
):
    @final
    def __getattr__(self, attr: str) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class TimeGrouper(Grouper):
    closed: Literal["left", "right"]
    label: Literal["left", "right"]
    kind: str | None
    convention: TimestampConvention
    how: str
    fill_method: str | None
    limit: int | None
    group_keys: bool
    origin: TimeGrouperOrigin
    offset: Timedelta | None
