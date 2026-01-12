# pyright: strict
from datetime import (
    date,
    datetime,
    timedelta,
)
from typing import (
    ClassVar,
    Literal,
    NamedTuple,
    TypeAlias,
    overload,
)

import numpy as np
from numpy import typing as npt
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.tslibs import NaTType
from pandas._libs.tslibs.period import Period
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    Frequency,
    Just,
    ShapeT,
    TimeUnit,
    np_1darray_bool,
    np_ndarray,
)

class Components(NamedTuple):
    days: int
    hours: int
    minutes: int
    seconds: int
    milliseconds: int
    microseconds: int
    nanoseconds: int

# This should be kept consistent with the keys in the dict timedelta_abbrevs
# in pandas/_libs/tslibs/timedeltas.pyx
TimeDeltaUnitChoices: TypeAlias = Literal[
    "W",
    "w",
    "D",
    "d",
    "days",
    "day",
    "hours",
    "hour",
    "hr",
    "h",
    "m",
    "minute",
    "min",
    "minutes",
    "s",
    "seconds",
    "sec",
    "second",
    "ms",
    "milliseconds",
    "millisecond",
    "milli",
    "millis",
    "us",
    "microseconds",
    "microsecond",
    "µs",
    "micro",
    "micros",
    "ns",
    "nanoseconds",
    "nano",
    "nanos",
    "nanosecond",
]

UnitChoices: TypeAlias = (
    TimeDeltaUnitChoices
    | Literal[
        "Y",
        "y",
        "M",
    ]
)

class Timedelta(timedelta):
    min: ClassVar[Timedelta]  # pyright: ignore[reportIncompatibleVariableOverride]
    max: ClassVar[Timedelta]  # pyright: ignore[reportIncompatibleVariableOverride]
    resolution: ClassVar[  # pyright: ignore[reportIncompatibleVariableOverride]
        Timedelta
    ]
    value: int
    def __new__(
        cls,
        value: str | float | Timedelta | timedelta | np.timedelta64 = ...,
        unit: TimeDeltaUnitChoices = ...,
        *,
        days: float | np.integer | np.floating = ...,
        seconds: float | np.integer | np.floating = ...,
        microseconds: float | np.integer | np.floating = ...,
        milliseconds: float | np.integer | np.floating = ...,
        minutes: float | np.integer | np.floating = ...,
        hours: float | np.integer | np.floating = ...,
        weeks: float | np.integer | np.floating = ...,
    ) -> Self: ...
    # GH 46171
    # While Timedelta can return pd.NaT, having the constructor return
    # a Union with NaTType makes things awkward for users of pandas
    @property
    def days(self) -> int: ...
    @property
    def nanoseconds(self) -> int: ...
    @property
    def seconds(self) -> int: ...
    @property
    def microseconds(self) -> int: ...
    def total_seconds(self) -> float: ...
    def to_pytimedelta(self) -> timedelta: ...
    def to_timedelta64(self) -> np.timedelta64: ...
    @property
    def asm8(self) -> np.timedelta64: ...
    # TODO: pandas-dev/pandas-stubs#1432 round/floor/ceil could return NaT?
    def round(self, freq: Frequency) -> Self: ...
    def floor(self, freq: Frequency) -> Self: ...
    def ceil(self, freq: Frequency) -> Self: ...
    @property
    def resolution_string(self) -> str: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    def __add__(self, other: datetime | np.datetime64) -> Timestamp: ...
    @overload
    def __add__(self, other: timedelta | np.timedelta64) -> Self: ...
    @overload
    def __add__(self, other: NaTType) -> NaTType: ...
    @overload
    def __add__(self, other: Period) -> Period: ...
    @overload
    def __add__(self, other: date) -> date: ...
    @overload
    def __add__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    @overload
    def __add__(
        self, other: np_ndarray[ShapeT, np.datetime64]
    ) -> np_ndarray[ShapeT, np.datetime64]: ...
    @overload
    def __radd__(self, other: datetime | np.datetime64) -> Timestamp: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: timedelta | np.timedelta64) -> Self: ...
    @overload
    def __radd__(self, other: NaTType) -> NaTType: ...
    @overload
    def __radd__(self, other: date) -> date: ...
    @overload
    def __radd__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    @overload
    def __radd__(
        self, other: np_ndarray[ShapeT, np.datetime64]
    ) -> np_ndarray[ShapeT, np.datetime64]: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    def __sub__(self, other: timedelta | np.timedelta64 | Self) -> Self: ...
    @overload
    def __sub__(self, other: NaTType) -> NaTType: ...
    @overload
    def __sub__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    @overload
    def __sub__(self, other: TimedeltaIndex) -> TimedeltaIndex: ...
    @overload
    def __rsub__(self, other: timedelta | np.timedelta64 | Self) -> Self: ...
    @overload
    def __rsub__(self, other: datetime | Timestamp | np.datetime64) -> Timestamp: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: NaTType) -> NaTType: ...
    @overload
    def __rsub__(self, other: Period) -> Period: ...
    @overload
    def __rsub__(self, other: PeriodIndex) -> PeriodIndex: ...
    @overload
    def __rsub__(self, other: DatetimeIndex) -> DatetimeIndex: ...
    @overload
    def __rsub__(
        self, other: np_ndarray[ShapeT, np.datetime64]
    ) -> np_ndarray[ShapeT, np.datetime64]: ...
    @overload
    def __rsub__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    @overload
    def __rsub__(self, other: TimedeltaIndex) -> TimedeltaIndex: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    def __mul__(self, other: Just[float] | Just[int]) -> Self: ...
    @overload
    def __mul__(
        self, other: np_ndarray[ShapeT, np.bool_ | np.integer | np.floating]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    @overload
    def __rmul__(self, other: Just[float] | Just[int]) -> Self: ...
    @overload
    def __rmul__(
        self, other: np_ndarray[ShapeT, np.bool_ | np.integer | np.floating]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    # Override due to more types supported than timedelta
    # error: Signature of "__floordiv__" incompatible with supertype "timedelta"
    @overload  # type: ignore[override]
    def __floordiv__(self, other: timedelta | np.timedelta64 | Self) -> int: ...
    @overload
    def __floordiv__(self, other: float) -> Self: ...
    @overload
    def __floordiv__(
        self, other: np_ndarray[ShapeT, np.integer | np.floating]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    @overload
    def __floordiv__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.int_]: ...
    @overload
    def __floordiv__(self, other: Index[int] | Index[float]) -> TimedeltaIndex: ...
    @overload
    def __floordiv__(self, other: Series[int]) -> Series[Timedelta]: ...
    @overload
    def __floordiv__(self, other: Series[float]) -> Series[Timedelta]: ...
    @overload
    def __floordiv__(self, other: Series[Timedelta]) -> Series[int]: ...
    @overload
    def __floordiv__(self, other: NaTType | None) -> float: ...
    @overload
    def __rfloordiv__(self, other: timedelta | Timedelta | str) -> int: ...
    @overload
    def __rfloordiv__(self, other: NaTType | None) -> float: ...
    @overload
    def __rfloordiv__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.int_]: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    # pyrefly: ignore[bad-override]
    def __truediv__(self, other: Just[int] | Just[float]) -> Self: ...
    @overload
    def __truediv__(self, other: Self | NaTType) -> float: ...
    @overload
    def __truediv__(
        self, other: np_ndarray[ShapeT, np.integer | np.floating]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    @overload
    def __truediv__(  # ty: ignore[invalid-method-override]
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.floating]: ...
    @overload
    def __rtruediv__(self, other: Self | NaTType) -> float: ...
    @overload
    def __rtruediv__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.floating]: ...
    # Override due to more types supported than timedelta
    @overload
    def __eq__(self, other: timedelta | np.timedelta64 | Self) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __eq__(self, other: Series[Timedelta]) -> Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: Index) -> np_1darray_bool: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(  # type: ignore[overload-overlap]
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.bool_]: ...
    @overload
    def __eq__(self, other: object) -> Literal[False]: ...
    # Override due to more types supported than timedelta
    @overload
    def __ne__(self, other: timedelta | np.timedelta64 | Self) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ne__(self, other: Series[Timedelta]) -> Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: Index) -> np_1darray_bool: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(  # type: ignore[overload-overlap]
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.bool_]: ...
    @overload
    def __ne__(self, other: object) -> Literal[True]: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    def __mod__(self, other: timedelta) -> Self: ...
    @overload
    def __mod__(self, other: float) -> Self: ...
    @overload
    def __mod__(self, other: Series[int] | Series[float]) -> Series[Timedelta]: ...
    @overload
    def __mod__(self, other: Index[int] | Index[float]) -> TimedeltaIndex: ...
    @overload
    def __mod__(
        self, other: np_ndarray[ShapeT, np.integer | np.floating]
    ) -> np_ndarray[ShapeT, np.timedelta64]: ...
    @overload
    def __mod__(
        self, other: Series[int] | Series[float] | Series[Timedelta]
    ) -> Series[Timedelta]: ...
    def __divmod__(self, other: timedelta) -> tuple[int, Timedelta]: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    def __le__(self, other: timedelta | np.timedelta64 | Self) -> bool: ...
    @overload
    def __le__(self, other: TimedeltaIndex) -> np_1darray_bool: ...
    @overload
    def __le__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.bool_]: ...
    @overload
    def __le__(self, other: Series[Timedelta]) -> Series[bool]: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    def __lt__(self, other: timedelta | np.timedelta64 | Self) -> bool: ...
    @overload
    def __lt__(self, other: TimedeltaIndex) -> np_1darray_bool: ...
    @overload
    def __lt__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.bool_]: ...
    @overload
    def __lt__(self, other: Series[Timedelta]) -> Series[bool]: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    def __ge__(self, other: timedelta | np.timedelta64 | Self) -> bool: ...
    @overload
    def __ge__(self, other: TimedeltaIndex) -> np_1darray_bool: ...
    @overload
    def __ge__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.bool_]: ...
    @overload
    def __ge__(self, other: Series[Timedelta]) -> Series[bool]: ...
    # Override due to more types supported than timedelta
    @overload  # type: ignore[override]
    def __gt__(self, other: timedelta | np.timedelta64 | Self) -> bool: ...
    @overload
    def __gt__(self, other: TimedeltaIndex) -> np_1darray_bool: ...
    @overload
    def __gt__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.bool_]: ...
    @overload
    def __gt__(self, other: Series[Timedelta]) -> Series[bool]: ...
    def __hash__(self) -> int: ...
    def isoformat(self) -> str: ...
    def to_numpy(self) -> np.timedelta64: ...
    @property
    def components(self) -> Components: ...
    def view(self, dtype: npt.DTypeLike = ...) -> object: ...
    @property
    def unit(self) -> TimeUnit: ...
    def as_unit(self, unit: TimeUnit, round_ok: bool = True) -> Self: ...
