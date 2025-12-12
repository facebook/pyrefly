# pyright: strict
from datetime import (
    date as _date,
    datetime,
    time as _time,
    timedelta,
    tzinfo as _tzinfo,
)
from datetime import _IsoCalendarDate  # pyright: ignore[reportPrivateUsage]
import sys
from time import struct_time
from typing import (
    ClassVar,
    Literal,
    SupportsIndex,
    TypeAlias,
    overload,
)

import numpy as np
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import (
    Never,
    Self,
)

from pandas._libs.tslibs import (
    Period,
    Tick,
    Timedelta,
)
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    PeriodFrequency,
    ShapeT,
    TimestampNonexistent,
    TimeUnit,
    np_1darray_bool,
    np_ndarray,
)

_Ambiguous: TypeAlias = bool | Literal["raise", "NaT"]

# Repeated from `_typing.pyi` so as to satisfy mixed strict / non-strict paths.
# https://github.com/pandas-dev/pandas-stubs/pull/1151#issuecomment-2715130190
TimeZones: TypeAlias = str | _tzinfo | None | int

class Timestamp(datetime, SupportsIndex):
    min: ClassVar[Timestamp]  # pyright: ignore[reportIncompatibleVariableOverride]
    max: ClassVar[Timestamp]  # pyright: ignore[reportIncompatibleVariableOverride]

    resolution: ClassVar[  # pyright: ignore[reportIncompatibleVariableOverride]
        Timedelta
    ]
    value: int
    def __new__(
        cls,
        ts_input: np.integer | float | str | _date | datetime | np.datetime64 = ...,
        year: int | None = ...,
        month: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
        microsecond: int | None = ...,
        tzinfo: _tzinfo | None = ...,
        *,
        nanosecond: int | None = ...,
        tz: TimeZones = ...,
        unit: str | int | None = ...,
        fold: Literal[0, 1] | None = ...,
    ) -> Self: ...
    # GH 46171
    # While Timestamp can return pd.NaT, having the constructor return
    # a Union with NaTType makes things awkward for users of pandas
    @property
    def year(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def day(self) -> int: ...
    @property
    def hour(self) -> int: ...
    @property
    def minute(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def microsecond(self) -> int: ...
    @property
    def nanosecond(self) -> int: ...
    @property
    def tzinfo(self) -> _tzinfo | None: ...
    @property
    def tz(self) -> _tzinfo | None: ...
    @property
    def fold(self) -> int: ...
    if sys.version_info >= (3, 12):
        @classmethod
        def fromtimestamp(  # pyright: ignore[reportIncompatibleMethodOverride]
            cls, t: float, tz: _tzinfo | str | None = ...
        ) -> Self: ...
    else:
        @classmethod
        def fromtimestamp(cls, t: float, tz: _tzinfo | str | None = ...) -> Self: ...

    @classmethod
    def utcfromtimestamp(cls, ts: float) -> Self: ...
    @classmethod
    def today(cls, tz: _tzinfo | str | None = None) -> Self: ...
    @classmethod
    def fromordinal(
        cls,
        ordinal: int,
        tz: _tzinfo | str | None = ...,
    ) -> Self: ...
    @classmethod
    def now(cls, tz: _tzinfo | str | None = None) -> Self: ...
    @classmethod
    def utcnow(cls) -> Self: ...
    # error: Signature of "combine" incompatible with supertype "datetime"
    @classmethod
    def combine(cls, date: _date, time: _time) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    @classmethod
    def fromisoformat(cls, date_string: str) -> Self: ...
    def strftime(self, format: str) -> str: ...
    def __format__(self, fmt: str) -> str: ...
    def toordinal(self) -> int: ...
    def timetuple(self) -> struct_time: ...
    def timestamp(self) -> float: ...
    def utctimetuple(self) -> struct_time: ...
    def date(self) -> _date: ...
    def time(self) -> _time: ...
    def timetz(self) -> _time: ...
    # Override since fold is more precise than datetime.replace(fold:int)
    # Here it is restricted to be 0 or 1 using a Literal
    # Violation of Liskov substitution principle
    def replace(  # type:ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore
        self,
        year: int | None = ...,
        month: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
        microsecond: int | None = ...,
        tzinfo: _tzinfo | None = ...,
        fold: Literal[0, 1] | None = ...,
    ) -> Timestamp: ...
    def astimezone(self, tz: _tzinfo | None = ...) -> Self: ...
    def ctime(self) -> str: ...
    def isoformat(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        sep: str = "T",
        timespec: Literal[
            "auto",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ] = "auto",
    ) -> str: ...
    @classmethod
    def strptime(cls, date_string: Never, format: Never) -> Never: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def utcoffset(self) -> timedelta | None: ...
    def tzname(self) -> str | None: ...
    def dst(self) -> timedelta | None: ...
    # Mypy complains Forward operator "<inequality op>" is not callable, so ignore misc
    # for le, lt ge and gt
    @overload  # type: ignore[override]
    def __le__(self, other: datetime | np.datetime64 | Self) -> bool: ...
    @overload
    def __le__(self, other: DatetimeIndex) -> np_1darray_bool: ...
    @overload
    def __le__(
        self, other: np_ndarray[ShapeT, np.datetime64]
    ) -> np_ndarray[ShapeT, np.bool]: ...
    @overload
    def __le__(  # ty: ignore[invalid-method-override]
        self, other: Series[Timestamp]
    ) -> Series[bool]: ...
    @overload  # type: ignore[override]
    def __lt__(self, other: datetime | np.datetime64 | Self) -> bool: ...
    @overload
    def __lt__(self, other: DatetimeIndex) -> np_1darray_bool: ...
    @overload
    def __lt__(
        self, other: np_ndarray[ShapeT, np.datetime64]
    ) -> np_ndarray[ShapeT, np.bool]: ...
    @overload
    def __lt__(  # ty: ignore[invalid-method-override]
        self, other: Series[Timestamp]
    ) -> Series[bool]: ...
    @overload  # type: ignore[override]
    def __ge__(self, other: datetime | np.datetime64 | Self) -> bool: ...
    @overload
    def __ge__(self, other: DatetimeIndex) -> np_1darray_bool: ...
    @overload
    def __ge__(
        self, other: np_ndarray[ShapeT, np.datetime64]
    ) -> np_ndarray[ShapeT, np.bool]: ...
    @overload
    def __ge__(  # ty: ignore[invalid-method-override]
        self, other: Series[Timestamp]
    ) -> Series[bool]: ...
    @overload  # type: ignore[override]
    def __gt__(self, other: datetime | np.datetime64 | Self) -> bool: ...
    @overload
    def __gt__(self, other: DatetimeIndex) -> np_1darray_bool: ...
    @overload
    def __gt__(
        self, other: np_ndarray[ShapeT, np.datetime64]
    ) -> np_ndarray[ShapeT, np.bool]: ...
    @overload
    def __gt__(  # ty: ignore[invalid-method-override]
        self, other: Series[Timestamp]
    ) -> Series[bool]: ...
    # error: Signature of "__add__" incompatible with supertype "date"/"datetime"
    @overload  # type: ignore[override]
    def __add__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.datetime64]: ...
    @overload
    def __add__(self, other: timedelta | np.timedelta64 | Tick) -> Self: ...
    @overload
    def __add__(self, other: TimedeltaIndex) -> DatetimeIndex: ...
    @overload
    def __radd__(self, other: timedelta) -> Self: ...
    @overload
    def __radd__(self, other: TimedeltaIndex) -> DatetimeIndex: ...
    @overload
    def __radd__(
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.datetime64]: ...
    def __rsub__(self, other: datetime | np.datetime64) -> Timedelta: ...
    @overload  # type: ignore[override]
    def __sub__(self, other: datetime | np.datetime64) -> Timedelta: ...
    @overload
    def __sub__(self, other: timedelta | np.timedelta64 | Tick) -> Self: ...
    @overload
    def __sub__(self, other: TimedeltaIndex) -> DatetimeIndex: ...
    @overload
    def __sub__(  # ty: ignore[invalid-method-override]
        self, other: np_ndarray[ShapeT, np.timedelta64]
    ) -> np_ndarray[ShapeT, np.datetime64]: ...
    @overload
    def __eq__(self, other: datetime | np.datetime64 | Self) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __eq__(self, other: Series[Timestamp]) -> Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: Index) -> np_1darray_bool: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: np_ndarray[ShapeT, np.datetime64]) -> np_ndarray[ShapeT, np.bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: object) -> Literal[False]: ...
    @overload
    def __ne__(self, other: datetime | np.datetime64 | Self) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ne__(self, other: Series[Timestamp]) -> Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: Index) -> np_1darray_bool: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: np_ndarray[ShapeT, np.datetime64]) -> np_ndarray[ShapeT, np.bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: object) -> Literal[True]: ...
    def __hash__(self) -> int: ...
    def weekday(self) -> int: ...
    def isoweekday(self) -> int: ...
    def isocalendar(self) -> _IsoCalendarDate: ...
    @property
    def is_leap_year(self) -> bool: ...
    @property
    def is_month_start(self) -> bool: ...
    @property
    def is_quarter_start(self) -> bool: ...
    @property
    def is_year_start(self) -> bool: ...
    @property
    def is_month_end(self) -> bool: ...
    @property
    def is_quarter_end(self) -> bool: ...
    @property
    def is_year_end(self) -> bool: ...
    def to_pydatetime(self, warn: bool = ...) -> datetime: ...
    def to_datetime64(self) -> np.datetime64: ...
    def to_period(self, freq: PeriodFrequency | None = ...) -> Period: ...
    def to_julian_date(self) -> np.float64: ...
    @property
    def asm8(self) -> np.datetime64: ...
    def tz_convert(self, tz: TimeZones) -> Self: ...
    @overload
    def tz_localize(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        self,
        tz: TimeZones,
        ambiguous: _Ambiguous = "raise",
        *,
        nonexistent: Literal["NaT"],
    ) -> Self | NaTType: ...
    @overload
    def tz_localize(
        self,
        tz: TimeZones,
        ambiguous: _Ambiguous = "raise",
        nonexistent: TimestampNonexistent = "raise",
    ) -> Self: ...
    def normalize(self) -> Self: ...
    # TODO: pandas-dev/pandas-stubs#1432 round/floor/ceil could return NaT?
    def round(
        self,
        freq: str,
        ambiguous: _Ambiguous = "raise",
        nonexistent: TimestampNonexistent = "raise",
    ) -> Self: ...
    def floor(
        self,
        freq: str,
        ambiguous: _Ambiguous = "raise",
        nonexistent: TimestampNonexistent = "raise",
    ) -> Self: ...
    def ceil(
        self,
        freq: str,
        ambiguous: _Ambiguous = "raise",
        nonexistent: TimestampNonexistent = "raise",
    ) -> Self: ...
    def day_name(self, locale: str | None = None) -> str: ...
    def month_name(self, locale: str | None = None) -> str: ...
    @property
    def day_of_week(self) -> int: ...
    @property
    def dayofweek(self) -> int: ...
    @property
    def day_of_year(self) -> int: ...
    @property
    def dayofyear(self) -> int: ...
    @property
    def weekofyear(self) -> int: ...
    @property
    def quarter(self) -> int: ...
    @property
    def week(self) -> int: ...
    def to_numpy(self) -> np.datetime64: ...
    @property
    def days_in_month(self) -> int: ...
    @property
    def daysinmonth(self) -> int: ...
    @property
    def unit(self) -> TimeUnit: ...
    def as_unit(self, unit: TimeUnit, round_ok: bool = True) -> Self: ...
    # To support slicing
    def __index__(self) -> int: ...
