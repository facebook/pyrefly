import datetime
import sys
from typing import (
    Literal,
    TypeAlias,
    overload,
)

import numpy as np
from pandas.core.indexes.base import Index
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.tslibs import NaTType
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    PeriodFrequency,
    ShapeT,
    np_1darray_bool,
    np_ndarray,
)

class IncompatibleFrequency(ValueError): ...

_PeriodAddSub: TypeAlias = (
    Timedelta | datetime.timedelta | np.timedelta64 | np.int64 | int | BaseOffset
)

_PeriodFreqHow: TypeAlias = Literal[
    "S",
    "E",
    "start",
    "end",
]

_PeriodToTimestampHow: TypeAlias = (
    _PeriodFreqHow
    | Literal[
        "Start",
        "Finish",
        "Begin",
        "End",
        "s",
        "e",
        "finish",
        "begin",
    ]
)

class PeriodMixin:
    @property
    def end_time(self) -> Timestamp: ...
    @property
    def start_time(self) -> Timestamp: ...

class Period(PeriodMixin):
    def __init__(
        self,
        value: (
            Period | str | datetime.datetime | datetime.date | Timestamp | None
        ) = ...,
        freq: PeriodFrequency | None = None,
        ordinal: int | None = ...,
        year: int | None = ...,
        month: int | None = ...,
        quarter: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
    ) -> None: ...
    @overload
    def __sub__(self, other: _PeriodAddSub) -> Period: ...
    @overload
    def __sub__(self, other: Self) -> BaseOffset: ...
    @overload
    def __sub__(self, other: NaTType) -> NaTType: ...
    @overload
    def __sub__(self, other: PeriodIndex) -> Index: ...
    @overload
    def __sub__(
        self, other: Series[Timedelta]
    ) -> Series[Period]: ...  # pyrefly: ignore[bad-specialization]
    @overload
    def __sub__(self, other: TimedeltaIndex) -> PeriodIndex: ...
    @overload
    def __add__(self, other: _PeriodAddSub) -> Self: ...
    @overload
    def __add__(self, other: NaTType) -> NaTType: ...
    # Ignored due to indecipherable error from mypy:
    # Forward operator "__add__" is not callable  [misc]
    if sys.version_info >= (3, 11):
        @overload
        def __radd__(self, other: _PeriodAddSub) -> Self: ...
    else:
        @overload
        def __radd__(self, other: _PeriodAddSub) -> Self: ...  # type: ignore[misc]

    @overload
    def __radd__(self, other: NaTType) -> NaTType: ...
    #  ignore[misc] here because we know all other comparisons
    #  are False, so we use Literal[False]
    @overload
    def __eq__(self, other: Self) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __eq__(self, other: Index) -> np_1darray_bool: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: Series[Period]) -> Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: np_ndarray[ShapeT, np.object_]) -> np_ndarray[ShapeT, np.bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: object) -> Literal[False]: ...
    @overload
    def __ge__(self, other: Self) -> bool: ...
    @overload
    def __ge__(self, other: PeriodIndex) -> np_1darray_bool: ...
    @overload
    def __ge__(
        self, other: Series[Period]  # pyrefly: ignore[bad-specialization]
    ) -> Series[bool]: ...
    @overload
    def __ge__(
        self, other: np_ndarray[ShapeT, np.object_]
    ) -> np_ndarray[ShapeT, np.bool]: ...
    @overload
    def __gt__(self, other: Self) -> bool: ...
    @overload
    def __gt__(self, other: PeriodIndex) -> np_1darray_bool: ...
    @overload
    def __gt__(
        self, other: Series[Period]  # pyrefly: ignore[bad-specialization]
    ) -> Series[bool]: ...
    @overload
    def __gt__(
        self, other: np_ndarray[ShapeT, np.object_]
    ) -> np_ndarray[ShapeT, np.bool]: ...
    @overload
    def __le__(self, other: Self) -> bool: ...
    @overload
    def __le__(self, other: PeriodIndex) -> np_1darray_bool: ...
    @overload
    def __le__(
        self, other: Series[Period]  # pyrefly: ignore[bad-specialization]
    ) -> Series[bool]: ...
    @overload
    def __le__(
        self, other: np_ndarray[ShapeT, np.object_]
    ) -> np_ndarray[ShapeT, np.bool]: ...
    @overload
    def __lt__(self, other: Self) -> bool: ...
    @overload
    def __lt__(self, other: PeriodIndex) -> np_1darray_bool: ...
    @overload
    def __lt__(
        self, other: Series[Period]  # pyrefly: ignore[bad-specialization]
    ) -> Series[bool]: ...
    @overload
    def __lt__(
        self, other: np_ndarray[ShapeT, np.object_]
    ) -> np_ndarray[ShapeT, np.bool]: ...
    #  ignore[misc] here because we know all other comparisons
    #  are False, so we use Literal[False]
    @overload
    def __ne__(self, other: Self) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ne__(self, other: Index) -> np_1darray_bool: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: Series[Period]) -> Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: np_ndarray[ShapeT, np.object_]) -> np_ndarray[ShapeT, np.bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: object) -> Literal[True]: ...
    @property
    def day(self) -> int: ...
    @property
    def dayofweek(self) -> int: ...
    @property
    def dayofyear(self) -> int: ...
    @property
    def daysinmonth(self) -> int: ...
    @property
    def days_in_month(self) -> int: ...
    @property
    def end_time(self) -> Timestamp: ...
    @property
    def freq(self) -> BaseOffset: ...
    @property
    def freqstr(self) -> str: ...
    @property
    def hour(self) -> int: ...
    @property
    def minute(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def quarter(self) -> int: ...
    @property
    def qyear(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def ordinal(self) -> int: ...
    @property
    def is_leap_year(self) -> bool: ...
    @property
    def start_time(self) -> Timestamp: ...
    @property
    def week(self) -> int: ...
    @property
    def weekday(self) -> int: ...
    @property
    def weekofyear(self) -> int: ...
    @property
    def year(self) -> int: ...
    @property
    def day_of_year(self) -> int: ...
    @property
    def day_of_week(self) -> int: ...
    def asfreq(self, freq: PeriodFrequency, how: _PeriodFreqHow = "end") -> Period: ...
    @classmethod
    def now(cls, freq: PeriodFrequency | None = None) -> Period: ...
    def strftime(self, fmt: str) -> str: ...
    def to_timestamp(
        self,
        freq: PeriodFrequency | None = None,
        how: _PeriodToTimestampHow = "S",
    ) -> Timestamp: ...
