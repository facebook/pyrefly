from datetime import (
    date,
    time,
    timedelta,
    tzinfo as _tzinfo,
)
from typing import (
    Generic,
    Literal,
    TypeVar,
    overload,
    type_check_only,
)

from pandas.core.accessor import PandasDelegate
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.arrays.period import PeriodArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.base import (
    IndexOpsMixin,
    NoNewAttributesMixin,
)
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import Never

from pandas._libs.interval import Interval
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.period import Period
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    Frequency,
    PeriodFrequency,
    TimeAmbiguous,
    TimeNonexistent,
    TimestampConvention,
    TimeUnit,
    TimeZones,
    np_1darray_bool,
    np_1darray_object,
    np_ndarray_bool,
)

from pandas.core.dtypes.dtypes import CategoricalDtype

class Properties(PandasDelegate, NoNewAttributesMixin): ...

_DTFieldOpsReturnType = TypeVar("_DTFieldOpsReturnType", bound=Series[int] | Index[int])

class _DayLikeFieldOps(Generic[_DTFieldOpsReturnType]):
    @property
    def year(self) -> _DTFieldOpsReturnType: ...
    @property
    def month(self) -> _DTFieldOpsReturnType: ...
    @property
    def day(self) -> _DTFieldOpsReturnType: ...
    @property
    def hour(self) -> _DTFieldOpsReturnType: ...
    @property
    def minute(self) -> _DTFieldOpsReturnType: ...
    @property
    def second(self) -> _DTFieldOpsReturnType: ...
    @property
    def weekday(self) -> _DTFieldOpsReturnType: ...
    @property
    def dayofweek(self) -> _DTFieldOpsReturnType: ...
    @property
    def day_of_week(self) -> _DTFieldOpsReturnType: ...
    @property
    def dayofyear(self) -> _DTFieldOpsReturnType: ...
    @property
    def day_of_year(self) -> _DTFieldOpsReturnType: ...
    @property
    def quarter(self) -> _DTFieldOpsReturnType: ...
    @property
    def days_in_month(self) -> _DTFieldOpsReturnType: ...
    @property
    def daysinmonth(self) -> _DTFieldOpsReturnType: ...

class _MiniSeconds(Generic[_DTFieldOpsReturnType]):
    @property
    def microsecond(self) -> _DTFieldOpsReturnType: ...
    @property
    def nanosecond(self) -> _DTFieldOpsReturnType: ...

class _DatetimeFieldOps(
    _DayLikeFieldOps[_DTFieldOpsReturnType], _MiniSeconds[_DTFieldOpsReturnType]
): ...

_DTBoolOpsReturnType = TypeVar(
    "_DTBoolOpsReturnType", bound=Series[bool] | np_1darray_bool
)

class _IsLeapYearProperty(Generic[_DTBoolOpsReturnType]):
    @property
    def is_leap_year(self) -> _DTBoolOpsReturnType: ...

class _DatetimeBoolOps(
    _IsLeapYearProperty[_DTBoolOpsReturnType], Generic[_DTBoolOpsReturnType]
):
    @property
    def is_month_start(self) -> _DTBoolOpsReturnType: ...
    @property
    def is_month_end(self) -> _DTBoolOpsReturnType: ...
    @property
    def is_quarter_start(self) -> _DTBoolOpsReturnType: ...
    @property
    def is_quarter_end(self) -> _DTBoolOpsReturnType: ...
    @property
    def is_year_start(self) -> _DTBoolOpsReturnType: ...
    @property
    def is_year_end(self) -> _DTBoolOpsReturnType: ...

_DTFreqReturnType = TypeVar("_DTFreqReturnType", bound=str | BaseOffset)

class _FreqProperty(Generic[_DTFreqReturnType]):
    @property
    def freq(self) -> _DTFreqReturnType | None: ...

class _TZProperty:
    @property
    def tz(self) -> _tzinfo | None: ...

class _DatetimeObjectOps(
    _FreqProperty[_DTFreqReturnType], _TZProperty, Generic[_DTFreqReturnType]
): ...

_DTOtherOpsDateReturnType = TypeVar(
    "_DTOtherOpsDateReturnType", bound=Series[date] | np_1darray_object
)
_DTOtherOpsTimeReturnType = TypeVar(
    "_DTOtherOpsTimeReturnType", bound=Series[time] | np_1darray_object
)

class _DatetimeOtherOps(Generic[_DTOtherOpsDateReturnType, _DTOtherOpsTimeReturnType]):
    @property
    def date(self) -> _DTOtherOpsDateReturnType: ...
    @property
    def time(self) -> _DTOtherOpsTimeReturnType: ...
    @property
    def timetz(self) -> _DTOtherOpsTimeReturnType: ...

class _DatetimeLikeOps(
    _DatetimeFieldOps[_DTFieldOpsReturnType],
    _DatetimeObjectOps[_DTFreqReturnType],
    _DatetimeBoolOps[_DTBoolOpsReturnType],
    _DatetimeOtherOps[_DTOtherOpsDateReturnType, _DTOtherOpsTimeReturnType],
    Generic[
        _DTFieldOpsReturnType,
        _DTBoolOpsReturnType,
        _DTOtherOpsDateReturnType,
        _DTOtherOpsTimeReturnType,
        _DTFreqReturnType,
    ],
): ...

# Ideally, the rounding methods would return Series[Timestamp] when `Series.dt.method`
# is invoked, but because of how Series.dt is hooked in and that we may not know the
# type of the series, we don't know which kind of series was ...ed
# in to the dt accessor

_DTTimestampTimedeltaReturnType = TypeVar(
    "_DTTimestampTimedeltaReturnType",
    bound=Series
    | Series[Timestamp]
    | Series[Timedelta]
    | DatetimeIndex
    | TimedeltaIndex,
)

class _DatetimeRoundingMethods(Generic[_DTTimestampTimedeltaReturnType]):
    def round(
        self,
        freq: Frequency | None,
        ambiguous: Literal["raise", "infer", "NaT"] | bool | np_ndarray_bool = ...,
        nonexistent: (
            Literal["shift_forward", "shift_backward", "NaT", "raise"]
            | timedelta
            | Timedelta
        ) = ...,
    ) -> _DTTimestampTimedeltaReturnType: ...
    def floor(
        self,
        freq: Frequency | None,
        ambiguous: Literal["raise", "infer", "NaT"] | bool | np_ndarray_bool = ...,
        nonexistent: (
            Literal["shift_forward", "shift_backward", "NaT", "raise"]
            | timedelta
            | Timedelta
        ) = ...,
    ) -> _DTTimestampTimedeltaReturnType: ...
    def ceil(
        self,
        freq: Frequency | None,
        ambiguous: Literal["raise", "infer", "NaT"] | bool | np_ndarray_bool = ...,
        nonexistent: (
            Literal["shift_forward", "shift_backward", "NaT", "raise"]
            | timedelta
            | Timedelta
        ) = ...,
    ) -> _DTTimestampTimedeltaReturnType: ...

_DTNormalizeReturnType = TypeVar(
    "_DTNormalizeReturnType", Series[Timestamp], DatetimeIndex
)
_DTStrKindReturnType = TypeVar("_DTStrKindReturnType", bound=Series[str] | Index)
_DTToPeriodReturnType = TypeVar(
    "_DTToPeriodReturnType", bound=Series[Period] | PeriodIndex
)

class _DatetimeLikeNoTZMethods(
    _DatetimeRoundingMethods[_DTTimestampTimedeltaReturnType],
    Generic[
        _DTTimestampTimedeltaReturnType,
        _DTNormalizeReturnType,
        _DTStrKindReturnType,
        _DTToPeriodReturnType,
    ],
):
    def to_period(
        self, freq: PeriodFrequency | None = None
    ) -> _DTToPeriodReturnType: ...
    def tz_localize(
        self,
        tz: TimeZones,
        ambiguous: TimeAmbiguous = ...,
        nonexistent: TimeNonexistent = ...,
    ) -> _DTNormalizeReturnType: ...
    def tz_convert(self, tz: TimeZones) -> _DTNormalizeReturnType: ...
    def normalize(self) -> _DTNormalizeReturnType: ...
    def strftime(self, date_format: str) -> _DTStrKindReturnType: ...
    def month_name(self, locale: str | None = ...) -> _DTStrKindReturnType: ...
    def day_name(self, locale: str | None = ...) -> _DTStrKindReturnType: ...

class _DatetimeNoTZProperties(
    _DatetimeLikeOps[
        _DTFieldOpsReturnType,
        _DTBoolOpsReturnType,
        _DTOtherOpsDateReturnType,
        _DTOtherOpsTimeReturnType,
        _DTFreqReturnType,
    ],
    _DatetimeLikeNoTZMethods[
        _DTTimestampTimedeltaReturnType,
        _DTNormalizeReturnType,
        _DTStrKindReturnType,
        _DTToPeriodReturnType,
    ],
    Generic[
        _DTFieldOpsReturnType,
        _DTBoolOpsReturnType,
        _DTTimestampTimedeltaReturnType,
        _DTOtherOpsDateReturnType,
        _DTOtherOpsTimeReturnType,
        _DTFreqReturnType,
        _DTNormalizeReturnType,
        _DTStrKindReturnType,
        _DTToPeriodReturnType,
    ],
): ...

class DatetimeProperties(
    Properties,
    _DatetimeNoTZProperties[
        _DTFieldOpsReturnType,
        _DTBoolOpsReturnType,
        _DTTimestampTimedeltaReturnType,
        _DTOtherOpsDateReturnType,
        _DTOtherOpsTimeReturnType,
        _DTFreqReturnType,
        _DTNormalizeReturnType,
        _DTStrKindReturnType,
        _DTToPeriodReturnType,
    ],
    Generic[
        _DTFieldOpsReturnType,
        _DTBoolOpsReturnType,
        _DTTimestampTimedeltaReturnType,
        _DTOtherOpsDateReturnType,
        _DTOtherOpsTimeReturnType,
        _DTFreqReturnType,
        _DTNormalizeReturnType,
        _DTStrKindReturnType,
        _DTToPeriodReturnType,
    ],
):
    def to_pydatetime(self) -> np_1darray_object: ...
    def isocalendar(self) -> DataFrame: ...
    @property
    def unit(self) -> TimeUnit: ...
    def as_unit(self, unit: TimeUnit) -> _DTTimestampTimedeltaReturnType: ...

_TDNoRoundingMethodReturnType = TypeVar(
    "_TDNoRoundingMethodReturnType", bound=Series[int] | Index
)
_TDTotalSecondsReturnType = TypeVar(
    "_TDTotalSecondsReturnType", bound=Series[float] | Index
)

class _TimedeltaPropertiesNoRounding(
    Generic[_TDNoRoundingMethodReturnType, _TDTotalSecondsReturnType]
):
    def to_pytimedelta(self) -> np_1darray_object: ...
    @property
    def components(self) -> DataFrame: ...
    @property
    def days(self) -> _TDNoRoundingMethodReturnType: ...
    @property
    def seconds(self) -> _TDNoRoundingMethodReturnType: ...
    @property
    def microseconds(self) -> _TDNoRoundingMethodReturnType: ...
    @property
    def nanoseconds(self) -> _TDNoRoundingMethodReturnType: ...
    def total_seconds(self) -> _TDTotalSecondsReturnType: ...

class TimedeltaProperties(
    Properties,
    _TimedeltaPropertiesNoRounding[Series[int], Series[float]],
    _DatetimeRoundingMethods[Series[Timedelta]],
):
    @property
    def unit(self) -> TimeUnit: ...
    def as_unit(self, unit: TimeUnit) -> Series[Timedelta]: ...

_PeriodDTReturnTypes = TypeVar(
    "_PeriodDTReturnTypes", bound=Series[Timestamp] | DatetimeIndex
)
_PeriodIntReturnTypes = TypeVar("_PeriodIntReturnTypes", bound=Series[int] | Index[int])
_PeriodStrReturnTypes = TypeVar("_PeriodStrReturnTypes", bound=Series[str] | Index)
_PeriodDTAReturnTypes = TypeVar(
    "_PeriodDTAReturnTypes", bound=DatetimeArray | DatetimeIndex
)
_PeriodPAReturnTypes = TypeVar("_PeriodPAReturnTypes", bound=PeriodArray | PeriodIndex)

class _PeriodProperties(
    Generic[
        _PeriodDTReturnTypes,
        _PeriodIntReturnTypes,
        _PeriodStrReturnTypes,
        _PeriodDTAReturnTypes,
        _PeriodPAReturnTypes,
    ]
):
    @property
    def start_time(self) -> _PeriodDTReturnTypes: ...
    @property
    def end_time(self) -> _PeriodDTReturnTypes: ...
    @property
    def qyear(self) -> _PeriodIntReturnTypes: ...
    def strftime(self, date_format: str) -> _PeriodStrReturnTypes: ...
    def to_timestamp(
        self,
        freq: PeriodFrequency | None = None,
        how: TimestampConvention = ...,
    ) -> _PeriodDTAReturnTypes: ...
    def asfreq(
        self,
        freq: PeriodFrequency | None = None,
        how: Literal["E", "END", "FINISH", "S", "START", "BEGIN"] = ...,
    ) -> _PeriodPAReturnTypes: ...

class PeriodIndexFieldOps(
    _DayLikeFieldOps[Index[int]],
    _PeriodProperties[DatetimeIndex, Index[int], Index, DatetimeIndex, PeriodIndex],
): ...
class PeriodProperties(
    Properties,
    _PeriodProperties[
        Series[Timestamp], Series[int], Series[str], DatetimeArray, PeriodArray
    ],
    _DatetimeFieldOps[Series[int]],
    _IsLeapYearProperty,
    _FreqProperty[BaseOffset],
): ...
class CombinedDatetimelikeProperties(
    DatetimeProperties[
        Series[int],
        Series[bool],
        Series,
        Series[date],
        Series[time],
        str,
        Series[Timestamp],
        Series[str],
        Series[Period],
    ],
    _TimedeltaPropertiesNoRounding[Series[int], Series[float]],
    _PeriodProperties,
): ...

@type_check_only
class TimestampProperties(
    DatetimeProperties[
        Series[int],
        Series[bool],
        Series[Timestamp],
        Series[date],
        Series[time],
        str,
        Series[Timestamp],
        Series[str],
        Series[Period],
    ]
): ...

class DatetimeIndexProperties(
    Properties,
    _DatetimeNoTZProperties[
        Index[int],
        np_1darray_bool,
        DatetimeIndex,
        np_1darray_object,
        np_1darray_object,
        BaseOffset,
        DatetimeIndex,
        Index,
        PeriodIndex,
    ],
    _TZProperty,
):
    @property
    def is_normalized(self) -> bool: ...
    @property
    def tzinfo(self) -> _tzinfo | None: ...
    def to_pydatetime(self) -> np_1darray_object: ...
    def std(
        self, axis: int | None = ..., ddof: int = ..., skipna: bool = ...
    ) -> Timedelta: ...

class TimedeltaIndexProperties(
    Properties,
    _TimedeltaPropertiesNoRounding[Index, Index],
    _DatetimeRoundingMethods[TimedeltaIndex],
): ...

@type_check_only
class DtDescriptor:
    @overload
    def __get__(self, instance: Series[Never], owner: type[Series]) -> Properties: ...
    @overload
    def __get__(
        self, instance: Series[Timestamp], owner: type[Series]
    ) -> TimestampProperties: ...
    @overload
    def __get__(
        self, instance: Series[Timedelta], owner: type[Series]
    ) -> TimedeltaProperties: ...
    @overload
    def __get__(
        self, instance: Series[Period], owner: type[Series]
    ) -> PeriodProperties: ...

@type_check_only
class ArrayDescriptor:
    @overload
    def __get__(
        self, instance: IndexOpsMixin[Never], owner: type[IndexOpsMixin]
    ) -> ExtensionArray: ...
    @overload
    def __get__(
        self, instance: IndexOpsMixin[CategoricalDtype], owner: type[IndexOpsMixin]
    ) -> Categorical: ...
    @overload
    def __get__(
        self, instance: IndexOpsMixin[Interval], owner: type[IndexOpsMixin]
    ) -> IntervalArray: ...
    @overload
    def __get__(
        self, instance: IndexOpsMixin[Timestamp], owner: type[IndexOpsMixin]
    ) -> DatetimeArray: ...
    @overload
    def __get__(
        self, instance: IndexOpsMixin[Timedelta], owner: type[IndexOpsMixin]
    ) -> TimedeltaArray: ...
    # should be NumpyExtensionArray
    @overload
    def __get__(
        self, instance: IndexOpsMixin, owner: type[IndexOpsMixin]
    ) -> ExtensionArray: ...
