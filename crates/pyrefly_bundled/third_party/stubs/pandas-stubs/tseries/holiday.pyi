from collections.abc import Callable
from datetime import (
    date as _date,
    datetime,
)
from typing import (
    Literal,
    overload,
)

import numpy as np
from pandas import (
    DatetimeIndex,
    Series,
)

from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.timestamps import Timestamp

def next_monday(dt: datetime) -> datetime: ...
def next_monday_or_tuesday(dt: datetime) -> datetime: ...
def previous_friday(dt: datetime) -> datetime: ...
def sunday_to_monday(dt: datetime) -> datetime: ...
def weekend_to_monday(dt: datetime) -> datetime: ...
def nearest_workday(dt: datetime) -> datetime: ...
def next_workday(dt: datetime) -> datetime: ...
def previous_workday(dt: datetime) -> datetime: ...
def before_nearest_workday(dt: datetime) -> datetime: ...
def after_nearest_workday(dt: datetime) -> datetime: ...

class Holiday:
    def __init__(
        self,
        name: str,
        year: int | None = ...,
        month: int | None = ...,
        day: int | None = ...,
        offset: BaseOffset | list[BaseOffset] | None = ...,
        observance: Callable[[datetime], datetime] | None = ...,
        # Values accepted by Timestamp(), or None:
        start_date: (
            np.integer | float | str | _date | datetime | np.datetime64 | None
        ) = ...,
        end_date: (
            np.integer | float | str | _date | datetime | np.datetime64 | None
        ) = ...,
        days_of_week: tuple[int, ...] | None = ...,
    ) -> None: ...
    @overload
    def dates(
        self,
        start_date: np.integer | float | str | _date | datetime | np.datetime64 | None,
        end_date: np.integer | float | str | _date | datetime | np.datetime64 | None,
        return_name: Literal[False],
    ) -> DatetimeIndex: ...
    @overload
    def dates(
        self,
        start_date: np.integer | float | str | _date | datetime | np.datetime64 | None,
        end_date: np.integer | float | str | _date | datetime | np.datetime64 | None,
        return_name: Literal[True] = True,
    ) -> Series: ...

holiday_calendars: dict[str, type[AbstractHolidayCalendar]]

def register(cls: type[AbstractHolidayCalendar]) -> None: ...
def get_calendar(name: str) -> AbstractHolidayCalendar: ...

class AbstractHolidayCalendar:
    rules: list[Holiday]
    start_date: Timestamp
    end_date: Timestamp

    def __init__(self, name: str = "", rules: list[Holiday] | None = None) -> None: ...
    def rule_from_name(self, name: str) -> Holiday | None: ...
    @overload
    def holidays(
        self,
        start: datetime | None = ...,
        end: datetime | None = ...,
        *,
        return_name: Literal[True],
    ) -> Series: ...
    @overload
    def holidays(
        self,
        start: datetime | None = ...,
        end: datetime | None = ...,
        return_name: Literal[False] = False,
    ) -> DatetimeIndex: ...
    @staticmethod
    def merge_class(
        base: AbstractHolidayCalendar | type[AbstractHolidayCalendar] | list[Holiday],
        other: AbstractHolidayCalendar | type[AbstractHolidayCalendar] | list[Holiday],
    ) -> list[Holiday]: ...
    @overload
    def merge(
        self,
        other: AbstractHolidayCalendar | type[AbstractHolidayCalendar],
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def merge(
        self,
        other: AbstractHolidayCalendar | type[AbstractHolidayCalendar],
        inplace: Literal[False] = False,
    ) -> list[Holiday]: ...

USMemorialDay: Holiday
USLaborDay: Holiday
USColumbusDay: Holiday
USThanksgivingDay: Holiday
USMartinLutherKingJr: Holiday
USPresidentsDay: Holiday
GoodFriday: Holiday
EasterMonday: Holiday

class USFederalHolidayCalendar(AbstractHolidayCalendar): ...

def HolidayCalendarFactory(
    name: str,
    base: type[AbstractHolidayCalendar],
    other: type[AbstractHolidayCalendar],
    base_class: type[AbstractHolidayCalendar] = ...,
) -> type[AbstractHolidayCalendar]: ...
