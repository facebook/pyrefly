from typing import overload

from pandas import (
    DatetimeIndex,
    Series,
    TimedeltaIndex,
)

from pandas._typing import Frequency

from pandas.tseries.offsets import BaseOffset

@overload
def to_offset(freq: None, is_period: bool = ...) -> None: ...
@overload
def to_offset(freq: Frequency, is_period: bool = ...) -> BaseOffset: ...
def infer_freq(index: Series | DatetimeIndex | TimedeltaIndex) -> str | None: ...
