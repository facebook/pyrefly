from typing import (
    Any,
    final,
    overload,
)

from pandas.core.resample import TimeGrouper
from typing_extensions import Self

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    Axis,
    Frequency,
    Incomplete,
    KeysArgType,
    Level,
    ListLikeHashable,
)

class Grouper:
    key: KeysArgType | None
    level: Level | ListLikeHashable[Level] | None
    freq: Frequency | None
    axis: Axis
    sort: bool
    dropna: bool
    binner: Incomplete
    @overload
    def __new__(
        cls,
        key: KeysArgType | None = ...,
        level: Level | ListLikeHashable[Level] | None = ...,
        axis: Axis | _NoDefaultDoNotUse = ...,
        sort: bool = ...,
        dropna: bool = ...,
    ) -> Self: ...
    @overload
    def __new__(cls, *args: Any, freq: Frequency, **kwargs: Any) -> TimeGrouper: ...
    @final
    def __repr__(self) -> str: ...  # noqa: PYI029 __repr__ here is final
