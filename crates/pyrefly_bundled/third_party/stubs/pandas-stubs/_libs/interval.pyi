from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    overload,
    type_check_only,
)

from pandas import (
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
)

from pandas._typing import (
    IntervalClosedType,
    IntervalT,
    np_1darray_bool,
    np_ndarray,
)

VALID_CLOSED: frozenset[str]

_OrderableScalarT = TypeVar("_OrderableScalarT", bound=int | float)
_OrderableTimesT = TypeVar("_OrderableTimesT", bound=Timestamp | Timedelta)
_OrderableT = TypeVar("_OrderableT", bound=int | float | Timestamp | Timedelta)

@type_check_only
class _LengthDescriptor:
    @overload
    def __get__(
        self, instance: Interval[_OrderableScalarT], owner: Any
    ) -> _OrderableScalarT: ...
    @overload
    def __get__(
        self, instance: Interval[_OrderableTimesT], owner: Any
    ) -> Timedelta: ...
    @overload
    def __get__(self, instance: IntervalMixin, owner: Any) -> np_ndarray: ...

@type_check_only
class _MidDescriptor:
    @overload
    def __get__(self, instance: Interval[_OrderableScalarT], owner: Any) -> float: ...
    @overload
    def __get__(
        self, instance: Interval[_OrderableTimesT], owner: Any
    ) -> _OrderableTimesT: ...
    @overload
    def __get__(self, instance: IntervalMixin, owner: Any) -> np_ndarray: ...

class IntervalMixin:
    @property
    def closed_left(self) -> bool: ...
    @property
    def closed_right(self) -> bool: ...
    @property
    def open_left(self) -> bool: ...
    @property
    def open_right(self) -> bool: ...
    @property
    def is_empty(self) -> bool: ...

class Interval(IntervalMixin, Generic[_OrderableT]):
    @property
    def left(self: Interval[_OrderableT]) -> _OrderableT: ...
    @property
    def right(self: Interval[_OrderableT]) -> _OrderableT: ...
    @property
    def closed(self) -> IntervalClosedType: ...
    mid = _MidDescriptor()
    length = _LengthDescriptor()
    def __init__(
        self,
        left: _OrderableT,
        right: _OrderableT,
        closed: IntervalClosedType = ...,
    ) -> None: ...
    def __hash__(self) -> int: ...
    # for __contains__, it seems that we have to separate out the 4 cases to make
    # mypy happy
    @overload
    def __contains__(self: Interval[Timestamp], key: Timestamp) -> bool: ...
    @overload
    def __contains__(self: Interval[Timedelta], key: Timedelta) -> bool: ...
    @overload
    def __contains__(self: Interval[int], key: float) -> bool: ...
    @overload
    def __contains__(self: Interval[float], key: float) -> bool: ...
    @overload
    def __add__(self: Interval[Timestamp], y: Timedelta) -> Interval[Timestamp]: ...
    @overload
    def __add__(self: Interval[Timedelta], y: Timedelta) -> Interval[Timedelta]: ...
    @overload
    def __add__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __add__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __radd__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __radd__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __radd__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __sub__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __sub__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __sub__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __rsub__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __rsub__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __rsub__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __mul__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __mul__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __mul__(self: Interval[Timedelta], y: float) -> Interval[Timedelta]: ...
    @overload
    def __rmul__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __rmul__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __rmul__(self: Interval[Timedelta], y: float) -> Interval[Timedelta]: ...
    @overload
    def __truediv__(self: Interval[int], y: float) -> Interval[float]: ...
    @overload
    def __truediv__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __truediv__(self: Interval[Timedelta], y: float) -> Interval[Timedelta]: ...
    @overload
    def __floordiv__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __floordiv__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __floordiv__(self: Interval[Timedelta], y: float) -> Interval[Timedelta]: ...
    @overload
    def overlaps(self: Interval[_OrderableT], other: Interval[_OrderableT]) -> bool: ...
    @overload
    def overlaps(self: Interval[int], other: Interval[float]) -> bool: ...
    @overload
    def overlaps(self: Interval[float], other: Interval[int]) -> bool: ...
    @overload
    def __gt__(self, other: Interval[_OrderableT]) -> bool: ...
    @overload
    def __gt__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_1darray_bool: ...
    @overload
    def __gt__(
        self,
        other: Series[int] | Series[float] | Series[Timestamp] | Series[Timedelta],
    ) -> Series[bool]: ...
    @overload
    def __lt__(self, other: Interval[_OrderableT]) -> bool: ...
    @overload
    def __lt__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_1darray_bool: ...
    @overload
    def __lt__(
        self,
        other: Series[int] | Series[float] | Series[Timestamp] | Series[Timedelta],
    ) -> Series[bool]: ...
    @overload
    def __ge__(self, other: Interval[_OrderableT]) -> bool: ...
    @overload
    def __ge__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_1darray_bool: ...
    @overload
    def __ge__(
        self,
        other: Series[int] | Series[float] | Series[Timestamp] | Series[Timedelta],
    ) -> Series[bool]: ...
    @overload
    def __le__(self, other: Interval[_OrderableT]) -> bool: ...
    @overload
    def __le__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_1darray_bool: ...
    @overload
    def __eq__(self, other: Interval[_OrderableT]) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __eq__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_1darray_bool: ...
    @overload
    def __eq__(self, other: Series[_OrderableT]) -> Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: object) -> Literal[False]: ...
    @overload
    def __ne__(self, other: Interval[_OrderableT]) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ne__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_1darray_bool: ...
    @overload
    def __ne__(self, other: Series[_OrderableT]) -> Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: object) -> Literal[True]: ...
