from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
import datetime as dt
from typing import (
    Any,
    Concatenate,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    final,
    overload,
)

import numpy as np
from pandas.core.base import SelectionMixin
from pandas.core.frame import DataFrame
from pandas.core.groupby import generic
from pandas.core.groupby.indexing import (
    GroupByIndexingMixin,
    GroupByNthSelector,
)
from pandas.core.indexers import BaseIndexer
from pandas.core.indexes.api import Index
from pandas.core.resample import (
    DatetimeIndexResamplerGroupby,
    PeriodIndexResamplerGroupby,
    TimedeltaIndexResamplerGroupby,
)
from pandas.core.series import Series
from pandas.core.window import (
    ExpandingGroupby,
    ExponentialMovingWindowGroupby,
    RollingGroupby,
)
from typing_extensions import Self

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._libs.tslibs import BaseOffset
from pandas._typing import (
    S1,
    AnyArrayLike,
    Axis,
    AxisInt,
    CalculationMethod,
    Dtype,
    Frequency,
    IndexLabel,
    IntervalClosedType,
    MaskType,
    NDFrameT,
    P,
    RandomState,
    Scalar,
    T,
    TimedeltaConvertibleTypes,
    TimeGrouperOrigin,
    TimestampConvention,
    TimestampConvertibleTypes,
    WindowingEngine,
    WindowingEngineKwargs,
    np_ndarray_dt,
    np_ndarray_int64,
)

from pandas.plotting import PlotAccessor

_ResamplerGroupBy: TypeAlias = (
    DatetimeIndexResamplerGroupby[NDFrameT]  # ty: ignore[invalid-argument-type]
    | PeriodIndexResamplerGroupby[NDFrameT]  # ty: ignore[invalid-argument-type]
    | TimedeltaIndexResamplerGroupby[NDFrameT]  # ty: ignore[invalid-argument-type]
)

class GroupBy(BaseGroupBy[NDFrameT]):
    def __getattr__(self, attr: str) -> Any: ...
    def apply(self, func: Callable | str, *args: Any, **kwargs: Any) -> NDFrameT: ...
    @final
    @overload
    def any(self: GroupBy[Series], skipna: bool = ...) -> Series[bool]: ...
    @overload
    def any(self: GroupBy[DataFrame], skipna: bool = ...) -> DataFrame: ...
    @final
    @overload
    def all(self: GroupBy[Series], skipna: bool = ...) -> Series[bool]: ...
    @overload
    def all(self: GroupBy[DataFrame], skipna: bool = ...) -> DataFrame: ...
    @final
    @overload
    def count(self: GroupBy[Series]) -> Series[int]: ...
    @overload
    def count(self: GroupBy[DataFrame]) -> DataFrame: ...
    @final
    def mean(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    @final
    def median(self, numeric_only: bool = False) -> NDFrameT: ...
    @final
    @overload
    def std(
        self: GroupBy[Series],
        ddof: int = ...,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        numeric_only: bool = ...,
    ) -> Series[float]: ...
    @overload
    def std(
        self: GroupBy[DataFrame],
        ddof: int = ...,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    @final
    @overload
    def var(
        self: GroupBy[Series],
        ddof: int = ...,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        numeric_only: bool = ...,
    ) -> Series[float]: ...
    @overload
    def var(
        self: GroupBy[DataFrame],
        ddof: int = ...,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    @final
    @overload
    def sem(
        self: GroupBy[Series], ddof: int = ..., numeric_only: bool = ...
    ) -> Series[float]: ...
    @overload
    def sem(
        self: GroupBy[DataFrame], ddof: int = ..., numeric_only: bool = ...
    ) -> DataFrame: ...
    def size(self: GroupBy[Series]) -> Series[int]: ...
    @final
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    @final
    def prod(self, numeric_only: bool = False, min_count: int = 0) -> NDFrameT: ...
    @final
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    @final
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    @final
    def first(
        self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True
    ) -> NDFrameT: ...
    @final
    def last(
        self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True
    ) -> NDFrameT: ...
    @final
    def ohlc(self) -> DataFrame: ...
    def describe(
        self,
        percentiles: Iterable[float] | None = ...,
        include: Literal["all"] | list[Dtype] | None = ...,
        exclude: list[Dtype] | None = ...,
    ) -> DataFrame: ...
    @final
    def resample(
        self,
        rule: Frequency | dt.timedelta,
        how: str | None = ...,
        fill_method: str | None = ...,
        limit: int | None = ...,
        kind: str | None = ...,
        on: Hashable | None = ...,
        *,
        closed: Literal["left", "right"] | None = ...,
        label: Literal["left", "right"] | None = ...,
        axis: Axis = ...,
        convention: TimestampConvention | None = ...,
        origin: TimeGrouperOrigin | TimestampConvertibleTypes = ...,
        offset: TimedeltaConvertibleTypes | None = ...,
        group_keys: bool = ...,
        **kwargs: Any,
    ) -> _ResamplerGroupBy[NDFrameT]: ...
    @final
    def rolling(
        self,
        window: int | dt.timedelta | str | BaseOffset | BaseIndexer | None = ...,
        min_periods: int | None = None,
        center: bool | None = False,
        win_type: str | None = None,
        axis: Axis = 0,
        on: str | Index | None = None,
        closed: IntervalClosedType | None = None,
        method: CalculationMethod = "single",
        *,
        selection: IndexLabel | None = None,
    ) -> RollingGroupby[NDFrameT]: ...
    @final
    def expanding(
        self,
        min_periods: int = ...,
        axis: Axis = ...,
        method: CalculationMethod = ...,
        selection: IndexLabel | None = ...,
    ) -> ExpandingGroupby[NDFrameT]: ...
    @final
    def ewm(
        self,
        com: float | None = ...,
        span: float | None = ...,
        halflife: TimedeltaConvertibleTypes | None = ...,
        alpha: float | None = ...,
        min_periods: int | None = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        axis: Axis = ...,
        times: str | np_ndarray_dt | Series | np.timedelta64 | None = ...,
        method: CalculationMethod = ...,
        *,
        selection: IndexLabel | None = ...,
    ) -> ExponentialMovingWindowGroupby[NDFrameT]: ...
    @final
    def ffill(self, limit: int | None = ...) -> NDFrameT: ...
    @final
    def bfill(self, limit: int | None = ...) -> NDFrameT: ...
    @final
    @property
    def nth(
        self,
    ) -> GroupByNthSelector[Self]: ...  # pyrefly: ignore[bad-specialization]
    @final
    def quantile(
        self,
        q: float | AnyArrayLike = 0.5,
        interpolation: str = "linear",
        numeric_only: bool = False,
    ) -> NDFrameT: ...
    @final
    def ngroup(self, ascending: bool = True) -> Series[int]: ...
    @final
    def cumcount(self, ascending: bool = True) -> Series[int]: ...
    @final
    def rank(
        self,
        method: str = "average",
        ascending: bool = True,
        na_option: str = "keep",
        pct: bool = False,
        axis: AxisInt | _NoDefaultDoNotUse = 0,
    ) -> NDFrameT: ...
    @final
    def cumprod(
        self, axis: Axis | _NoDefaultDoNotUse = ..., *args: Any, **kwargs: Any
    ) -> NDFrameT: ...
    @final
    def cumsum(
        self, axis: Axis | _NoDefaultDoNotUse = ..., *args: Any, **kwargs: Any
    ) -> NDFrameT: ...
    @final
    def cummin(
        self,
        axis: AxisInt | _NoDefaultDoNotUse = ...,
        numeric_only: bool = ...,
        **kwargs: Any,
    ) -> NDFrameT: ...
    @final
    def cummax(
        self,
        axis: AxisInt | _NoDefaultDoNotUse = ...,
        numeric_only: bool = ...,
        **kwargs: Any,
    ) -> NDFrameT: ...
    @final
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq: Frequency | None = ...,
        axis: Axis | _NoDefaultDoNotUse = 0,
        fill_value: Scalar | None = None,
        suffix: str | None = ...,
    ) -> NDFrameT: ...
    @final
    def diff(
        self, periods: int = 1, axis: AxisInt | _NoDefaultDoNotUse = 0
    ) -> NDFrameT: ...
    @final
    def pct_change(
        self,
        periods: int = ...,
        fill_method: Literal["bfill", "ffill"] | None | _NoDefaultDoNotUse = ...,
        limit: int | None | _NoDefaultDoNotUse = ...,
        freq: Frequency | None = None,
        axis: Axis | _NoDefaultDoNotUse = ...,
    ) -> NDFrameT: ...
    @final
    def head(self, n: int = ...) -> NDFrameT: ...
    @final
    def tail(self, n: int = ...) -> NDFrameT: ...
    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: Sequence | Series | None = ...,
        random_state: RandomState | None = ...,
    ) -> NDFrameT: ...

_GroupByT = TypeVar("_GroupByT", bound=GroupBy)

# GroupByPlot does not really inherit from PlotAccessor but it delegates
# to it using __call__ and __getattr__. We lie here to avoid repeating the
# whole stub of PlotAccessor
@final
class GroupByPlot(PlotAccessor, Generic[_GroupByT]):
    def __init__(self, groupby: _GroupByT) -> None: ...
    # The following methods are inherited from the fake parent class PlotAccessor
    # def __call__(self, *args: Any, **kwargs: Any): ...
    # def __getattr__(self, name: str): ...

class BaseGroupBy(SelectionMixin[NDFrameT], GroupByIndexingMixin):
    @final
    def __len__(self) -> int: ...
    @final
    def __repr__(self) -> str: ...  # noqa: PYI029 __repr__ here is final
    @final
    @property
    def groups(self) -> dict[Hashable, Index]: ...
    @final
    @property
    def ngroups(self) -> int: ...
    @final
    @property
    def indices(self) -> dict[Hashable, Index | np_ndarray_int64 | list[int]]: ...
    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
    @final
    def get_group(self, name: Hashable) -> NDFrameT: ...
    @final
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]: ...
    @overload
    def __getitem__(self: BaseGroupBy[DataFrame], key: Scalar) -> generic.SeriesGroupBy: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __getitem__(
        self: BaseGroupBy[DataFrame], key: Iterable[Hashable]
    ) -> generic.DataFrameGroupBy: ...
    @overload
    def __getitem__(
        self: BaseGroupBy[Series[S1]],
        idx: list[str] | Index | Series[S1] | MaskType | tuple[Hashable | slice, ...],
    ) -> generic.SeriesGroupBy: ...
    @overload
    def __getitem__(self: BaseGroupBy[Series[S1]], idx: Scalar) -> S1: ...
