from builtins import (
    bool as _bool,
    str as _str,
)
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    Sequence,
    ValuesView,
)
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from pathlib import Path
import sys
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    Protocol,
    TypeAlias,
    final,
    overload,
    type_check_only,
)

from _typeshed import (
    SupportsAdd,
    SupportsGetItem,
    SupportsMul,
    SupportsRAdd,
    SupportsRMul,
    _T_contra,
)
from matplotlib.axes import (
    Axes as PlotAxes,
    SubplotBase,
)
import numpy as np
from pandas import (
    Index,
    Period,
    PeriodDtype,
    Timedelta,
    Timestamp,
)
from pandas.core.api import (
    Int8Dtype as Int8Dtype,
    Int16Dtype as Int16Dtype,
    Int32Dtype as Int32Dtype,
    Int64Dtype as Int64Dtype,
)
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.categorical import (
    Categorical,
    CategoricalAccessor,
)
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.base import (
    T_INTERVAL_NP,
    ArrayIndexSeriesTimedeltaNoSeq,
    ArrayIndexTimedeltaNoSeq,
    ElementOpsMixin,
    IndexOpsMixin,
    NumListLike,
    ScalarArrayIndexSeriesComplex,
    ScalarArrayIndexSeriesIntNoBool,
    ScalarArrayIndexSeriesJustComplex,
    ScalarArrayIndexSeriesJustFloat,
    ScalarArrayIndexSeriesJustInt,
    ScalarArrayIndexSeriesReal,
    ScalarArrayIndexSeriesTimedelta,
    SeriesComplex,
    SeriesReal,
    Supports_ProtoAdd,
    Supports_ProtoFloorDiv,
    Supports_ProtoMul,
    Supports_ProtoRAdd,
    Supports_ProtoRFloorDiv,
    Supports_ProtoRMul,
    Supports_ProtoRTrueDiv,
    Supports_ProtoTrueDiv,
)
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.core.indexers import BaseIndexer
from pandas.core.indexes.accessors import DtDescriptor
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.indexing import (
    _AtIndexer,
    _iAtIndexer,
    _iLocIndexer,
    _IndexSliceTuple,
    _LocIndexer,
)
from pandas.core.strings.accessor import StringMethods
from pandas.core.window import (
    Expanding,
    ExponentialMovingWindow,
)
from pandas.core.window.rolling import (
    Rolling,
    Window,
)
from typing_extensions import (
    Never,
    Self,
)
import xarray as xr

from pandas._libs.interval import (
    Interval,
    _OrderableT,
)
from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._libs.missing import NAType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    S1,
    S2,
    S2_NSDT,
    T_COMPLEX,
    AggFuncTypeBase,
    AggFuncTypeDictFrame,
    AggFuncTypeSeriesToFrame,
    AnyAll,
    AnyArrayLike,
    ArrayLike,
    Axes,
    AxesData,
    Axis,
    AxisColumn,
    AxisIndex,
    BooleanDtypeArg,
    BytesDtypeArg,
    CalculationMethod,
    CategoryDtypeArg,
    ComplexDtypeArg,
    CompressionOptions,
    DropKeep,
    Dtype,
    DTypeLike,
    DtypeObj,
    FilePath,
    FillnaOptions,
    FloatDtypeArg,
    FloatFormatType,
    Frequency,
    GenericT,
    GroupByObjectNonScalar,
    HashableT1,
    IgnoreRaise,
    IndexingInt,
    IndexKeyFunc,
    IndexLabel,
    IntDtypeArg,
    InterpolateOptions,
    IntervalClosedType,
    IntervalT,
    JoinHow,
    JSONSerializable,
    JsonSeriesOrient,
    Just,
    Label,
    Level,
    ListLike,
    ListLikeU,
    MaskType,
    NaPosition,
    NsmallestNlargestKeep,
    NumpyStrDtypeArg,
    ObjectDtypeArg,
    PandasAstypeComplexDtypeArg,
    PandasAstypeFloatDtypeArg,
    PandasAstypeTimedeltaDtypeArg,
    PandasAstypeTimestampDtypeArg,
    PeriodFrequency,
    QuantileInterpolation,
    RandomState,
    ReindexMethod,
    Renamer,
    ReplaceValue,
    S2_contra,
    S2_NDT_contra,
    Scalar,
    ScalarT,
    SequenceNotStr,
    SeriesByT,
    SortKind,
    StrDtypeArg,
    StrLike,
    Suffixes,
    SupportsDType,
    T as _T,
    TimeAmbiguous,
    TimedeltaDtypeArg,
    TimestampDtypeArg,
    TimeUnit,
    TimeZones,
    ToTimestampHow,
    UIntDtypeArg,
    ValueKeyFunc,
    VoidDtypeArg,
    WriteBuffer,
    _T_co,
    np_1darray,
    np_1darray_anyint,
    np_1darray_bool,
    np_1darray_bytes,
    np_1darray_complex,
    np_1darray_dt,
    np_1darray_float,
    np_1darray_int64,
    np_1darray_intp,
    np_1darray_object,
    np_1darray_str,
    np_1darray_td,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_dt,
    np_ndarray_float,
    np_ndarray_num,
    np_ndarray_str,
    np_ndarray_td,
)

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.plotting import PlotAccessor

@type_check_only
class _SupportsAdd(Protocol[_T_co]):
    def __add__(self, value: Self, /) -> _T_co: ...

@type_check_only
class SupportsSelfSub(Protocol[_T_co]):
    def __sub__(self, x: Self, /) -> _T_co: ...

@type_check_only
class _SupportsMul(Protocol[_T_co]):
    def __mul__(self, value: Self, /) -> _T_co: ...

@type_check_only
class SupportsTruedivInt(Protocol[_T_co]):
    def __truediv__(self, value: int, /) -> _T_co: ...

class _iLocIndexerSeries(_iLocIndexer, Generic[S1]):
    # get item
    # Keep in sync with `Series.__getitem__`
    @overload
    def __getitem__(self, idx: IndexingInt) -> S1: ...
    @overload
    def __getitem__(
        self, key: Index | Series | slice | np_ndarray_anyint
    ) -> Series[S1]: ...

    # set item
    # Keep in sync with `Series.__setitem__`
    @overload
    def __setitem__(self, idx: int, value: S1 | None) -> None: ...
    @overload
    def __setitem__(
        self,
        key: Index | slice | np_ndarray_anyint | list[int],
        value: S1 | IndexOpsMixin[S1] | None,
    ) -> None: ...

class _LocIndexerSeries(_LocIndexer, Generic[S1]):
    # Keep in sync with `Series.__getitem__`
    # ignore needed because of mypy.  Overlapping, but we want to distinguish
    # having a tuple of just scalars, versus tuples that include slices or Index
    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self,
        key: Scalar | tuple[Scalar, ...],
        # tuple case is for getting a specific element when using a MultiIndex
    ) -> S1: ...
    @overload
    def __getitem__(
        self,
        idx: (
            MaskType
            | Index
            | Series
            | SequenceNotStr[float | _str | Timestamp]
            | slice
            | _IndexSliceTuple
            | Sequence[_IndexSliceTuple]
            | Callable
        ),
        # _IndexSliceTuple is when having a tuple that includes a slice.  Could just
        # be s.loc[1, :], or s.loc[pd.IndexSlice[1, :]]
    ) -> Series[S1]: ...

    # Keep in sync with `Series.__setitem__`
    @overload
    def __setitem__(
        self,
        idx: IndexOpsMixin[S1] | MaskType | slice,
        value: S1 | ArrayLike | IndexOpsMixin[S1] | None,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: _str,
        value: S1 | None,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        key: MaskType | StrLike | _IndexSliceTuple | list[ScalarT],
        value: S1 | ArrayLike | IndexOpsMixin[S1] | None,
    ) -> None: ...

_DataLike: TypeAlias = ArrayLike | dict[str, np_ndarray] | SequenceNotStr[S1]

class Series(IndexOpsMixin[S1], ElementOpsMixin[S1], NDFrame):
    # Define __index__ because mypy thinks Series follows protocol `SupportsIndex` https://github.com/pandas-dev/pandas-stubs/pull/1332#discussion_r2285648790
    __index__: ClassVar[None]
    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]

    @overload
    def __new__(
        cls,
        data: Sequence[Never],
        index: AxesData | None = None,
        dtype: None = None,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series: ...
    @overload
    def __new__(
        cls,
        data: Sequence[list[_str]],
        index: AxesData | None = None,
        dtype: None = None,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[list[_str]]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[_str],
        index: AxesData | None = None,
        dtype: Dtype | None = None,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[_str]: ...
    @overload
    def __new__(
        cls,
        data: (
            DatetimeIndex
            | Sequence[np.datetime64 | datetime | date]
            | dict[HashableT1, np.datetime64 | datetime | date]
            | np.datetime64
            | datetime
            | date
        ),
        index: AxesData | None = None,
        dtype: TimestampDtypeArg = ...,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[Timestamp]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[datetime | np.timedelta64] | np_ndarray_dt | DatetimeArray,
        index: AxesData | None = None,
        *,
        dtype: TimestampDtypeArg,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[Timestamp]: ...
    @overload
    def __new__(
        cls,
        data: _DataLike,
        index: AxesData | None = None,
        *,
        dtype: CategoryDtypeArg,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[CategoricalDtype]: ...
    @overload
    def __new__(
        cls,
        data: PeriodIndex | Sequence[Period],
        index: AxesData | None = None,
        dtype: PeriodDtype = ...,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[Period]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[BaseOffset],
        index: AxesData | None = None,
        dtype: PeriodDtype = ...,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[BaseOffset]: ...
    @overload
    def __new__(
        cls,
        data: (
            TimedeltaIndex
            | Sequence[np.timedelta64 | timedelta]
            | dict[HashableT1, np.timedelta64 | timedelta]
            | np.timedelta64
            | timedelta
        ),
        index: AxesData | None = None,
        dtype: TimedeltaDtypeArg = ...,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[Timedelta]: ...
    @overload
    def __new__(
        cls,
        data: (
            IntervalIndex[Interval[_OrderableT]]
            | Interval[_OrderableT]
            | Sequence[Interval[_OrderableT]]
            | dict[HashableT1, Interval[_OrderableT]]
        ),
        index: AxesData | None = None,
        dtype: Literal["Interval"] = ...,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[Interval[_OrderableT]]: ...
    @overload
    def __new__(  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Scalar | _DataLike | dict[HashableT1, Any] | None,
        index: AxesData | None = None,
        *,
        dtype: type[S1],
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Self: ...
    @overload
    def __new__(  # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[bool | np.bool],
        index: AxesData | None = None,
        dtype: BooleanDtypeArg | None = None,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[bool]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[int | np.integer],
        index: AxesData | None = None,
        dtype: IntDtypeArg | UIntDtypeArg | None = None,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[int]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[float | np.floating] | np_ndarray_float | FloatingArray,
        index: AxesData | None = None,
        dtype: None = None,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[float]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        index: None = None,
        *,
        dtype: FloatDtypeArg,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[float]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        index: AxesData,
        dtype: FloatDtypeArg,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series[float]: ...
    @overload
    def __new__(
        cls,
        data: (
            S1
            | ArrayLike
            | dict[_str, np_ndarray]
            | Sequence[S1]
            | IndexOpsMixin[S1]
            | dict[HashableT1, S1]
            | KeysView[S1]
            | ValuesView[S1]
        ),
        index: AxesData | None = None,
        dtype: Dtype | None = None,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: (
            Scalar
            | _DataLike
            | Mapping[HashableT1, Any]
            | BaseGroupBy
            | NaTType
            | NAType
            | None
        ) = None,
        index: AxesData | None = None,
        dtype: Dtype | None = None,
        name: Hashable = None,
        copy: bool | None = None,
    ) -> Series: ...
    @property
    def hasnans(self) -> bool: ...
    @property
    def dtype(self) -> DtypeObj: ...
    @property
    def dtypes(self) -> DtypeObj: ...
    @property
    def name(self) -> Hashable | None: ...
    @name.setter
    def name(self, value: Hashable | None) -> None: ...
    @property
    def values(self) -> np_1darray | ExtensionArray | Categorical: ...
    def ravel(self, order: _str = ...) -> np_1darray: ...
    def __len__(self) -> int: ...
    @final
    def __array_ufunc__(
        self, ufunc: Callable, method: _str, *inputs: Any, **kwargs: Any
    ) -> Any: ...
    def __array__(  # ty: ignore[invalid-method-override]
        self, dtype: _str | np.dtype = ..., copy: bool | None = ...
    ) -> np_1darray: ...
    @property
    def axes(self) -> list: ...
    @final
    def __getattr__(self, name: _str) -> S1: ...

    # Keep in sync with `_iLocIndexerSeries.__getitem__`
    @overload
    def __getitem__(self, idx: IndexingInt) -> S1: ...
    @overload
    def __getitem__(
        self, idx: Index | Series | slice | np_ndarray_anyint
    ) -> Series[S1]: ...
    # Keep in sync with `_LocIndexerSeries.__getitem__`
    @overload
    def __getitem__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        self,
        idx: Scalar | tuple[Scalar, ...],
        # tuple case is for getting a specific element when using a MultiIndex
    ) -> S1: ...
    @overload
    def __getitem__(
        self,
        idx: (
            MaskType
            | Index
            | Series
            | SequenceNotStr[float | _str | Timestamp]
            | slice
            | _IndexSliceTuple
            | Sequence[_IndexSliceTuple]
            | Callable
        ),
        # _IndexSliceTuple is when having a tuple that includes a slice.  Could just
        # be s.loc[1, :], or s.loc[pd.IndexSlice[1, :]]
    ) -> Series[S1]: ...

    # Keep in sync with `_iLocIndexerSeries.__setitem__`
    @overload
    def __setitem__(self, idx: int, value: S1 | None) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: Index | slice | np_ndarray_anyint | list[int],
        value: S1 | IndexOpsMixin[S1] | None,
    ) -> None: ...
    # Keep in sync with `_LocIndexerSeries.__setitem__`
    @overload
    def __setitem__(
        self,
        idx: Index | MaskType | slice,
        value: S1 | ArrayLike | IndexOpsMixin[S1] | None,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: _str,
        value: S1 | None,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: MaskType | StrLike | _IndexSliceTuple | list[ScalarT],
        value: S1 | ArrayLike | IndexOpsMixin[S1] | None,
    ) -> None: ...
    @overload
    def get(self, key: Hashable, default: None = None) -> S1 | None: ...
    @overload
    def get(self, key: Hashable, default: S1) -> S1: ...
    @overload
    def get(self, key: Hashable, default: _T) -> S1 | _T: ...
    def repeat(
        self, repeats: int | list[int], axis: AxisIndex | None = 0
    ) -> Series[S1]: ...
    @property
    def index(self) -> Index: ...
    @index.setter
    def index(
        self, idx: AnyArrayLike | SequenceNotStr[Hashable] | tuple[Hashable, ...]
    ) -> None: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = ...,
        *,
        drop: Literal[False] = False,
        name: Level = ...,
        inplace: Literal[False] = False,
        allow_duplicates: bool = ...,
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = ...,
        *,
        drop: Literal[True],
        name: Level = ...,
        inplace: Literal[False] = False,
        allow_duplicates: bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = ...,
        *,
        drop: bool = ...,
        name: Level = ...,
        inplace: Literal[True],
        allow_duplicates: bool = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[_str],
        na_rep: _str = ...,
        float_format: FloatFormatType = ...,
        header: _bool = ...,
        index: _bool = ...,
        length: _bool = ...,
        dtype: _bool = ...,
        name: _bool = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: None = None,
        na_rep: _str = ...,
        float_format: FloatFormatType = ...,
        header: _bool = ...,
        index: _bool = ...,
        length: _bool = ...,
        dtype: _bool = ...,
        name: _bool = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[_str],
        *,
        orient: Literal["records"],
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: Literal[True],
        compression: CompressionOptions = ...,
        index: _bool = ...,
        indent: int | None = ...,
        mode: Literal["a"],
    ) -> None: ...
    @overload
    def to_json(
        self,
        path_or_buf: None = None,
        *,
        orient: Literal["records"],
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: Literal[True],
        compression: CompressionOptions = ...,
        index: _bool = ...,
        indent: int | None = ...,
        mode: Literal["a"],
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[_str] | WriteBuffer[bytes],
        *,
        orient: JsonSeriesOrient | None = ...,
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: _bool = ...,
        compression: CompressionOptions = ...,
        index: _bool = ...,
        indent: int | None = ...,
        mode: Literal["w"] = ...,
    ) -> None: ...
    @overload
    def to_json(
        self,
        path_or_buf: None = None,
        *,
        orient: JsonSeriesOrient | None = ...,
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: _bool = ...,
        compression: CompressionOptions = ...,
        index: _bool = ...,
        indent: int | None = ...,
        mode: Literal["w"] = ...,
    ) -> _str: ...
    @final
    def to_xarray(self) -> xr.DataArray: ...
    def items(self) -> Iterator[tuple[Hashable, S1]]: ...
    def keys(self) -> Index: ...
    @overload
    def to_dict(self, *, into: type[dict] = ...) -> dict[Any, S1]: ...
    @overload
    def to_dict(
        self, *, into: type[MutableMapping] | MutableMapping
    ) -> MutableMapping[Hashable, S1]: ...
    def to_frame(self, name: object | None = ...) -> DataFrame: ...
    @overload
    def groupby(
        self,
        by: Scalar,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: DatetimeIndex,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Timestamp]: ...
    @overload
    def groupby(
        self,
        by: TimedeltaIndex,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Timedelta]: ...
    @overload
    def groupby(
        self,
        by: PeriodIndex,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Period]: ...
    @overload
    def groupby(
        self,
        by: IntervalIndex[IntervalT],
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, IntervalT]: ...
    @overload
    def groupby(
        self,
        by: MultiIndex | GroupByObjectNonScalar,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, tuple]: ...
    @overload
    def groupby(
        self,
        by: None,
        level: IndexLabel,  # level is required when by=None (passed as positional)
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: None = None,
        *,
        level: IndexLabel,  # level is required when by=None (passed as keyword)
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: Series[SeriesByT],
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, SeriesByT]: ...
    @overload
    def groupby(
        self,
        by: CategoricalIndex | Index | Series,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Any]: ...
    def count(self) -> int: ...
    def mode(self, dropna: bool = True) -> Series[S1]: ...
    @overload
    def unique(self: Series[Never]) -> np_1darray: ...  # type: ignore[overload-overlap]
    @overload
    def unique(self: Series[Timestamp]) -> DatetimeArray: ...  # type: ignore[overload-overlap]
    @overload
    def unique(self: Series[Timedelta]) -> TimedeltaArray: ...  # type: ignore[overload-overlap]
    @overload
    def unique(self) -> np_1darray: ...
    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[True],
        ignore_index: _bool = ...,
    ) -> None: ...
    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[False] = False,
        ignore_index: _bool = ...,
    ) -> Series[S1]: ...
    def duplicated(self, keep: DropKeep = "first") -> Series[_bool]: ...
    def idxmax(
        self,
        axis: AxisIndex = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> int | _str: ...
    def idxmin(
        self,
        axis: AxisIndex = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> int | _str: ...
    def round(self, decimals: int = 0, *args: Any, **kwargs: Any) -> Series[S1]: ...
    @overload
    def quantile(
        self,
        q: float = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float: ...
    @overload
    def quantile(
        self,
        q: ListLike,
        interpolation: QuantileInterpolation = ...,
    ) -> Series[S1]: ...
    def corr(
        self,
        other: Series[S1],
        method: Literal["pearson", "kendall", "spearman"] = ...,
        min_periods: int | None = ...,
    ) -> float: ...
    def cov(
        self, other: Series[S1], min_periods: int | None = None, ddof: int = 1
    ) -> float: ...
    @overload
    def diff(  # type: ignore[overload-overlap]
        self: Series[Never] | Series[int], periods: int = ...
    ) -> Series[float]: ...
    @overload
    def diff(self: Series[_bool], periods: int = ...) -> Series: ...
    @overload
    def diff(
        self: Series[BooleanDtype], periods: int = ...
    ) -> Series[BooleanDtype]: ...
    @overload
    def diff(self: Series[Interval], periods: int = ...) -> Never: ...
    @overload
    def diff(
        self: SupportsGetItem[Scalar, SupportsSelfSub[S2]], periods: int = ...
    ) -> Series[S2]: ...
    def autocorr(self, lag: int = 1) -> float: ...
    @overload
    def dot(self, other: Series[S1]) -> Scalar: ...
    @overload
    def dot(self, other: DataFrame) -> Series[S1]: ...
    @overload
    def dot(
        self,
        other: ArrayLike | dict[_str, np_ndarray_num] | Sequence[S1] | Index[S1],
    ) -> np_ndarray_num: ...
    @overload
    def __matmul__(self, other: Series) -> Scalar: ...
    @overload
    def __matmul__(self, other: DataFrame) -> Series: ...
    @overload
    def __matmul__(self, other: np_ndarray_num) -> np_ndarray_num: ...
    @overload
    def __rmatmul__(self, other: Series) -> Scalar: ...
    @overload
    def __rmatmul__(self, other: DataFrame) -> Series: ...
    @overload
    def __rmatmul__(self, other: np_ndarray_num) -> np_ndarray_num: ...
    @overload
    def searchsorted(
        self,
        value: ListLike,
        side: Literal["left", "right"] = ...,
        sorter: ListLike | None = None,
    ) -> np_1darray_intp: ...
    @overload
    def searchsorted(
        self,
        value: Scalar,
        side: Literal["left", "right"] = ...,
        sorter: ListLike | None = None,
    ) -> np.intp: ...
    @overload
    def compare(
        self,
        other: Series,
        align_axis: AxisIndex,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
        result_names: Suffixes = ...,
    ) -> Series: ...
    @overload
    def compare(
        self,
        other: Series,
        align_axis: AxisColumn = ...,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
        result_names: Suffixes = ...,
    ) -> DataFrame: ...
    def combine(
        self, other: Series[S1], func: Callable, fill_value: Scalar | None = ...
    ) -> Series[S1]: ...
    def combine_first(self, other: Series[S1]) -> Series[S1]: ...
    def update(self, other: Series[S1] | Sequence[S1] | Mapping[int, S1]) -> None: ...
    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: _bool = ...,
        inplace: Literal[True],
        key: ValueKeyFunc = ...,
    ) -> None: ...
    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: _bool = ...,
        inplace: Literal[False] = False,
        key: ValueKeyFunc = ...,
    ) -> Series[S1]: ...
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: Level | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        inplace: Literal[True],
        key: IndexKeyFunc = ...,
    ) -> None: ...
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: Level | list[int] | list[_str] | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        inplace: Literal[False] = False,
        key: IndexKeyFunc = ...,
    ) -> Series[S1]: ...
    def argsort(
        self,
        axis: AxisIndex = 0,
        kind: SortKind = "quicksort",
        order: None = None,
        stable: None = None,
    ) -> Series[int]: ...
    def nlargest(
        self, n: int = 5, keep: NsmallestNlargestKeep = "first"
    ) -> Series[S1]: ...
    def nsmallest(
        self, n: int = 5, keep: NsmallestNlargestKeep = "first"
    ) -> Series[S1]: ...
    def swaplevel(
        self, i: Level = -2, j: Level = -1, copy: _bool = True
    ) -> Series[S1]: ...
    def reorder_levels(self, order: list) -> Series[S1]: ...
    def explode(self, ignore_index: _bool = ...) -> Series[S1]: ...
    def unstack(
        self,
        level: IndexLabel = -1,
        fill_value: int | _str | dict | None = None,
        sort: _bool = True,
    ) -> DataFrame: ...
    @overload
    def map(
        self,
        arg: Callable[[S1], S2 | NAType] | Mapping[S1, S2] | Series[S2],
        na_action: Literal["ignore"] = ...,
    ) -> Series[S2]: ...
    @overload
    def map(
        self,
        arg: Callable[[S1 | NAType], S2 | NAType] | Mapping[S1, S2] | Series[S2],
        na_action: None = None,
    ) -> Series[S2]: ...
    @overload
    def map(
        self,
        arg: Callable[[Any], Any] | Mapping[Any, Any] | Series,
        na_action: Literal["ignore"] | None = ...,
    ) -> Series: ...
    @overload
    def aggregate(
        self: Series[int],
        func: Literal["mean"],
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> float: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase,
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> S1: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeSeriesToFrame = ...,
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series: ...
    agg = aggregate
    @overload
    def transform(
        self,
        func: AggFuncTypeBase,
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @overload
    def transform(
        self,
        func: list[AggFuncTypeBase] | AggFuncTypeDictFrame,
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def apply(
        self,
        func: Callable[
            ..., Scalar | Sequence | set | Mapping | NAType | frozenset | None
        ],
        convertDType: _bool = ...,
        args: tuple = ...,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def apply(
        self,
        func: Callable[..., BaseOffset],
        convertDType: _bool = ...,
        args: tuple = ...,
        **kwargs: Any,
    ) -> Series[BaseOffset]: ...
    @overload
    def apply(
        self,
        func: Callable[..., Series],
        convertDType: _bool = ...,
        args: tuple = ...,
        **kwargs: Any,
    ) -> DataFrame: ...
    @final
    def align(
        self,
        other: DataFrame | Series,
        join: JoinHow = "outer",
        axis: Axis | None = 0,
        level: Level | None = None,
        copy: _bool = True,
        fill_value: Scalar | NAType | None = None,
    ) -> tuple[Series, Series]: ...
    @overload
    def rename(
        self,
        index: Callable[[Any], Label],
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def rename(
        self,
        index: Mapping[Any, Label],
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def rename(
        self,
        index: Scalar | tuple[Hashable, ...] | None = None,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Self: ...
    @overload
    def rename(
        self,
        index: Renamer | Scalar | tuple[Hashable, ...] | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[False] = False,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Self: ...
    @final
    def reindex_like(
        self,
        other: Series[S1],
        method: FillnaOptions | Literal["nearest"] | None = None,
        copy: _bool = True,
        limit: int | None = None,
        tolerance: Scalar | AnyArrayLike | Sequence[Scalar] | None = None,
    ) -> Self: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict | Series[S1] | DataFrame | None = ...,
        *,
        axis: AxisIndex = ...,
        limit: int | None = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict | Series[S1] | DataFrame | None = ...,
        *,
        axis: AxisIndex = ...,
        limit: int | None = ...,
        inplace: Literal[False] = False,
    ) -> Series[S1]: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue = ...,
        value: ReplaceValue = ...,
        *,
        regex: ReplaceValue = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue = ...,
        value: ReplaceValue = ...,
        *,
        regex: ReplaceValue = ...,
        inplace: Literal[False] = False,
    ) -> Series[S1]: ...
    def shift(
        self,
        periods: int | Sequence[int] = ...,
        freq: BaseOffset | timedelta | _str | None = None,
        axis: Axis = 0,
        fill_value: Scalar | NAType | None = ...,
    ) -> Series: ...
    def info(
        self,
        verbose: bool | None = ...,
        buf: WriteBuffer[_str] | None = ...,
        memory_usage: bool | Literal["deep"] | None = ...,
        show_counts: bool | None = ...,
    ) -> None: ...
    def memory_usage(self, index: _bool = True, deep: _bool = False) -> int: ...
    def isin(self, values: Iterable[Any]) -> Series[_bool]: ...
    def between(
        self,
        left: Scalar | ListLikeU,
        right: Scalar | ListLikeU,
        inclusive: Literal["both", "neither", "left", "right"] = ...,
    ) -> Series[_bool]: ...
    def isna(self) -> Series[_bool]: ...
    def isnull(self) -> Series[_bool]: ...
    def notna(self) -> Series[_bool]: ...
    def notnull(self) -> Series[_bool]: ...
    @overload
    def dropna(
        self,
        *,
        axis: AxisIndex = ...,
        inplace: Literal[True],
        how: AnyAll | None = ...,
        ignore_index: _bool = ...,
    ) -> None: ...
    @overload
    def dropna(
        self,
        *,
        axis: AxisIndex = ...,
        inplace: Literal[False] = False,
        how: AnyAll | None = ...,
        ignore_index: _bool = ...,
    ) -> Series[S1]: ...
    def to_timestamp(
        self,
        freq: PeriodFrequency | None = None,
        how: ToTimestampHow = "start",
        copy: _bool = True,
    ) -> Series[S1]: ...
    def to_period(
        self, freq: PeriodFrequency | None = None, copy: _bool = True
    ) -> DataFrame: ...
    @property
    def str(
        self,
    ) -> StringMethods[  # pyrefly: ignore[bad-specialization]
        Self,
        DataFrame,
        Series[bool],
        Series[list[_str]],
        Series[int],
        Series[bytes],
        Series[_str],
        Series,
    ]: ...
    dt = DtDescriptor()
    @property
    def plot(self) -> PlotAccessor: ...
    sparse = ...
    def hist(
        self,
        by: object | None = None,
        ax: PlotAxes | None = None,
        grid: _bool = True,
        xlabelsize: float | _str | None = None,
        xrot: float | None = None,
        ylabelsize: float | _str | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | Sequence = 10,
        backend: _str | None = None,
        legend: _bool = False,
        **kwargs: Any,
    ) -> SubplotBase: ...
    @final
    def swapaxes(
        self, axis1: AxisIndex, axis2: AxisIndex, copy: _bool = ...
    ) -> Series[S1]: ...
    @final
    def droplevel(self, level: Level | list[Level], axis: AxisIndex = 0) -> Self: ...
    def pop(self, item: Hashable) -> S1: ...
    @final
    def squeeze(self, axis: None = None) -> Series[S1] | Scalar: ...
    @final
    def __abs__(self) -> Series[S1]: ...
    @final
    def add_prefix(self, prefix: _str, axis: AxisIndex | None = None) -> Series[S1]: ...
    @final
    def add_suffix(self, suffix: _str, axis: AxisIndex | None = None) -> Series[S1]: ...
    def reindex(
        self,
        index: Axes | None = None,
        method: ReindexMethod | None = None,
        copy: bool = True,
        level: int | _str | None = None,
        fill_value: Scalar | None = None,
        limit: int | None = None,
        tolerance: float | Timedelta | None = None,
    ) -> Series[S1]: ...
    def filter(
        self,
        items: ListLike | None = None,
        like: _str | None = None,
        regex: _str | None = None,
        axis: AxisIndex | None = None,
    ) -> Series[S1]: ...
    @final
    def head(self, n: int = 5) -> Series[S1]: ...
    @final
    def tail(self, n: int = 5) -> Series[S1]: ...
    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: _bool = False,
        weights: _str | ListLike | np_ndarray_float | None = None,
        random_state: RandomState | None = None,
        axis: AxisIndex | None = None,
        ignore_index: _bool = False,
    ) -> Series[S1]: ...
    @overload
    def astype(
        self,
        dtype: BooleanDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[bool]: ...
    @overload
    def astype(
        self,
        dtype: IntDtypeArg | UIntDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[int]: ...
    @overload
    def astype(
        self,
        dtype: StrDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[_str]: ...
    @overload
    def astype(
        self,
        dtype: BytesDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[bytes]: ...
    @overload
    def astype(
        self,
        dtype: FloatDtypeArg | PandasAstypeFloatDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[float]: ...
    @overload
    def astype(
        self,
        dtype: ComplexDtypeArg | PandasAstypeComplexDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[complex]: ...
    @overload
    def astype(
        self,
        dtype: TimedeltaDtypeArg | PandasAstypeTimedeltaDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[Timedelta]: ...
    @overload
    def astype(
        self,
        dtype: TimestampDtypeArg | PandasAstypeTimestampDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[Timestamp]: ...
    @overload
    def astype(
        self,
        dtype: CategoryDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[CategoricalDtype]: ...
    @overload
    def astype(
        self,
        dtype: ObjectDtypeArg | VoidDtypeArg | ExtensionDtype | DtypeObj,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series: ...
    @final
    def copy(self, deep: _bool = True) -> Series[S1]: ...
    @final
    def infer_objects(self, copy: _bool = True) -> Series[S1]: ...
    @overload
    def ffill(
        self,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[True],
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> None: ...
    @overload
    def ffill(
        self,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[False] = False,
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Series[S1]: ...
    @overload
    def bfill(
        self,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[True],
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> None: ...
    @overload
    def bfill(
        self,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[False] = False,
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Series[S1]: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: AxisIndex | None = 0,
        limit: int | None = ...,
        inplace: Literal[True],
        limit_direction: Literal["forward", "backward", "both"] | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: AxisIndex | None = 0,
        limit: int | None = ...,
        inplace: Literal[False] = False,
        limit_direction: Literal["forward", "backward", "both"] | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @final
    def asof(
        self,
        where: Scalar | AnyArrayLike | Sequence[Scalar],
        subset: None = None,
    ) -> Scalar | Series[S1]: ...
    @overload
    def clip(  # pyright: ignore[reportOverlappingOverload]
        self,
        lower: None = None,
        upper: None = None,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike | float | None = ...,
        upper: AnyArrayLike | float | None = ...,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike | float | None = ...,
        upper: AnyArrayLike | float | None = ...,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @final
    def asfreq(
        self,
        freq: Frequency,
        method: FillnaOptions | None = None,
        how: Literal["start", "end"] | None = None,
        normalize: _bool = False,
        fill_value: Scalar | None = None,
    ) -> Series[S1]: ...
    @final
    def at_time(
        self,
        time: _str | time,
        asof: _bool = False,
        axis: AxisIndex | None = 0,
    ) -> Series[S1]: ...
    @final
    def between_time(
        self,
        start_time: _str | time,
        end_time: _str | time,
        inclusive: IntervalClosedType = "both",
        axis: AxisIndex | None = 0,
    ) -> Series[S1]: ...
    @final
    def rank(
        self,
        axis: AxisIndex = 0,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        numeric_only: _bool = False,
        na_option: Literal["keep", "top", "bottom"] = "keep",
        ascending: _bool = True,
        pct: _bool = False,
    ) -> Series[float]: ...
    @overload
    def where(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np_ndarray_bool
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: S1 | Self | Callable[..., S1 | Self] = ...,
        *,
        inplace: Literal[True],
        axis: AxisIndex | None = 0,
        level: Level | None = ...,
    ) -> None: ...
    @overload
    def where(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np_ndarray_bool
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Scalar | Self | Callable[..., Scalar | Self] = ...,
        *,
        inplace: Literal[False] = False,
        axis: AxisIndex | None = 0,
        level: Level | None = ...,
    ) -> Self: ...
    @overload
    def mask(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np_ndarray_bool
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Scalar | Series[S1] | DataFrame | Callable | NAType | None = ...,
        *,
        inplace: Literal[True],
        axis: AxisIndex | None = 0,
        level: Level | None = ...,
    ) -> None: ...
    @overload
    def mask(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np_ndarray_bool
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Scalar | Series[S1] | DataFrame | Callable | NAType | None = ...,
        *,
        inplace: Literal[False] = False,
        axis: AxisIndex | None = 0,
        level: Level | None = ...,
    ) -> Series[S1]: ...
    def case_when(
        self,
        caselist: Sequence[
            tuple[
                Sequence[bool]
                | np_1darray_bool
                | Series[bool]
                | Callable[[Series], Sequence[bool] | np_1darray_bool | Series[bool]],
                ListLikeU | Scalar | Callable[[Series], Series | np_ndarray],
            ],
        ],
    ) -> Series: ...
    @final
    def truncate(
        self,
        before: date | _str | int | None = ...,
        after: date | _str | int | None = ...,
        axis: AxisIndex | None = 0,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    @final
    def tz_convert(
        self,
        tz: TimeZones,
        axis: AxisIndex = 0,
        level: Level | None = None,
        copy: _bool = True,
    ) -> Series[S1]: ...
    @final
    def tz_localize(
        self,
        tz: TimeZones,
        axis: AxisIndex = 0,
        level: Level | None = None,
        copy: _bool = True,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: _str = "raise",
    ) -> Series[S1]: ...
    @final
    def abs(self) -> Series[S1]: ...
    @final
    def describe(
        self,
        percentiles: list[float] | None = ...,
        include: Literal["all"] | list[S1] | None = ...,
        exclude: S1 | list[S1] | None = ...,
    ) -> Series[S1]: ...
    @final
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq: Frequency | timedelta | None = None,
    ) -> Series[float]: ...
    @final
    def first_valid_index(self) -> Scalar: ...
    @final
    def last_valid_index(self) -> Scalar: ...
    @overload
    def value_counts(  # pyrefly: ignore
        self,
        normalize: Literal[False] = False,
        sort: _bool = ...,
        ascending: _bool = ...,
        bins: int | None = ...,
        dropna: _bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: _bool = ...,
        ascending: _bool = ...,
        bins: int | None = ...,
        dropna: _bool = ...,
    ) -> Series[float]: ...
    @final
    @property
    def T(self) -> Self: ...
    # The rest of these were left over from the old
    # stubs we shipped in preview. They may belong in
    # the base classes in some cases; I expect stubgen
    # just failed to generate these so I couldn't match
    # them up.
    @overload
    def __add__(self: Series[Never], other: complex | ListLike) -> Series: ...
    @overload
    def __add__(self, other: Index[Never] | Series[Never]) -> Series: ...
    @overload
    def __add__(self: Series[Timestamp], other: np_ndarray_dt) -> Never: ...
    @overload
    def __add__(
        self: Series[Timestamp],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
            | BaseOffset
        ),
    ) -> Series[Timestamp]: ...
    @overload
    def __add__(
        self: Series[Timedelta],
        other: (
            datetime | np.datetime64 | np_ndarray_dt | DatetimeIndex | Series[Timestamp]
        ),
    ) -> Series[Timestamp]: ...
    @overload
    def __add__(
        self: Series[Timedelta],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __add__(
        self: Supports_ProtoAdd[S2_contra, S2], other: S2_contra | Sequence[S2_contra]
    ) -> Series[S2]: ...
    @overload
    def __add__(
        self: Series[S2_contra], other: SupportsRAdd[S2_contra, S2]
    ) -> Series[S2]: ...
    # pandas-dev/pandas#62353
    @overload
    def __add__(
        self: Series[S2_NDT_contra], other: Sequence[SupportsRAdd[S2_NDT_contra, S2]]
    ) -> Series[S2]: ...
    @overload
    def __add__(
        self: Series[T_COMPLEX], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __add__(
        self: Series[bool], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[int]: ...
    @overload
    def __add__(
        self: Series[T_COMPLEX], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __add__(
        self: Series[bool] | Series[int],
        other: np_ndarray_float | Index[float] | Series[float],
    ) -> Series[float]: ...
    @overload
    def __add__(
        self: Series[T_COMPLEX], other: np_ndarray_float | Index[float] | Series[float]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __add__(
        self: Series[T_COMPLEX],
        other: np_ndarray_complex | Index[complex] | Series[complex],
    ) -> Series[complex]: ...
    @overload
    def __add__(
        self: Series[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __add__(
        self: Series[_str], other: np_ndarray_str | Index[_str] | Series[_str]
    ) -> Series[_str]: ...
    @overload
    def add(
        self: Series[Never],
        other: complex | ListLike,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def add(
        self,
        other: Index[Never] | Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def add(
        self: Series[Timestamp],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
            | BaseOffset
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timestamp]: ...
    @overload
    def add(
        self: Series[Timedelta],
        other: (
            datetime
            | Sequence[datetime]
            | np.datetime64
            | np_ndarray_dt
            | DatetimeIndex
            | Series[Timestamp]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timestamp]: ...
    @overload
    def add(
        self: Series[Timedelta],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def add(
        self: Supports_ProtoAdd[S2_contra, S2],
        other: S2_contra | Sequence[S2_contra],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S2]: ...
    @overload
    def add(
        self: Series[S2_contra],
        other: SupportsRAdd[S2_contra, S2] | Sequence[SupportsRAdd[S2_contra, S2]],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S2]: ...
    @overload
    def add(
        self: Series[T_COMPLEX],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def add(
        self: Series[bool],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def add(
        self: Series[T_COMPLEX],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def add(
        self: Series[bool] | Series[int],
        other: np_ndarray_float | Index[float] | Series[float],
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def add(
        self: Series[T_COMPLEX],
        other: np_ndarray_float | Index[float] | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def add(
        self: Series[T_COMPLEX],
        other: np_ndarray_complex | Index[complex] | Series[complex],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def add(
        self: Series[_str],
        other: np_ndarray_str | Index[_str] | Series[_str],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_str]: ...
    @overload
    def __radd__(self: Series[Never], other: complex | ListLike) -> Series: ...
    @overload
    def __radd__(self, other: Index[Never] | Series[Never]) -> Series: ...
    @overload
    def __radd__(self: Series[Timestamp], other: np_ndarray_dt) -> Never: ...
    @overload
    def __radd__(
        self: Series[Timestamp],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
            | BaseOffset
        ),
    ) -> Series[Timestamp]: ...
    @overload
    def __radd__(
        self: Series[Timedelta],
        other: (
            datetime | np.datetime64 | np_ndarray_dt | DatetimeIndex | Series[Timestamp]
        ),
    ) -> Series[Timestamp]: ...
    @overload
    def __radd__(
        self: Series[Timedelta],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
    ) -> Series[Timedelta]: ...
    # pyright is unhappy without the 3 overloads below
    @overload
    def __radd__(self: Series[bool], other: bool | Sequence[bool]) -> Series[bool]: ...
    @overload
    def __radd__(self: Series[float], other: int | Sequence[int]) -> Series[float]: ...
    @overload
    def __radd__(
        self: Series[complex], other: float | Sequence[float]
    ) -> Series[complex]: ...
    # pyright is unhappy without the above 3 overloads
    @overload
    def __radd__(
        self: Supports_ProtoRAdd[S2_contra, S2], other: S2_contra | Sequence[S2_contra]
    ) -> Series[S2]: ...
    @overload
    def __radd__(
        self: Series[S2_contra], other: SupportsAdd[S2_contra, S2]
    ) -> Series[S2]: ...
    # pandas-dev/pandas#62353
    @overload
    def __radd__(
        self: Series[S2_NDT_contra], other: Sequence[SupportsAdd[S2_NDT_contra, S2]]
    ) -> Series[S2]: ...
    @overload
    def __radd__(
        self: Series[T_COMPLEX], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Series[bool], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[int]: ...
    @overload
    def __radd__(
        self: Series[T_COMPLEX], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Series[bool] | Series[int],
        other: np_ndarray_float | Index[float] | Series[float],
    ) -> Series[float]: ...
    @overload
    def __radd__(
        self: Series[T_COMPLEX], other: np_ndarray_float | Index[float] | Series[float]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Series[T_COMPLEX],
        other: np_ndarray_complex | Index[complex] | Series[complex],
    ) -> Series[complex]: ...
    @overload
    def __radd__(
        self: Series[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __radd__(
        self: Series[_str], other: np_ndarray_str | Index[_str] | Series[_str]
    ) -> Series[_str]: ...
    @overload
    def __radd__(self: Series[BaseOffset], other: Period) -> Series[Period]: ...
    @overload
    def __radd__(self: Series[BaseOffset], other: BaseOffset) -> Series[BaseOffset]: ...
    @overload
    def radd(
        self: Series[Never],
        other: complex | ListLike,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def radd(
        self,
        other: Index[Never] | Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def radd(
        self: Series[Timestamp],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
            | BaseOffset
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timestamp]: ...
    @overload
    def radd(
        self: Series[Timedelta],
        other: (
            datetime
            | Sequence[datetime]
            | np.datetime64
            | np_ndarray_dt
            | DatetimeIndex
            | Series[Timestamp]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timestamp]: ...
    @overload
    def radd(
        self: Series[Timedelta],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def radd(
        self: Supports_ProtoRAdd[S2_contra, S2],
        other: S2_contra | Sequence[S2_contra],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S2]: ...
    @overload
    def radd(
        self: Series[S2_contra],
        other: SupportsAdd[S2_contra, S2] | Sequence[SupportsAdd[S2_contra, S2]],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S2]: ...
    @overload
    def radd(
        self: Series[T_COMPLEX],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def radd(
        self: Series[bool],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def radd(
        self: Series[T_COMPLEX],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def radd(
        self: Series[bool] | Series[int],
        other: np_ndarray_float | Index[float] | Series[float],
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def radd(
        self: Series[T_COMPLEX],
        other: np_ndarray_float | Index[float] | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def radd(
        self: Series[T_COMPLEX],
        other: np_ndarray_complex | Index[complex] | Series[complex],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def radd(
        self: Series[_str],
        other: np_ndarray_str | Index[_str] | Series[_str],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_str]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __and__(  # pyright: ignore[reportOverlappingOverload] # pyrefly: ignore[bad-override]
        self, other: bool | list[int] | MaskType
    ) -> Series[bool]: ...
    @overload
    def __and__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    def __eq__(self, other: object) -> Series[_bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    @overload
    def __floordiv__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __floordiv__(
        self: Series[Never], other: np_ndarray_td | TimedeltaIndex
    ) -> Never: ...
    @overload
    def __floordiv__(
        self: Series[int] | Series[float], other: np_ndarray_complex | np_ndarray_td
    ) -> Never: ...
    @overload
    def __floordiv__(  # type: ignore[overload-overlap]
        self: Series[Never], other: ScalarArrayIndexSeriesReal
    ) -> Series: ...
    @overload
    def __floordiv__(
        self: SeriesReal | Series[Timedelta], other: Index[Never] | Series[Never]
    ) -> Series: ...
    @overload
    def __floordiv__(
        self: Series[bool] | Series[complex], other: np_ndarray
    ) -> Never: ...
    @overload
    def __floordiv__(
        self: Supports_ProtoFloorDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Series[S2]: ...
    @overload
    def __floordiv__(
        self: Series[int], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[int]: ...
    @overload
    def __floordiv__(
        self: Series[float], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[float]: ...
    @overload
    def __floordiv__(
        self: Series[bool] | Series[int],
        other: np_ndarray_anyint | Index[int] | Series[int],
    ) -> Series[int]: ...
    @overload
    def __floordiv__(
        self: Series[float], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[float]: ...
    @overload
    def __floordiv__(
        self: Series[int] | Series[float],
        other: (
            float | Sequence[float] | np_ndarray_float | Index[float] | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __floordiv__(
        self: Series[Timedelta], other: np_ndarray_bool | np_ndarray_complex
    ) -> Never: ...
    @overload
    def __floordiv__(
        self: Series[Timedelta],
        other: ScalarArrayIndexSeriesJustInt | ScalarArrayIndexSeriesJustFloat,
    ) -> Series[Timedelta]: ...
    @overload
    def __floordiv__(
        self: Series[Timedelta], other: ArrayIndexSeriesTimedeltaNoSeq
    ) -> Series[int]: ...
    @overload
    def floordiv(
        self: Series[Never],
        other: np_ndarray_td | TimedeltaIndex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Never: ...
    @overload
    def floordiv(
        self: Series[Never],
        other: ScalarArrayIndexSeriesReal,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series: ...
    @overload
    def floordiv(
        self: SeriesReal | Series[Timedelta],
        other: Index[Never] | Series[Never],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series: ...
    @overload
    def floordiv(
        self: Supports_ProtoFloorDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[S2]: ...
    @overload
    def floordiv(
        self: Series[int],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[int]: ...
    @overload
    def floordiv(
        self: Series[float],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[float]: ...
    @overload
    def floordiv(
        self: Series[bool] | Series[int],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[int]: ...
    @overload
    def floordiv(
        self: Series[float],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[float]: ...
    @overload
    def floordiv(
        self: Series[int] | Series[float],
        other: (
            float | Sequence[float] | np_ndarray_float | Index[float] | Series[float]
        ),
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[float]: ...
    @overload
    def floordiv(
        self: Series[Timedelta],
        other: ScalarArrayIndexSeriesJustInt | ScalarArrayIndexSeriesJustFloat,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def floordiv(
        self: Series[Timedelta],
        other: ArrayIndexSeriesTimedeltaNoSeq,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[int]: ...
    if sys.version_info >= (3, 11):
        @overload
        def __rfloordiv__(  # type: ignore[overload-overlap]
            self: Series[Never], other: ScalarArrayIndexSeriesReal
        ) -> Series: ...
    else:
        @overload
        def __rfloordiv__(
            self: Series[Never], other: ScalarArrayIndexSeriesReal
        ) -> Series: ...

    @overload
    def __rfloordiv__(self, other: np_ndarray_complex | np_ndarray_dt) -> Never: ...
    @overload
    def __rfloordiv__(
        self: Series[int] | Series[float], other: np_ndarray_td
    ) -> Never: ...
    @overload
    def __rfloordiv__(
        self: Series[bool] | Series[complex], other: np_ndarray
    ) -> Never: ...
    @overload
    def __rfloordiv__(
        self: SeriesReal | Series[Timedelta], other: Index[Never] | Series[Never]
    ) -> Series: ...
    @overload
    def __rfloordiv__(
        self: Supports_ProtoRFloorDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Series[S2]: ...
    @overload
    def __rfloordiv__(
        self: Series[int], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[int]: ...
    @overload
    def __rfloordiv__(
        self: Series[float], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[float]: ...
    @overload
    def __rfloordiv__(
        self: Series[bool] | Series[int],
        other: np_ndarray_anyint | Index[int] | Series[int],
    ) -> Series[int]: ...
    @overload
    def __rfloordiv__(
        self: Series[float], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[float]: ...
    @overload
    def __rfloordiv__(
        self: Series[int] | Series[float],
        other: (
            float | Sequence[float] | np_ndarray_float | Index[float] | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __rfloordiv__(self: Series[Timedelta], other: np_ndarray_num) -> Never: ...
    @overload
    def __rfloordiv__(
        self: Series[int] | Series[float],
        other: timedelta | np.timedelta64 | ArrayIndexSeriesTimedeltaNoSeq,
    ) -> Series[Timedelta]: ...
    @overload
    def __rfloordiv__(
        self: Series[int] | Series[float],
        other: Sequence[timedelta | np.timedelta64],
    ) -> Series: ...
    @overload
    def __rfloordiv__(
        self: Series[Timedelta], other: ArrayIndexSeriesTimedeltaNoSeq
    ) -> Series[int]: ...
    @overload
    def rfloordiv(
        self: Series[Never],
        other: ScalarArrayIndexSeriesReal,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series: ...
    @overload
    def rfloordiv(
        self: SeriesReal | Series[Timedelta],
        other: Index[Never] | Series[Never],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series: ...
    @overload
    def rfloordiv(
        self: Supports_ProtoRFloorDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S2]: ...
    @overload
    def rfloordiv(
        self: Series[int],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[int]: ...
    @overload
    def rfloordiv(
        self: Series[float],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[float]: ...
    @overload
    def rfloordiv(
        self: Series[bool] | Series[int],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[int]: ...
    @overload
    def rfloordiv(
        self: Series[float],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[float]: ...
    @overload
    def rfloordiv(
        self: Series[int] | Series[float],
        other: (
            float | Sequence[float] | np_ndarray_float | Index[float] | Series[float]
        ),
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[float]: ...
    @overload
    def rfloordiv(
        self: Series[int] | Series[float],
        other: ScalarArrayIndexSeriesTimedelta,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[Timedelta]: ...
    @overload
    def rfloordiv(
        self: Series[Timedelta],
        other: timedelta | np.timedelta64 | ArrayIndexSeriesTimedeltaNoSeq,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[int]: ...
    def __ge__(  # type: ignore[override]
        self, other: S1 | ListLike | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __gt__(  # type: ignore[override]
        self, other: S1 | ListLike | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __le__(  # type: ignore[override]
        self, other: S1 | ListLike | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __lt__(  # type: ignore[override]
        self, other: S1 | ListLike | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    @overload
    def __mul__(  # type: ignore[overload-overlap]
        self: Series[Never], other: complex | NumListLike | Index | Series
    ) -> Series: ...
    @overload
    def __mul__(self, other: Index[Never] | Series[Never]) -> Series: ...
    @overload
    def __mul__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __mul__(
        self: Series[bool] | Series[complex], other: np_ndarray_td
    ) -> Never: ...
    @overload
    def __mul__(
        self: Series[int] | Series[float],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __mul__(self: Series[Timestamp], other: np_ndarray) -> Never: ...
    @overload
    def __mul__(
        self: Series[Timedelta], other: np_ndarray_bool | np_ndarray_complex
    ) -> Never: ...
    @overload
    def __mul__(
        self: Series[Timedelta],
        other: (
            np_ndarray_anyint
            | np_ndarray_float
            | Index[int]
            | Index[float]
            | Series[int]
            | Series[float]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __mul__(
        self: Series[_str],
        other: (
            np_ndarray_bool
            | np_ndarray_float
            | np_ndarray_complex
            | np_ndarray_dt
            | np_ndarray_td
        ),
    ) -> Never: ...
    @overload
    def __mul__(
        self: Series[_str], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[_str]: ...
    @overload
    def __mul__(
        self: Supports_ProtoMul[_T_contra, S2], other: _T_contra | Sequence[_T_contra]
    ) -> Series[S2]: ...
    @overload
    def __mul__(
        self: Series[S2_contra],
        other: (
            SupportsRMul[S2_contra, S2_NSDT]
            | Sequence[SupportsRMul[S2_contra, S2_NSDT]]
        ),
    ) -> Series[S2_NSDT]: ...
    @overload
    def __mul__(
        self: Series[T_COMPLEX], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __mul__(
        self: Series[bool], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[int]: ...
    @overload
    def __mul__(
        self: Series[T_COMPLEX], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __mul__(
        self: Series[bool] | Series[int],
        other: np_ndarray_float | Index[float] | Series[float],
    ) -> Series[float]: ...
    @overload
    def __mul__(
        self: Series[T_COMPLEX], other: np_ndarray_float | Index[float] | Series[float]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __mul__(
        self: Series[T_COMPLEX],
        other: np_ndarray_complex | Index[complex] | Series[complex],
    ) -> Series[complex]: ...
    @overload
    def mul(
        self: Series[Never],
        other: complex | ListLike,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def mul(
        self,
        other: Index[Never] | Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def mul(
        self: Series[int] | Series[float],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def mul(
        self: Series[Timedelta],
        other: (
            np_ndarray_anyint
            | np_ndarray_float
            | Index[int]
            | Index[float]
            | Series[int]
            | Series[float]
        ),
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def mul(
        self: Series[_str],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_str]: ...
    @overload
    def mul(
        self: Supports_ProtoMul[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S2]: ...
    @overload
    def mul(
        self: Series[S2_contra],
        other: (
            SupportsRMul[S2_contra, S2_NSDT]
            | Sequence[SupportsRMul[S2_contra, S2_NSDT]]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S2_NSDT]: ...
    @overload
    def mul(
        self: Series[T_COMPLEX],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def mul(
        self: Series[bool],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def mul(
        self: Series[T_COMPLEX],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def mul(
        self: Series[bool] | Series[int],
        other: np_ndarray_float | Index[float] | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def mul(
        self: Series[T_COMPLEX],
        other: np_ndarray_float | Index[float] | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def mul(
        self: Series[T_COMPLEX],
        other: np_ndarray_complex | Index[complex] | Series[complex],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def __rmul__(  # type: ignore[overload-overlap]
        self: Series[Never], other: complex | NumListLike | Index | Series
    ) -> Series: ...
    @overload
    def __rmul__(self, other: Index[Never] | Series[Never]) -> Series: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __rmul__(  # type: ignore[overload-overlap]
        self: Series[int] | Series[float],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __rmul__(self: Series[Timestamp], other: np_ndarray) -> Never: ...
    @overload
    def __rmul__(
        self: Series[bool] | Series[complex], other: np_ndarray_td
    ) -> Never: ...
    @overload
    def __rmul__(
        self: Series[Timedelta], other: np_ndarray_bool | np_ndarray_complex
    ) -> Never: ...
    @overload
    def __rmul__(
        self: Series[Timedelta],
        other: (
            np_ndarray_anyint
            | np_ndarray_float
            | Index[int]
            | Index[float]
            | Series[int]
            | Series[float]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __rmul__(
        self: Series[_str],
        other: (
            np_ndarray_bool
            | np_ndarray_float
            | np_ndarray_complex
            | np_ndarray_dt
            | np_ndarray_td
        ),
    ) -> Never: ...
    @overload
    def __rmul__(
        self: Series[_str], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[_str]: ...
    @overload
    def __rmul__(
        self: Supports_ProtoRMul[_T_contra, S2], other: _T_contra | Sequence[_T_contra]
    ) -> Series[S2]: ...
    @overload
    def __rmul__(
        self: Series[S2_contra],
        other: (
            SupportsMul[S2_contra, S2_NSDT] | Sequence[SupportsMul[S2_contra, S2_NSDT]]
        ),
    ) -> Series[S2_NSDT]: ...
    @overload
    def __rmul__(
        self: Series[T_COMPLEX], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __rmul__(
        self: Series[bool], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[int]: ...
    @overload
    def __rmul__(
        self: Series[T_COMPLEX], other: np_ndarray_anyint | Index[int] | Series[int]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __rmul__(
        self: Series[bool] | Series[int],
        other: np_ndarray_float | Index[float] | Series[float],
    ) -> Series[float]: ...
    @overload
    def __rmul__(
        self: Series[T_COMPLEX], other: np_ndarray_float | Index[float] | Series[float]
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __rmul__(
        self: Series[T_COMPLEX],
        other: np_ndarray_complex | Index[complex] | Series[complex],
    ) -> Series[complex]: ...
    @overload
    def rmul(
        self: Series[Never],
        other: complex | ListLike,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def rmul(
        self,
        other: Index[Never] | Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def rmul(
        self: Series[int] | Series[float],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def rmul(
        self: Series[Timedelta],
        other: (
            np_ndarray_anyint
            | np_ndarray_float
            | Index[int]
            | Index[float]
            | Series[int]
            | Series[float]
        ),
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def rmul(
        self: Series[_str],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_str]: ...
    @overload
    def rmul(
        self: Supports_ProtoRMul[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S2]: ...
    @overload
    def rmul(
        self: Series[S2_contra],
        other: (
            SupportsMul[S2_contra, S2_NSDT] | Sequence[SupportsMul[S2_contra, S2_NSDT]]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S2_NSDT]: ...
    @overload
    def rmul(
        self: Series[T_COMPLEX],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def rmul(
        self: Series[bool],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def rmul(
        self: Series[T_COMPLEX],
        other: np_ndarray_anyint | Index[int] | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def rmul(
        self: Series[bool] | Series[int],
        other: np_ndarray_float | Index[float] | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rmul(
        self: Series[T_COMPLEX],
        other: np_ndarray_float | Index[float] | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def rmul(
        self: Series[T_COMPLEX],
        other: np_ndarray_complex | Index[complex] | Series[complex],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    def __mod__(self, other: float | ListLike | Series[S1]) -> Series[S1]: ...
    def __ne__(self, other: object) -> Series[_bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __pow__(self, other: complex | ListLike | Series[S1]) -> Series[S1]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __or__(  # pyright: ignore[reportOverlappingOverload] # pyrefly: ignore[bad-override]
        self, other: bool | list[int] | MaskType
    ) -> Series[bool]: ...
    @overload
    def __or__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __rand__(  # pyright: ignore[reportOverlappingOverload] # pyrefly: ignore[bad-override]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __rand__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    def __rdivmod__(self, other: float | ListLike | Series[S1]) -> Series[S1]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __rmod__(self, other: float | ListLike | Series[S1]) -> Series[S1]: ...
    def __rpow__(self, other: complex | ListLike | Series[S1]) -> Series[S1]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __ror__(  # pyright: ignore[reportOverlappingOverload] # pyrefly: ignore[bad-override]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __ror__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __rxor__(  # pyright: ignore[reportOverlappingOverload] # pyrefly: ignore[bad-override]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __rxor__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    @overload
    def __sub__(
        self: Series[Never],
        other: complex | NumListLike | Index[T_COMPLEX] | Series[T_COMPLEX],
    ) -> Series: ...
    @overload
    def __sub__(self, other: Index[Never] | Series[Never]) -> Series: ...
    @overload
    def __sub__(
        self: Series[bool],
        other: (
            Just[int]
            | Sequence[Just[int]]
            | np_ndarray_anyint
            | Index[int]
            | Series[int]
        ),
    ) -> Series[int]: ...
    @overload
    def __sub__(
        self: Series[bool],
        other: (
            Just[float]
            | Sequence[Just[float]]
            | np_ndarray_float
            | Index[float]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __sub__(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Series[bool]
            | Index[int]
            | Series[int]
        ),
    ) -> Series[int]: ...
    @overload
    def __sub__(
        self: Series[int],
        other: (
            Just[float]
            | Sequence[Just[float]]
            | np_ndarray_float
            | Index[float]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __sub__(
        self: Series[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Series[bool]
            | Index[int]
            | Series[int]
            | Index[float]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __sub__(
        self: Series[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
            | Series[T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __sub__(
        self: Series[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
            | Series[complex]
        ),
    ) -> Series[complex]: ...
    @overload
    def __sub__(
        self: Series[Timestamp],
        other: (
            datetime | np.datetime64 | np_ndarray_dt | DatetimeIndex | Series[Timestamp]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __sub__(
        self: Series[Timestamp],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
            | BaseOffset
        ),
    ) -> Series[Timestamp]: ...
    @overload
    def __sub__(self: Series[Timedelta], other: np_ndarray_dt) -> Never: ...
    @overload
    def __sub__(
        self: Series[Timedelta],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __sub__(
        self: Series[Period], other: Series[Period] | Period
    ) -> Series[BaseOffset]: ...
    @overload
    def sub(
        self: Series[Never],
        other: complex | NumListLike | Index[T_COMPLEX] | Series[T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def sub(
        self,
        other: Index[Never] | Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def sub(
        self: Series[bool],
        other: (
            Just[int]
            | Sequence[Just[int]]
            | np_ndarray_anyint
            | Index[int]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def sub(
        self: Series[bool],
        other: (
            Just[float]
            | Sequence[Just[float]]
            | np_ndarray_float
            | Index[float]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def sub(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Series[bool]
            | Index[int]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def sub(
        self: Series[int],
        other: (
            Just[float]
            | Sequence[Just[float]]
            | np_ndarray_float
            | Index[float]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def sub(
        self: Series[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Series[bool]
            | Index[int]
            | Series[int]
            | Index[float]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def sub(
        self: Series[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
            | Series[T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def sub(
        self: Series[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
            | Series[complex]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def sub(
        self: Series[Timestamp],
        other: (
            datetime
            | Sequence[datetime]
            | np.datetime64
            | np_ndarray_dt
            | DatetimeIndex
            | Series[Timestamp]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def sub(
        self: Series[Timestamp],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
            | BaseOffset
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timestamp]: ...
    @overload
    def sub(
        self: Series[Timedelta],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def sub(
        self: Series[Period],
        other: Period | Sequence[Period] | PeriodIndex | Series[Period],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[BaseOffset]: ...
    @overload
    def __rsub__(
        self: Series[Never],
        other: (
            complex
            | datetime
            | np.datetime64
            | np_ndarray_dt
            | NumListLike
            | Index[T_COMPLEX]
            | Series[T_COMPLEX]
        ),
    ) -> Series: ...
    @overload
    def __rsub__(self, other: Index[Never] | Series[Never]) -> Series: ...
    @overload
    def __rsub__(
        self: Series[bool],
        other: (
            Just[int]
            | Sequence[Just[int]]
            | np_ndarray_anyint
            | Index[int]
            | Series[int]
        ),
    ) -> Series[int]: ...
    @overload
    def __rsub__(
        self: Series[bool],
        other: (
            Just[float]
            | Sequence[Just[float]]
            | np_ndarray_float
            | Index[float]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __rsub__(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Series[bool]
            | Index[int]
            | Series[int]
        ),
    ) -> Series[int]: ...
    @overload
    def __rsub__(
        self: Series[int],
        other: (
            Just[float]
            | Sequence[Just[float]]
            | np_ndarray_float
            | Index[float]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __rsub__(
        self: Series[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Series[bool]
            | Index[int]
            | Series[int]
            | Index[float]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __rsub__(
        self: Series[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
            | Series[T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __rsub__(
        self: Series[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
            | Series[complex]
        ),
    ) -> Series[complex]: ...
    @overload
    def __rsub__(self: Series[Timestamp], other: np_ndarray_td) -> Never: ...
    @overload
    def __rsub__(
        self: Series[Timestamp],
        other: (
            datetime | np.datetime64 | np_ndarray_dt | DatetimeIndex | Series[Timestamp]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __rsub__(
        self: Series[Timedelta],
        other: (
            datetime | np.datetime64 | np_ndarray_dt | DatetimeIndex | Series[Timestamp]
        ),
    ) -> Series[Timestamp]: ...
    @overload
    def __rsub__(
        self: Series[Timedelta],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
    ) -> Series[Timedelta]: ...
    @overload
    def __rsub__(
        self: Series[Period], other: Series[Period] | Period
    ) -> Series[BaseOffset]: ...
    @overload
    def rsub(
        self: Series[Never],
        other: (
            complex
            | datetime
            | Sequence[datetime]
            | np.datetime64
            | np_ndarray_dt
            | NumListLike
            | Index[T_COMPLEX]
            | Series[T_COMPLEX]
            | Series[Timestamp]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def rsub(
        self,
        other: Index[Never] | Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def rsub(
        self: Series[bool],
        other: (
            Just[int]
            | Sequence[Just[int]]
            | np_ndarray_anyint
            | Index[int]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def rsub(
        self: Series[bool],
        other: (
            Just[float]
            | Sequence[Just[float]]
            | np_ndarray_float
            | Index[float]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rsub(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Series[bool]
            | Index[int]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def rsub(
        self: Series[int],
        other: (
            Just[float]
            | Sequence[Just[float]]
            | np_ndarray_float
            | Index[float]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rsub(
        self: Series[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Series[bool]
            | Index[int]
            | Series[int]
            | Index[float]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rsub(
        self: Series[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
            | Series[T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def rsub(
        self: Series[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
            | Series[complex]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def rsub(
        self: Series[Timestamp],
        other: (
            datetime
            | Sequence[datetime]
            | np.datetime64
            | np_ndarray_dt
            | DatetimeIndex
            | Series[Timestamp]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def rsub(
        self: Series[Timedelta],
        other: (
            datetime
            | Sequence[datetime]
            | np.datetime64
            | np_ndarray_dt
            | DatetimeIndex
            | Series[Timestamp]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timestamp]: ...
    @overload
    def rsub(
        self: Series[Timedelta],
        other: (
            timedelta
            | Sequence[timedelta]
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaIndex
            | Series[Timedelta]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def rsub(
        self: Series[Period],
        other: Period | Sequence[Period] | PeriodIndex | Series[Period],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[BaseOffset]: ...
    @overload
    def __truediv__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __truediv__(  # type: ignore[overload-overlap]
        self: Series[Never], other: ScalarArrayIndexSeriesComplex
    ) -> Series: ...
    @overload
    def __truediv__(self: Series[Never], other: ArrayIndexTimedeltaNoSeq) -> Never: ...
    @overload
    def __truediv__(self: Series[T_COMPLEX], other: np_ndarray_td) -> Never: ...
    @overload
    def __truediv__(self: Series[bool], other: np_ndarray_bool) -> Never: ...
    @overload
    def __truediv__(
        self: SeriesComplex | Series[Timedelta], other: Index[Never] | Series[Never]
    ) -> Series: ...
    @overload
    def __truediv__(
        self: Series[Timedelta],
        other: np_ndarray_bool | np_ndarray_complex | np_ndarray_dt,
    ) -> Never: ...
    @overload
    def __truediv__(
        self: Supports_ProtoTrueDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Series[S2]: ...
    @overload
    def __truediv__(
        self: Series[int],
        other: np_ndarray_bool | Index[bool] | Series[bool],
    ) -> Series[float]: ...
    @overload
    def __truediv__(
        self: Series[bool] | Series[int], other: ScalarArrayIndexSeriesIntNoBool
    ) -> Series[float]: ...
    @overload
    def __truediv__(
        self: Series[float],
        other: (
            np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
            | Series[bool]
            | Series[int]
        ),
    ) -> Series[float]: ...
    @overload
    def __truediv__(
        self: Series[complex],
        other: (
            np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
            | Series[bool]
            | Series[int]
        ),
    ) -> Series[complex]: ...
    @overload
    def __truediv__(
        self: Series[bool] | Series[int], other: ScalarArrayIndexSeriesJustFloat
    ) -> Series[float]: ...
    @overload
    def __truediv__(
        self: Series[T_COMPLEX], other: ScalarArrayIndexSeriesJustFloat
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __truediv__(
        self: SeriesComplex, other: ScalarArrayIndexSeriesJustComplex
    ) -> Series[complex]: ...
    @overload
    def __truediv__(
        self: Series[Timedelta],
        other: ScalarArrayIndexSeriesJustInt | ScalarArrayIndexSeriesJustFloat,
    ) -> Series[Timedelta]: ...
    @overload
    def __truediv__(
        self: Series[Timedelta], other: ArrayIndexSeriesTimedeltaNoSeq
    ) -> Series[float]: ...
    @overload
    def __truediv__(self: Series[_str], other: Path) -> Series: ...
    @overload
    def truediv(  # type: ignore[overload-overlap]
        self: Series[Never],
        other: ScalarArrayIndexSeriesComplex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    @overload
    def truediv(
        self: Series[Never],
        other: ArrayIndexTimedeltaNoSeq,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Never: ...
    @overload
    def truediv(
        self: SeriesComplex | Series[Timedelta],
        other: Index[Never] | Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    @overload
    def truediv(
        self: Supports_ProtoTrueDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[S2]: ...
    @overload
    def truediv(
        self: Series[int],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def truediv(
        self: Series[bool] | Series[int],
        other: ScalarArrayIndexSeriesIntNoBool,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def truediv(
        self: Series[float],
        other: (
            np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
            | Series[bool]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def truediv(
        self: Series[complex],
        other: (
            np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
            | Series[bool]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def truediv(
        self: Series[bool] | Series[int],
        other: ScalarArrayIndexSeriesJustFloat,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def truediv(
        self: Series[T_COMPLEX],
        other: ScalarArrayIndexSeriesJustFloat,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def truediv(
        self: SeriesComplex,
        other: ScalarArrayIndexSeriesJustComplex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def truediv(
        self: Series[Timedelta],
        other: ScalarArrayIndexSeriesJustInt | ScalarArrayIndexSeriesJustFloat,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def truediv(
        self: Series[Timedelta],
        other: ArrayIndexSeriesTimedeltaNoSeq,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def truediv(
        self: Series[_str],
        other: Path,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    div = truediv
    @overload
    def __rtruediv__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __rtruediv__(
        self: Series[Never],
        other: ScalarArrayIndexSeriesComplex | ScalarArrayIndexSeriesTimedelta,
    ) -> Series: ...
    @overload
    def __rtruediv__(
        self: SeriesComplex, other: Index[Never] | Series[Never]
    ) -> Series: ...
    @overload
    def __rtruediv__(
        self: Series[int] | Series[float], other: Sequence[timedelta | np.timedelta64]
    ) -> Series: ...
    @overload
    def __rtruediv__(
        self: Supports_ProtoRTrueDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Series[S2]: ...
    @overload
    def __rtruediv__(
        self: Series[int], other: np_ndarray_bool | Index[bool] | Series[bool]
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(
        self: Series[bool] | Series[int], other: ScalarArrayIndexSeriesIntNoBool
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(  # type: ignore[misc]
        self: Series[float],
        other: (
            np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
            | Series[bool]
            | Series[int]
        ),
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(
        self: Series[complex],
        other: (
            np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
            | Series[bool]
            | Series[int]
        ),
    ) -> Series[complex]: ...
    @overload
    def __rtruediv__(
        self: Series[bool] | Series[int], other: ScalarArrayIndexSeriesJustFloat
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(
        self: Series[T_COMPLEX], other: ScalarArrayIndexSeriesJustFloat
    ) -> Series[T_COMPLEX]: ...
    @overload
    def __rtruediv__(
        self: SeriesComplex, other: ScalarArrayIndexSeriesJustComplex
    ) -> Series[complex]: ...
    @overload
    def __rtruediv__(
        self: Series[Timedelta], other: ArrayIndexSeriesTimedeltaNoSeq
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(
        self: Series[int] | Series[float], other: ScalarArrayIndexSeriesTimedelta
    ) -> Series[Timedelta]: ...
    @overload
    def __rtruediv__(self: Series[_str], other: Path) -> Series: ...
    @overload
    def rtruediv(
        self: Series[Never],
        other: ScalarArrayIndexSeriesComplex | ScalarArrayIndexSeriesTimedelta,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    @overload
    def rtruediv(
        self: SeriesComplex,
        other: Index[Never] | Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    @overload
    def rtruediv(
        self: Supports_ProtoRTrueDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[S2]: ...
    @overload
    def rtruediv(
        self: Series[int],
        other: np_ndarray_bool | Index[bool] | Series[bool],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def rtruediv(
        self: Series[bool] | Series[int],
        other: ScalarArrayIndexSeriesIntNoBool,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def rtruediv(
        self: Series[float],
        other: (
            np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
            | Series[bool]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def rtruediv(
        self: Series[complex],
        other: (
            np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
            | Series[bool]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def rtruediv(
        self: Series[bool] | Series[int],
        other: ScalarArrayIndexSeriesJustFloat,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def rtruediv(
        self: Series[T_COMPLEX],
        other: ScalarArrayIndexSeriesJustFloat,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[T_COMPLEX]: ...
    @overload
    def rtruediv(
        self: SeriesComplex,
        other: ScalarArrayIndexSeriesJustComplex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def rtruediv(
        self: Series[Timedelta],
        other: ArrayIndexSeriesTimedeltaNoSeq,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def rtruediv(
        self: SeriesReal,
        other: ScalarArrayIndexSeriesTimedelta,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[Timedelta]: ...
    @overload
    def rtruediv(
        self: Series[_str],
        other: Path,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    rdiv = rtruediv
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __xor__(  # pyright: ignore[reportOverlappingOverload] # pyrefly: ignore[bad-override]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __xor__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    @final
    def __invert__(self) -> Series[bool]: ...
    @property
    def at(self) -> _AtIndexer: ...
    @property
    def cat(self) -> CategoricalAccessor: ...
    @property
    def iat(self) -> _iAtIndexer: ...
    @property
    def iloc(self) -> _iLocIndexerSeries[S1]: ...
    @property
    def loc(self) -> _LocIndexerSeries[S1]: ...
    def all(
        self,
        axis: AxisIndex = 0,
        bool_only: _bool | None = False,
        skipna: _bool = True,
        **kwargs: Any,
    ) -> np.bool: ...
    def any(
        self,
        *,
        axis: AxisIndex = ...,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        **kwargs: Any,
    ) -> np.bool: ...
    def cummax(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def cummin(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @overload
    def cumprod(
        self: Series[Never],
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def cumprod(
        self: Series[bool],
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series[int]: ...
    @overload
    def cumprod(
        self: SupportsGetItem[Scalar, _SupportsMul[S1]],
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def cumsum(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def divmod(
        self,
        other: float | ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def eq(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    @final
    def ewm(
        self,
        com: float | None = None,
        span: float | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
        min_periods: int = 0,
        adjust: _bool = True,
        ignore_na: _bool = False,
        axis: Axis = 0,
        times: np_ndarray_dt | Series | None = None,
        method: CalculationMethod = "single",
    ) -> ExponentialMovingWindow[Series]: ...
    @final
    def expanding(
        self,
        min_periods: int = 1,
        axis: Literal[0] = 0,
        method: CalculationMethod = "single",
    ) -> Expanding[Series]: ...
    def ge(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    def gt(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    @final
    def item(self) -> S1: ...
    def kurt(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def kurtosis(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def le(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    def lt(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    def max(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S1: ...
    @overload
    def mean(
        self: Series[Never],
        axis: AxisIndex | None = ...,
        skipna: _bool = ...,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    @overload
    def mean(
        self: Series[Timestamp],
        axis: AxisIndex | None = ...,
        skipna: _bool = ...,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Timestamp: ...
    @overload
    def mean(
        self: SupportsGetItem[Scalar, SupportsTruedivInt[S2]],
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S2: ...
    @overload
    def median(
        self: Series[Never],
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    @overload
    def median(
        self: Series[complex],
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    @overload
    def median(
        self: SupportsGetItem[Scalar, SupportsTruedivInt[S2]],
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S2: ...
    @overload
    def median(
        self: Series[Timestamp],
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Timestamp: ...
    def min(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S1: ...
    def mod(
        self,
        other: float | ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[S1]: ...
    def ne(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    @final
    def nunique(self, dropna: _bool = True) -> int: ...
    def pow(
        self,
        other: complex | ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[S1]: ...
    def prod(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Scalar: ...
    def product(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Scalar: ...
    def rdivmod(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def rmod(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    @overload
    def rolling(
        self,
        window: int | Frequency | timedelta | BaseIndexer,
        min_periods: int | None = ...,
        center: _bool = ...,
        on: _str | None = ...,
        closed: IntervalClosedType | None = ...,
        step: int | None = ...,
        method: CalculationMethod = ...,
        *,
        win_type: _str,
    ) -> Window[Series]: ...
    @overload
    def rolling(
        self,
        window: int | Frequency | timedelta | BaseIndexer,
        min_periods: int | None = ...,
        center: _bool = ...,
        on: _str | None = ...,
        closed: IntervalClosedType | None = ...,
        step: int | None = ...,
        method: CalculationMethod = ...,
        *,
        win_type: None = None,
    ) -> Rolling[Series]: ...
    def rpow(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def sem(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def skew(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    @overload
    def std(
        self: Series[Never],
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    @overload
    def std(
        self: Series[complex],
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        level: None = None,
        ddof: int = ...,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> np.float64: ...
    @overload
    def std(
        self: Series[Timestamp],
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        level: None = None,
        ddof: int = ...,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Timedelta: ...
    @overload
    def std(
        self: SupportsGetItem[Scalar, SupportsTruedivInt[S2]],
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S2: ...
    def sum(
        self: SupportsGetItem[Scalar, _SupportsAdd[_T]],
        axis: AxisIndex | None = 0,
        skipna: _bool | None = ...,
        numeric_only: _bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> _T: ...
    def to_list(self) -> list[S1]: ...
    @overload  # type: ignore[override]
    def to_numpy(
        self: Series[Never],
        dtype: DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    @overload
    def to_numpy(
        self: Series[Timestamp],
        dtype: type[np.datetime64] | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_dt: ...
    @overload
    def to_numpy(
        self: Series[Timestamp],
        dtype: np.dtype[GenericT] | SupportsDType[GenericT] | type[GenericT],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT]: ...
    @overload
    def to_numpy(
        self: Series[Timedelta],
        dtype: type[np.timedelta64] | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_td: ...
    @overload
    def to_numpy(
        self: Series[Timedelta],
        dtype: np.dtype[GenericT] | SupportsDType[GenericT] | type[GenericT],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT]: ...
    @overload
    def to_numpy(
        self: Series[Period],
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_object: ...
    @overload
    def to_numpy(
        self: Series[Period],
        dtype: type[np.int64],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_int64: ...
    @overload
    def to_numpy(
        self: Series[BaseOffset],
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_object: ...
    @overload
    def to_numpy(
        self: Series[BaseOffset],
        dtype: type[np.bytes_],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    @overload
    def to_numpy(
        self: Series[Interval],
        dtype: type[np.object_] | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_object: ...
    @overload
    def to_numpy(
        self: Series[Interval],
        dtype: type[T_INTERVAL_NP],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    @overload
    def to_numpy(
        self: Series[int],
        dtype: DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_anyint: ...
    @overload
    def to_numpy(
        self: Series[float],
        dtype: DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_float: ...
    @overload
    def to_numpy(
        self: Series[complex],
        dtype: DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_complex: ...
    @overload
    def to_numpy(
        self: Series[bool],
        dtype: DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_bool: ...
    @overload
    def to_numpy(
        self: Series[_str],
        dtype: NumpyStrDtypeArg,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_str: ...
    @overload
    def to_numpy(
        self: Series[_str],
        dtype: DTypeLike,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    @overload
    def to_numpy(
        self: Series[_str],
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_object: ...
    @overload
    def to_numpy(
        self: Series[bytes],
        dtype: DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray_bytes: ...
    @overload
    def to_numpy(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    def tolist(self) -> list[S1]: ...
    def var(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    # Rename axis with `mapper`, `axis`, and `inplace=True`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: AxisIndex | None = 0,
        copy: _bool = ...,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `mapper`, `axis`, and `inplace=False`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: AxisIndex | None = 0,
        copy: _bool = ...,
        inplace: Literal[False] = False,
    ) -> Self: ...
    # Rename axis with `index` and `inplace=True`
    @overload
    def rename_axis(
        self,
        *,
        index: Scalar | ListLike | Callable | dict | None = ...,
        copy: _bool = ...,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `index` and `inplace=False`
    @overload
    def rename_axis(
        self,
        *,
        index: Scalar | ListLike | Callable | dict | None = ...,
        copy: _bool = ...,
        inplace: Literal[False] = False,
    ) -> Self: ...
    def set_axis(
        self,
        labels: AxesData,
        *,
        axis: Axis = 0,
        copy: _bool | _NoDefaultDoNotUse = ...,
    ) -> Self: ...
    @final
    def xs(  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        self,
        key: Hashable,
        axis: AxisIndex = 0,  # type: ignore[override]
        level: Level | None = ...,
        drop_level: _bool = True,
    ) -> Self: ...
    @final
    def __bool__(self) -> NoReturn: ...
