from builtins import (
    bool as _bool,
    str as _str,
)
from collections import (
    OrderedDict,
    defaultdict,
)
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
import datetime as dt
import sys
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    TypeAlias,
    TypeVar,
    final,
    overload,
)

from matplotlib.axes import Axes as PlotAxes
import numpy as np
from numpy import typing as npt
from pandas import (
    Period,
    Timedelta,
    Timestamp,
)
from pandas.core.arraylike import OpsMixin
from pandas.core.base import IndexOpsMixin
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.core.indexers import BaseIndexer
from pandas.core.indexes.base import Index
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
from pandas.core.reshape.pivot import (
    _PivotAggFunc,
    _PivotTableColumnsTypes,
    _PivotTableIndexTypes,
    _PivotTableValuesTypes,
)
from pandas.core.series import Series
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

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._libs.missing import NAType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    S2,
    AggFuncTypeBase,
    AggFuncTypeDictFrame,
    AggFuncTypeDictSeries,
    AggFuncTypeFrame,
    AlignJoin,
    AnyAll,
    AnyArrayLike,
    ArrayLike,
    AstypeArg,
    Axes,
    AxesData,
    Axis,
    AxisColumn,
    AxisIndex,
    CalculationMethod,
    ColspaceArgType,
    CompressionOptions,
    DropKeep,
    Dtype,
    FilePath,
    FillnaOptions,
    FloatFormatType,
    FormattersType,
    Frequency,
    GroupByObjectNonScalar,
    HashableT,
    HashableT1,
    HashableT2,
    HashableT3,
    IgnoreRaise,
    IndexingInt,
    IndexKeyFunc,
    IndexLabel,
    IndexType,
    InterpolateOptions,
    IntervalClosedType,
    IntervalT,
    IntoColumn,
    JoinValidate,
    JsonFrameOrient,
    JSONSerializable,
    Label,
    Level,
    ListLike,
    ListLikeExceptSeriesAndStr,
    ListLikeHashable,
    ListLikeU,
    MaskType,
    MergeHow,
    MergeValidate,
    NaPosition,
    NDFrameT,
    NsmallestNlargestKeep,
    ParquetEngine,
    PeriodFrequency,
    QuantileInterpolation,
    RandomState,
    ReadBuffer,
    ReindexMethod,
    Renamer,
    ReplaceValue,
    Scalar,
    ScalarT,
    SequenceNotStr,
    SeriesByT,
    SortKind,
    StataDateFormat,
    StorageOptions,
    StrDtypeArg,
    StrLike,
    Suffixes,
    T as _T,
    TimeAmbiguous,
    TimeNonexistent,
    TimeUnit,
    TimeZones,
    ToStataByteorder,
    ToTimestampHow,
    UpdateJoin,
    ValueKeyFunc,
    WriteBuffer,
    XMLParsers,
    np_2darray,
    np_ndarray,
    np_ndarray_bool,
    np_ndarray_float,
    np_ndarray_num,
)

from pandas.io.formats.style import Styler
from pandas.plotting import PlotAccessor
from pandas.plotting._core import _BoxPlotT

_T_MUTABLE_MAPPING_co = TypeVar(
    "_T_MUTABLE_MAPPING_co", bound=MutableMapping, covariant=True
)

class _iLocIndexerFrame(_iLocIndexer, Generic[_T]):
    @overload
    def __getitem__(self, key: tuple[int, int]) -> Scalar: ...
    @overload
    def __getitem__(self, key: IndexingInt) -> Series: ...
    @overload
    def __getitem__(self, key: tuple[IndexType | MaskType, int]) -> Series: ...
    @overload
    def __getitem__(self, key: tuple[int, IndexType | MaskType]) -> Series: ...
    @overload
    def __getitem__(
        self,
        key: (
            IndexType
            | MaskType
            | tuple[IndexType | MaskType, IndexType | MaskType]
            | tuple[slice]
        ),
    ) -> _T: ...

    # Keep in sync with `DataFrame.__setitem__`
    def __setitem__(
        self,
        key: (
            int
            | IndexType
            | tuple[int, int]
            | tuple[IndexType, int]
            | tuple[IndexType, IndexType]
            | tuple[int, IndexType]
        ),
        value: (
            Scalar
            | IndexOpsMixin
            | Sequence[Scalar]
            | DataFrame
            | np_ndarray
            | NAType
            | NaTType
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...

class _LocIndexerFrame(_LocIndexer, Generic[_T]):
    @overload
    def __getitem__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        self,
        idx: tuple[
            int
            | StrLike
            | Timestamp
            | tuple[Scalar, ...]
            | Callable[[DataFrame], ScalarT],
            int | StrLike | tuple[Scalar, ...],
        ],
    ) -> Scalar: ...
    @overload
    def __getitem__(
        self,
        idx: (
            Callable[[DataFrame], ScalarT]
            | tuple[
                IndexType
                | MaskType
                | _IndexSliceTuple
                | SequenceNotStr[float | str | Timestamp]
                | Callable[
                    [DataFrame], ScalarT | list[HashableT] | IndexType | MaskType
                ],
                ScalarT | None,
            ]
            | None
        ),
    ) -> Series: ...
    @overload
    def __getitem__(self, idx: Scalar) -> Series | _T: ...
    @overload
    def __getitem__(self, idx: tuple[Scalar, slice]) -> Series | _T: ...
    @overload
    def __getitem__(
        self,
        key: (
            IndexType
            | MaskType
            | Callable[[DataFrame], IndexType | MaskType | Sequence[Hashable]]
            | list[HashableT]
            | tuple[
                IndexType
                | MaskType
                | list[HashableT]
                | slice
                | _IndexSliceTuple
                | Callable,
                MaskType | Iterable[HashableT] | IndexType | Callable,
            ]
        ),
    ) -> _T: ...

    # Keep in sync with `DataFrame.__setitem__`
    @overload
    def __setitem__(
        self,
        key: (
            MaskType | StrLike | _IndexSliceTuple | list[ScalarT] | IndexingInt | slice
        ),
        value: (
            Scalar
            | NAType
            | NaTType
            | ArrayLike
            | IndexOpsMixin
            | Sequence[Scalar]
            | Sequence[Sequence[Scalar]]
            | DataFrame
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        key: tuple[_IndexSliceTuple, Hashable],
        value: (
            Scalar
            | NAType
            | NaTType
            | ArrayLike
            | IndexOpsMixin
            | Sequence[Scalar]
            | Sequence[Sequence[Scalar]]
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...

class _iAtIndexerFrame(_iAtIndexer):
    def __getitem__(self, key: tuple[int, int]) -> Scalar: ...
    def __setitem__(
        self,
        key: tuple[int, int],
        value: Scalar | NAType | NaTType | None,
    ) -> None: ...

class _AtIndexerFrame(_AtIndexer):
    def __getitem__(
        self,
        key: tuple[
            int
            | StrLike
            | Timestamp
            | tuple[Scalar, ...]
            | Callable[[DataFrame], ScalarT],
            int | StrLike | tuple[Scalar, ...],
        ],
    ) -> Scalar: ...
    def __setitem__(
        self,
        key: (
            MaskType | StrLike | _IndexSliceTuple | list[ScalarT] | IndexingInt | slice
        ),
        value: (
            Scalar
            | NAType
            | NaTType
            | ArrayLike
            | IndexOpsMixin
            | DataFrame
            | Sequence[Scalar]
            | Sequence[Sequence[Scalar]]
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...

# With mypy 1.14.1 and python 3.12, the second overload needs a type-ignore statement
if sys.version_info >= (3, 12):
    class _GetItemHack:
        @overload
        def __getitem__(self, key: Scalar | tuple[Hashable, ...]) -> Series: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        @overload
        def __getitem__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
            self, key: Iterable[Hashable] | slice
        ) -> Self: ...
        @overload
        def __getitem__(self, key: Hashable) -> Series: ...

else:
    class _GetItemHack:
        @overload
        def __getitem__(self, key: Scalar | tuple[Hashable, ...]) -> Series: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        @overload
        def __getitem__(  # pyright: ignore[reportOverlappingOverload]
            self, key: Iterable[Hashable] | slice
        ) -> Self: ...
        @overload
        def __getitem__(self, key: Hashable) -> Series: ...

_AstypeArgExt: TypeAlias = (
    AstypeArg
    | Literal[
        "number",
        "datetime64",
        "datetime",
        "integer",
        "timedelta",
        "timedelta64",
        "datetimetz",
        "datetime64[ns]",
    ]
)
_AstypeArgExtList: TypeAlias = _AstypeArgExt | list[_AstypeArgExt]

class DataFrame(NDFrame, OpsMixin, _GetItemHack):

    __hash__: ClassVar[None]  # type: ignore[assignment] # pyright: ignore[reportIncompatibleMethodOverride]

    @overload
    def __new__(
        cls,
        data: (
            ListLikeU
            | DataFrame
            | dict[Any, Any]
            | Iterable[ListLikeU | tuple[Hashable, ListLikeU] | dict[Any, Any]]
            | None
        ) = None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: _bool | None = None,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: Scalar,
        index: Axes,
        columns: Axes,
        dtype: Dtype | None = None,
        copy: _bool | None = None,
    ) -> Self: ...
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...
    @property
    def axes(self) -> list[Index]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def style(self) -> Styler: ...
    def items(self) -> Iterator[tuple[Hashable, Series]]: ...
    def iterrows(self) -> Iterator[tuple[Hashable, Series]]: ...
    @overload
    def itertuples(
        self, index: _bool = ..., name: _str = ...
    ) -> Iterator[_PandasNamedTuple]: ...
    @overload
    def itertuples(
        self, index: _bool = ..., name: None = None
    ) -> Iterator[tuple[Any, ...]]: ...
    def __len__(self) -> int: ...
    @overload
    def dot(self, other: DataFrame | ArrayLike) -> Self: ...
    @overload
    def dot(self, other: Series) -> Series: ...
    @overload
    def __matmul__(self, other: DataFrame) -> Self: ...
    @overload
    def __matmul__(self, other: Series) -> Series: ...
    @overload
    def __matmul__(self, other: np_ndarray_num) -> Self: ...
    def __rmatmul__(self, other: np_ndarray_num) -> Self: ...
    @overload
    @classmethod
    def from_dict(
        cls,
        data: dict[Any, Any],
        orient: Literal["index"],
        dtype: AstypeArg | None = ...,
        columns: Axes | None = ...,
    ) -> Self: ...
    @overload
    @classmethod
    def from_dict(
        cls,
        data: dict[Any, Any],
        orient: Literal["columns", "tight"] = ...,
        dtype: AstypeArg | None = ...,
    ) -> Self: ...
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = ...,
        copy: bool = False,
        na_value: Scalar = ...,
    ) -> np_2darray: ...
    @overload
    def to_dict(
        self,
        orient: str = ...,
        *,
        into: type[defaultdict],
        index: Literal[True] = True,
    ) -> Never: ...
    @overload
    def to_dict(
        self,
        orient: Literal["records"],
        *,
        into: _T_MUTABLE_MAPPING_co | type[_T_MUTABLE_MAPPING_co],
        index: Literal[True] = True,
    ) -> list[_T_MUTABLE_MAPPING_co]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["records"],
        *,
        into: type[dict] = ...,
        index: Literal[True] = True,
    ) -> list[dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["index"],
        *,
        into: defaultdict,
        index: Literal[True] = True,
    ) -> defaultdict[Hashable, dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["index"],
        *,
        into: OrderedDict | type[OrderedDict],
        index: Literal[True] = True,
    ) -> OrderedDict[Hashable, dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["index"],
        *,
        into: type[MutableMapping],
        index: Literal[True] = True,
    ) -> MutableMapping[Hashable, dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["index"],
        *,
        into: type[dict] = ...,
        index: Literal[True] = True,
    ) -> dict[Hashable, dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["dict", "list", "series"] = ...,
        *,
        into: _T_MUTABLE_MAPPING_co | type[_T_MUTABLE_MAPPING_co],
        index: Literal[True] = True,
    ) -> _T_MUTABLE_MAPPING_co: ...
    @overload
    def to_dict(
        self,
        orient: Literal["dict", "list", "series"] = ...,
        *,
        into: type[dict] = ...,
        index: Literal[True] = True,
    ) -> dict[Hashable, Any]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["split", "tight"],
        *,
        into: MutableMapping | type[MutableMapping],
        index: bool = ...,
    ) -> MutableMapping[str, list]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["split", "tight"],
        *,
        into: type[dict] = ...,
        index: bool = ...,
    ) -> dict[str, list]: ...
    @classmethod
    def from_records(
        cls,
        data: (
            np_2darray
            | Iterable[SequenceNotStr[Any]]
            | Iterable[Mapping[HashableT, Any]]
            | Mapping[HashableT, Any]
            | Mapping[HashableT, SequenceNotStr[Any]]
        ),
        index: str | Axes | None = None,
        exclude: ListLike | None = None,
        columns: ListLike | None = None,
        coerce_float: bool = False,
        nrows: int | None = None,
    ) -> Self: ...
    def to_records(
        self,
        index: _bool = True,
        column_dtypes: (
            _str | npt.DTypeLike | Mapping[HashableT1, npt.DTypeLike] | None
        ) = None,
        index_dtypes: (
            _str | npt.DTypeLike | Mapping[HashableT2, npt.DTypeLike] | None
        ) = None,
    ) -> np.recarray: ...
    @overload
    def to_stata(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        convert_dates: dict[HashableT1, StataDateFormat] | None = ...,
        write_index: _bool = ...,
        byteorder: ToStataByteorder | None = ...,
        time_stamp: dt.datetime | None = ...,
        data_label: _str | None = ...,
        variable_labels: dict[HashableT2, str] | None = ...,
        version: Literal[117, 118, 119],
        convert_strl: SequenceNotStr[Hashable] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
        value_labels: dict[Hashable, dict[float, str]] | None = ...,
    ) -> None: ...
    @overload
    def to_stata(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        convert_dates: dict[HashableT1, StataDateFormat] | None = ...,
        write_index: _bool = ...,
        byteorder: Literal["<", ">", "little", "big"] | None = ...,
        time_stamp: dt.datetime | None = ...,
        data_label: _str | None = ...,
        variable_labels: dict[HashableT2, str] | None = ...,
        version: Literal[114, 117, 118, 119] | None = ...,
        convert_strl: None = None,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
        value_labels: dict[Hashable, dict[float, str]] | None = ...,
    ) -> None: ...
    def to_feather(
        self, path: FilePath | WriteBuffer[bytes], **kwargs: Any
    ) -> None: ...
    @overload
    def to_parquet(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        engine: ParquetEngine = ...,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = ...,
        index: bool | None = ...,
        partition_cols: Sequence[Hashable] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def to_parquet(
        self,
        path: None = None,
        *,
        engine: ParquetEngine = ...,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = ...,
        index: bool | None = ...,
        partition_cols: Sequence[Hashable] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs: Any,
    ) -> bytes: ...
    @overload
    def to_orc(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        engine: Literal["pyarrow"] = ...,
        index: bool | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    @overload
    def to_orc(
        self,
        path: None = None,
        *,
        engine: Literal["pyarrow"] = ...,
        index: bool | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> bytes: ...
    @overload
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        columns: SequenceNotStr[Hashable] | Index | Series | None = ...,
        col_space: ColspaceArgType | None = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters: (
            list[Callable[[object], str]]
            | tuple[Callable[[object], str], ...]
            | Mapping[Hashable, Callable[[object], str]]
            | None
        ) = ...,
        float_format: Callable[[float], str] | None = ...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        justify: (
            Literal[
                "left",
                "right",
                "center",
                "justify",
                "justify-all",
                "start",
                "end",
                "inherit",
                "match-parent",
                "initial",
                "unset",
            ]
            | None
        ) = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        bold_rows: _bool = ...,
        classes: Sequence[str] | None = ...,
        escape: _bool = ...,
        notebook: _bool = ...,
        border: int | None = ...,
        table_id: _str | None = ...,
        render_links: _bool = ...,
        encoding: _str | None = ...,
    ) -> None: ...
    @overload
    def to_html(
        self,
        buf: None = None,
        *,
        columns: Sequence[Hashable] | None = ...,
        col_space: ColspaceArgType | None = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters: (
            list[Callable[[object], str]]
            | tuple[Callable[[object], str], ...]
            | Mapping[Hashable, Callable[[object], str]]
            | None
        ) = ...,
        float_format: Callable[[float], str] | None = ...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        justify: (
            Literal[
                "left",
                "right",
                "center",
                "justify",
                "justify-all",
                "start",
                "end",
                "inherit",
                "match-parent",
                "initial",
                "unset",
            ]
            | None
        ) = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        bold_rows: _bool = ...,
        classes: Sequence[str] | None = ...,
        escape: _bool = ...,
        notebook: _bool = ...,
        border: int | None = ...,
        table_id: _str | None = ...,
        render_links: _bool = ...,
        encoding: _str | None = ...,
    ) -> _str: ...
    @overload
    def to_xml(
        self,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str],
        index: bool = ...,
        root_name: str = ...,
        row_name: str = ...,
        na_rep: str | None = ...,
        attr_cols: SequenceNotStr[Hashable] | None = ...,
        elem_cols: SequenceNotStr[Hashable] | None = ...,
        namespaces: dict[str | None, str] | None = ...,
        prefix: str | None = ...,
        encoding: str = ...,
        xml_declaration: bool = ...,
        pretty_print: bool = ...,
        parser: XMLParsers = ...,
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @overload
    def to_xml(
        self,
        path_or_buffer: None = None,
        index: bool = ...,
        root_name: str | None = ...,
        row_name: str | None = ...,
        na_rep: str | None = ...,
        attr_cols: list[Hashable] | None = ...,
        elem_cols: list[Hashable] | None = ...,
        namespaces: dict[str | None, str] | None = ...,
        prefix: str | None = ...,
        encoding: str = ...,
        xml_declaration: bool | None = ...,
        pretty_print: bool | None = ...,
        parser: str | None = ...,
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> str: ...
    def info(
        self,
        verbose: bool | None = ...,
        buf: WriteBuffer[str] | None = ...,
        max_cols: int | None = ...,
        memory_usage: bool | Literal["deep"] | None = ...,
        show_counts: bool | None = ...,
    ) -> None: ...
    def memory_usage(self, index: _bool = True, deep: _bool = False) -> Series: ...
    def transpose(self, *args: Any, copy: _bool = ...) -> Self: ...
    @property
    def T(self) -> Self: ...
    @final
    def __getattr__(self, name: str) -> Series: ...
    def isetitem(
        self, loc: int | Sequence[int], value: Scalar | ArrayLike | list[Any]
    ) -> None: ...

    # Keep in sync with `_iLocIndexerFrame.__setitem__`
    @overload
    def __setitem__(
        self,
        idx: (
            int
            | IndexType
            | tuple[int, int]
            | tuple[IndexType, int]
            | tuple[IndexType, IndexType]
            | tuple[int, IndexType]
        ),
        value: (
            Scalar
            | IndexOpsMixin
            | Sequence[Scalar]
            | DataFrame
            | np_ndarray
            | NAType
            | NaTType
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...
    # Keep in sync with `_LocIndexerFrame.__setitem__`
    @overload
    def __setitem__(
        self,
        idx: (
            MaskType | StrLike | _IndexSliceTuple | list[ScalarT] | IndexingInt | slice
        ),
        value: (
            Scalar
            | NAType
            | NaTType
            | ArrayLike
            | IndexOpsMixin
            | Sequence[Scalar]
            | Sequence[Sequence[Scalar]]
            | DataFrame
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: tuple[_IndexSliceTuple, Hashable],
        value: (
            Scalar
            | NAType
            | NaTType
            | ArrayLike
            | IndexOpsMixin
            | Sequence[Scalar]
            | Sequence[Sequence[Scalar]]
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...
    # Extra cases not supported by  `_LocIndexerFrame.__setitem__` /
    # `_iLocIndexerFrame.__setitem__`.
    @overload
    def __setitem__(
        self,
        idx: IndexOpsMixin | DataFrame,
        value: (
            Scalar
            | NAType
            | NaTType
            | ArrayLike
            | IndexOpsMixin
            | Sequence[Scalar]
            | Sequence[Sequence[Scalar]]
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...
    @overload
    def query(
        self,
        expr: _str,
        *,
        parser: Literal["pandas", "python"] = ...,
        engine: Literal["python", "numexpr"] | None = ...,
        local_dict: dict[_str, Any] | None = ...,
        global_dict: dict[_str, Any] | None = ...,
        resolvers: list[Mapping] | None = ...,
        level: int = ...,
        target: object | None = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def query(
        self,
        expr: _str,
        *,
        inplace: Literal[False] = False,
        parser: Literal["pandas", "python"] = ...,
        engine: Literal["python", "numexpr"] | None = ...,
        local_dict: dict[_str, Any] | None = ...,
        global_dict: dict[_str, Any] | None = ...,
        resolvers: list[Mapping] | None = ...,
        level: int = ...,
        target: object | None = ...,
    ) -> Self: ...
    @overload
    def eval(self, expr: _str, *, inplace: Literal[True], **kwargs: Any) -> None: ...
    @overload
    def eval(
        self, expr: _str, *, inplace: Literal[False] = False, **kwargs: Any
    ) -> Scalar | np_ndarray | Self | Series: ...
    @overload
    def select_dtypes(
        self, include: StrDtypeArg, exclude: _AstypeArgExtList | None = ...
    ) -> Never: ...
    @overload
    def select_dtypes(
        self, include: _AstypeArgExtList | None, exclude: StrDtypeArg
    ) -> Never: ...
    @overload
    def select_dtypes(self, exclude: StrDtypeArg) -> Never: ...
    @overload
    def select_dtypes(self, include: list[Never], exclude: list[Never]) -> Never: ...
    @overload
    def select_dtypes(
        self,
        include: _AstypeArgExtList,
        exclude: _AstypeArgExtList | None = ...,
    ) -> Self: ...
    @overload
    def select_dtypes(
        self,
        include: _AstypeArgExtList | None,
        exclude: _AstypeArgExtList,
    ) -> Self: ...
    @overload
    def select_dtypes(
        self,
        exclude: _AstypeArgExtList,
    ) -> Self: ...
    def insert(
        self,
        loc: int,
        column: Hashable,
        value: Scalar | ListLikeU | None,
        allow_duplicates: _bool = ...,
    ) -> None: ...
    def assign(self, **kwargs: IntoColumn) -> Self: ...
    @final
    def align(
        self,
        other: NDFrameT,
        join: AlignJoin = "outer",
        axis: Axis | None = None,
        level: Level | None = None,
        copy: _bool = True,
        fill_value: Scalar | NAType | None = ...,
    ) -> tuple[Self, NDFrameT]: ...
    def reindex(
        self,
        labels: Axes | None = ...,
        *,
        index: Axes | None = ...,
        columns: Axes | None = ...,
        axis: Axis | None = ...,
        method: ReindexMethod | None = ...,
        copy: bool = True,
        level: int | _str = ...,
        fill_value: Scalar | None = ...,
        limit: int | None = None,
        tolerance: float | Timedelta | None = ...,
    ) -> Self: ...
    @overload
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = None,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[False] = False,
        level: Level | None = None,
        errors: IgnoreRaise = ...,
    ) -> Self: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict | Series | DataFrame | None = ...,
        *,
        axis: Axis | None = ...,
        limit: int = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict | Series | DataFrame | None = ...,
        *,
        axis: Axis | None = ...,
        limit: int = ...,
        inplace: Literal[False] = False,
    ) -> Self: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue | Mapping[HashableT2, ReplaceValue] = ...,
        value: ReplaceValue | Mapping[HashableT3, ReplaceValue] = ...,
        *,
        inplace: Literal[True],
        regex: ReplaceValue | Mapping[HashableT3, ReplaceValue] = ...,
    ) -> None: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue | Mapping[HashableT2, ReplaceValue] = ...,
        value: ReplaceValue | Mapping[HashableT3, ReplaceValue] = ...,
        *,
        inplace: Literal[False] = False,
        regex: ReplaceValue | Mapping[HashableT3, ReplaceValue] = ...,
    ) -> Self: ...
    def shift(
        self,
        periods: int | Sequence[int] = ...,
        freq: BaseOffset | dt.timedelta | _str | None = ...,
        axis: Axis | None = None,
        fill_value: Scalar | NAType | None = ...,
    ) -> Self: ...
    @overload
    def set_index(
        self,
        keys: (
            Label
            | Series
            | Index
            | np_ndarray
            | Iterator[Hashable]
            | Sequence[Hashable]
        ),
        *,
        drop: _bool = ...,
        append: _bool = ...,
        verify_integrity: _bool = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def set_index(
        self,
        keys: (
            Label
            | Series
            | Index
            | np_ndarray
            | Iterator[Hashable]
            | Sequence[Hashable]
        ),
        *,
        drop: _bool = ...,
        append: _bool = ...,
        verify_integrity: _bool = ...,
        inplace: Literal[False] = False,
    ) -> Self: ...
    @overload
    def reset_index(
        self,
        level: Level | Sequence[Level] = ...,
        *,
        drop: _bool = ...,
        col_level: int | _str = ...,
        col_fill: Hashable = ...,
        inplace: Literal[True],
        allow_duplicates: _bool = ...,
        names: Hashable | Sequence[Hashable] = ...,
    ) -> None: ...
    @overload
    def reset_index(
        self,
        level: Level | Sequence[Level] = ...,
        *,
        col_level: int | _str = ...,
        col_fill: Hashable = ...,
        drop: _bool = ...,
        inplace: Literal[False] = False,
        allow_duplicates: _bool = ...,
        names: Hashable | Sequence[Hashable] = ...,
    ) -> Self: ...
    def isna(self) -> Self: ...
    def isnull(self) -> Self: ...
    def notna(self) -> Self: ...
    def notnull(self) -> Self: ...
    @overload
    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll = ...,
        thresh: int | None = ...,
        subset: ListLikeU | Scalar | None = ...,
        inplace: Literal[True],
        ignore_index: _bool = ...,
    ) -> None: ...
    @overload
    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll = ...,
        thresh: int | None = ...,
        subset: ListLikeU | Scalar | None = ...,
        inplace: Literal[False] = False,
        ignore_index: _bool = ...,
    ) -> Self: ...
    @overload
    def drop_duplicates(
        self,
        subset: Hashable | Iterable[Hashable] | None = ...,
        *,
        keep: DropKeep = ...,
        inplace: Literal[True],
        ignore_index: _bool = ...,
    ) -> None: ...
    @overload
    def drop_duplicates(
        self,
        subset: Hashable | Iterable[Hashable] | None = ...,
        *,
        keep: DropKeep = ...,
        inplace: Literal[False] = False,
        ignore_index: _bool = ...,
    ) -> Self: ...
    def duplicated(
        self,
        subset: Hashable | Iterable[Hashable] | None = ...,
        keep: DropKeep = "first",
    ) -> Series: ...
    @overload
    def sort_values(
        self,
        by: _str | Sequence[_str],
        *,
        axis: Axis = 0,
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
        by: _str | Sequence[_str],
        *,
        axis: Axis = 0,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: _bool = ...,
        inplace: Literal[False] = False,
        key: ValueKeyFunc = ...,
    ) -> Self: ...
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = 0,
        level: Level | None = None,
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
        axis: Axis = 0,
        level: Level | list[int] | list[_str] | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        inplace: Literal[False] = False,
        key: IndexKeyFunc = ...,
    ) -> Self: ...
    @overload
    def value_counts(
        self,
        subset: Sequence[Hashable] | None = ...,
        normalize: Literal[False] = False,
        sort: _bool = ...,
        ascending: _bool = ...,
        dropna: _bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        subset: Sequence[Hashable] | None = ...,
        sort: _bool = ...,
        ascending: _bool = ...,
        dropna: _bool = ...,
    ) -> Series[float]: ...
    def nlargest(
        self,
        n: int,
        columns: _str | list[_str],
        keep: NsmallestNlargestKeep = "first",
    ) -> Self: ...
    def nsmallest(
        self,
        n: int,
        columns: _str | list[_str],
        keep: NsmallestNlargestKeep = "first",
    ) -> Self: ...
    def swaplevel(self, i: Level = ..., j: Level = ..., axis: Axis = 0) -> Self: ...
    def reorder_levels(self, order: list, axis: Axis = 0) -> Self: ...
    def compare(
        self,
        other: DataFrame,
        align_axis: Axis = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ...,
    ) -> Self: ...
    def combine(
        self,
        other: DataFrame,
        func: Callable,
        fill_value: Scalar | None = None,
        overwrite: _bool = True,
    ) -> Self: ...
    def combine_first(self, other: DataFrame) -> Self: ...
    def update(
        self,
        other: DataFrame | Series,
        join: UpdateJoin = "left",
        overwrite: _bool = True,
        filter_func: Callable | None = ...,
        errors: IgnoreRaise = "ignore",
    ) -> None: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: Scalar,
        level: IndexLabel | None = ...,
        as_index: Literal[True] = True,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Scalar, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: Scalar,
        level: IndexLabel | None = ...,
        as_index: Literal[False] = False,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Scalar, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: DatetimeIndex,
        level: IndexLabel | None = ...,
        as_index: Literal[True] = True,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Timestamp, Literal[True]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: DatetimeIndex,
        level: IndexLabel | None = ...,
        as_index: Literal[False] = False,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Timestamp, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: TimedeltaIndex,
        level: IndexLabel | None = ...,
        as_index: Literal[True] = True,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Timedelta, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: TimedeltaIndex,
        level: IndexLabel | None = ...,
        as_index: Literal[False] = False,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Timedelta, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: PeriodIndex,
        level: IndexLabel | None = ...,
        as_index: Literal[True] = True,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Period, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: PeriodIndex,
        level: IndexLabel | None = ...,
        as_index: Literal[False] = False,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Period, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: IntervalIndex[IntervalT],
        level: IndexLabel | None = ...,
        as_index: Literal[True] = True,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[IntervalT, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: IntervalIndex[IntervalT],
        level: IndexLabel | None = ...,
        as_index: Literal[False] = False,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[IntervalT, Literal[False]]: ...
    @overload
    def groupby(  # type: ignore[overload-overlap] # pyright: ignore reportOverlappingOverload
        self,
        by: MultiIndex | GroupByObjectNonScalar | None = ...,
        level: IndexLabel | None = ...,
        as_index: Literal[True] = True,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[tuple, Literal[True]]: ...
    @overload
    def groupby(  # type: ignore[overload-overlap]
        self,
        by: MultiIndex | GroupByObjectNonScalar | None = ...,
        level: IndexLabel | None = ...,
        as_index: Literal[False] = False,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[tuple, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: Series[SeriesByT],
        level: IndexLabel | None = ...,
        as_index: Literal[True] = True,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[SeriesByT, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: Series[SeriesByT],
        level: IndexLabel | None = ...,
        as_index: Literal[False] = False,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[SeriesByT, Literal[False]]: ...
    @overload
    def groupby(
        self,
        by: CategoricalIndex | Index | Series,
        level: IndexLabel | None = ...,
        as_index: Literal[True] = True,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Any, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: CategoricalIndex | Index | Series,
        level: IndexLabel | None = ...,
        as_index: Literal[False] = False,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy[Any, Literal[False]]: ...
    def pivot(
        self,
        *,
        columns: IndexLabel,
        index: IndexLabel = ...,
        values: IndexLabel = ...,
    ) -> Self: ...
    def pivot_table(
        self,
        values: _PivotTableValuesTypes = None,
        index: _PivotTableIndexTypes = None,
        columns: _PivotTableColumnsTypes = None,
        aggfunc: (
            _PivotAggFunc | Sequence[_PivotAggFunc] | Mapping[Hashable, _PivotAggFunc]
        ) = "mean",
        fill_value: Scalar | None = None,
        margins: _bool = False,
        dropna: _bool = True,
        margins_name: _str = "All",
        observed: _bool = True,
        sort: _bool = True,
    ) -> Self: ...
    @overload
    def stack(
        self,
        level: IndexLabel = ...,
        *,
        future_stack: Literal[True],
    ) -> Self | Series: ...
    @overload
    def stack(
        self,
        level: IndexLabel = ...,
        dropna: _bool = ...,
        sort: _bool = ...,
        future_stack: Literal[False] = False,
    ) -> Self | Series: ...
    def explode(
        self, column: Sequence[Hashable], ignore_index: _bool = False
    ) -> Self: ...
    def unstack(
        self,
        level: IndexLabel = -1,
        fill_value: Scalar | None = None,
        sort: _bool = True,
    ) -> Self | Series: ...
    def melt(
        self,
        id_vars: tuple | Sequence | np_ndarray | None = ...,
        value_vars: tuple | Sequence | np_ndarray | None = ...,
        var_name: Scalar | None = None,
        value_name: Scalar = "value",
        col_level: int | _str | None = ...,
        ignore_index: _bool = True,
    ) -> Self: ...
    def diff(self, periods: int = 1, axis: Axis = 0) -> Self: ...
    @overload
    def aggregate(  # pyright: ignore[reportOverlappingOverload]
        self,
        func: AggFuncTypeBase | AggFuncTypeDictSeries,
        axis: Axis = 0,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def aggregate(
        self,
        func: list[AggFuncTypeBase] | AggFuncTypeDictFrame | None = ...,
        axis: Axis = 0,
        **kwargs: Any,
    ) -> Self: ...
    agg = aggregate
    def transform(
        self,
        func: AggFuncTypeFrame,
        axis: Axis = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...

    # apply() overloads with default result_type of None, and is indifferent to axis
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Series],
        axis: AxisIndex = ...,
        raw: _bool = ...,
        result_type: None = None,
        args: Any = ...,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def apply(
        self,
        # Use S2 (TypeVar without `default=Any`) instead of S1 due to https://github.com/python/mypy/issues/19182.
        f: Callable[..., S2 | NAType],
        axis: AxisIndex = ...,
        raw: _bool = ...,
        result_type: None = None,
        args: Any = ...,
        **kwargs: Any,
    ) -> Series[S2]: ...
    # Since non-scalar type T is not supported in Series[T],
    # we separate this overload from the above one
    @overload
    def apply(
        self,
        f: Callable[..., Mapping[Any, Any]],
        axis: AxisIndex = ...,
        raw: _bool = ...,
        result_type: None = None,
        args: Any = ...,
        **kwargs: Any,
    ) -> Series: ...

    # apply() overloads with keyword result_type, and axis does not matter
    @overload
    def apply(
        self,
        # Use S2 (TypeVar without `default=Any`) instead of S1 due to https://github.com/python/mypy/issues/19182.
        f: Callable[..., S2 | NAType],
        axis: Axis = 0,
        raw: _bool = ...,
        args: Any = ...,
        *,
        result_type: Literal["expand", "reduce"],
        **kwargs: Any,
    ) -> Series[S2]: ...
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Series | Mapping[Any, Any]],
        axis: Axis = 0,
        raw: _bool = ...,
        args: Any = ...,
        *,
        result_type: Literal["expand"],
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Mapping[Any, Any]],
        axis: Axis = 0,
        raw: _bool = ...,
        args: Any = ...,
        *,
        result_type: Literal["reduce"],
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def apply(
        self,
        f: Callable[
            ..., ListLikeExceptSeriesAndStr | Series | Scalar | Mapping[Any, Any]
        ],
        axis: Axis = 0,
        raw: _bool = ...,
        args: Any = ...,
        *,
        result_type: Literal["broadcast"],
        **kwargs: Any,
    ) -> Self: ...

    # apply() overloads with keyword result_type, and axis does matter
    @overload
    def apply(
        self,
        f: Callable[..., Series],
        axis: AxisIndex = 0,
        raw: _bool = ...,
        args: Any = ...,
        *,
        result_type: Literal["reduce"],
        **kwargs: Any,
    ) -> Series: ...

    # apply() overloads with default result_type of None, and keyword axis=1 matters
    @overload
    def apply(
        self,
        # Use S2 (TypeVar without `default=Any`) instead of S1 due to https://github.com/python/mypy/issues/19182.
        f: Callable[..., S2 | NAType],
        raw: _bool = ...,
        result_type: None = None,
        args: Any = ...,
        *,
        axis: AxisColumn,
        **kwargs: Any,
    ) -> Series[S2]: ...
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Mapping[Any, Any]],
        raw: _bool = ...,
        result_type: None = None,
        args: Any = ...,
        *,
        axis: AxisColumn,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def apply(
        self,
        f: Callable[..., Series],
        raw: _bool = ...,
        result_type: None = None,
        args: Any = ...,
        *,
        axis: AxisColumn,
        **kwargs: Any,
    ) -> Self: ...

    # apply() overloads with keyword axis=1 and keyword result_type
    @overload
    def apply(
        self,
        f: Callable[..., Series],
        raw: _bool = ...,
        args: Any = ...,
        *,
        axis: AxisColumn,
        result_type: Literal["reduce"],
        **kwargs: Any,
    ) -> Self: ...

    # Add spacing between apply() overloads and remaining annotations
    def map(
        self, func: Callable, na_action: Literal["ignore"] | None = None, **kwargs: Any
    ) -> Self: ...
    def join(
        self,
        other: DataFrame | Series | list[DataFrame | Series],
        on: _str | list[_str] | None = ...,
        how: MergeHow = "left",
        lsuffix: _str = "",
        rsuffix: _str = "",
        sort: _bool = False,
        validate: JoinValidate | None = ...,
    ) -> Self: ...
    def merge(
        self,
        right: DataFrame | Series,
        how: MergeHow = "inner",
        on: IndexLabel | AnyArrayLike | None = ...,
        left_on: IndexLabel | AnyArrayLike | None = ...,
        right_on: IndexLabel | AnyArrayLike | None = ...,
        left_index: _bool = False,
        right_index: _bool = False,
        sort: _bool = False,
        suffixes: Suffixes = ...,
        copy: _bool = True,
        indicator: _bool | _str = False,
        validate: MergeValidate | None = None,
    ) -> Self: ...
    def round(
        self, decimals: int | dict | Series = ..., *args: Any, **kwargs: Any
    ) -> Self: ...
    def corr(
        self,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        min_periods: int = ...,
        numeric_only: _bool = False,
    ) -> Self: ...
    def cov(
        self,
        min_periods: int | None = None,
        ddof: int = 1,
        numeric_only: _bool = False,
    ) -> Self: ...
    def corrwith(
        self,
        other: DataFrame | Series,
        axis: Axis | None = 0,
        drop: _bool = False,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        numeric_only: _bool = False,
    ) -> Series: ...
    def count(self, axis: Axis = 0, numeric_only: _bool = False) -> Series[int]: ...
    def nunique(self, axis: Axis = 0, dropna: bool = True) -> Series[int]: ...
    def idxmax(
        self,
        axis: Axis = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
    ) -> Series[int]: ...
    def idxmin(
        self,
        axis: Axis = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
    ) -> Series[int]: ...
    def mode(
        self,
        axis: Axis = 0,
        numeric_only: _bool = False,
        dropna: _bool = True,
    ) -> Series: ...
    @overload
    def quantile(
        self,
        q: float = ...,
        axis: Axis = 0,
        numeric_only: _bool = ...,
        interpolation: QuantileInterpolation = ...,
        method: CalculationMethod = ...,
    ) -> Series: ...
    @overload
    def quantile(
        self,
        q: Sequence[float] | np_ndarray_float,
        axis: Axis = 0,
        numeric_only: _bool = ...,
        interpolation: QuantileInterpolation = ...,
        method: CalculationMethod = ...,
    ) -> Self: ...
    def to_timestamp(
        self,
        freq: PeriodFrequency | None = None,
        how: ToTimestampHow = ...,
        axis: Axis = 0,
        copy: _bool = True,
    ) -> Self: ...
    def to_period(
        self,
        freq: PeriodFrequency | None = None,
        axis: Axis = 0,
        copy: _bool = True,
    ) -> Self: ...
    def isin(
        self, values: Iterable[Any] | Mapping[Hashable, Iterable[Any]] | DataFrame
    ) -> Self: ...
    @property
    def plot(self) -> PlotAccessor: ...
    def hist(
        self,
        by: _str | ListLike | None = None,
        bins: int | list = 10,
        *,
        grid: _bool = True,
        xlabelsize: float | str | None = None,
        xrot: float | None = None,
        ylabelsize: float | str | None = None,
        yrot: float | None = None,
        ax: PlotAxes | None = None,
        sharex: _bool = False,
        sharey: _bool = False,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        backend: _str | None = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> npt.NDArray[np.object_]: ...

    # Keep in sync with `pd.plotting.boxplot`
    @overload
    def boxplot(
        self,
        by: None = None,
        ax: PlotAxes | None = None,
        fontsize: float | _str | None = None,
        rot: float = 0,
        grid: _bool = True,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        *,
        return_type: Literal["axes"] | None = None,
        backend: _str | None = None,
        **kwargs: Any,
    ) -> PlotAxes: ...
    @overload
    def boxplot(
        self,
        by: None = None,
        ax: PlotAxes | None = None,
        fontsize: float | _str | None = None,
        rot: float = 0,
        grid: _bool = True,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        *,
        return_type: Literal["dict"],
        backend: _str | None = None,
        **kwargs: Any,
    ) -> dict[str, PlotAxes]: ...
    @overload
    def boxplot(
        self,
        by: None = None,
        ax: PlotAxes | None = None,
        fontsize: float | _str | None = None,
        rot: float = 0,
        grid: _bool = True,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        *,
        return_type: Literal["both"],
        backend: _str | None = None,
        **kwargs: Any,
    ) -> _BoxPlotT: ...
    @overload
    def boxplot(
        self,
        by: Hashable | ListLikeHashable,
        ax: PlotAxes | None = None,
        fontsize: float | _str | None = None,
        rot: float = 0,
        grid: _bool = True,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        *,
        return_type: None = None,
        backend: _str | None = None,
        **kwargs: Any,
    ) -> PlotAxes: ...
    @overload
    def boxplot(
        self,
        by: Hashable | ListLikeHashable,
        ax: PlotAxes | None = None,
        fontsize: float | _str | None = None,
        rot: float = 0,
        grid: _bool = True,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        *,
        return_type: Literal["axes", "dict", "both"],
        backend: _str | None = None,
        **kwargs: Any,
    ) -> Series: ...

    sparse = ...

    # The rest of these are remnants from the
    # stubs shipped at preview. They may belong in
    # base classes, or stubgen just failed to generate
    # these.

    Name: _str
    #
    # dunder methods
    def __iter__(self) -> Iterator[Hashable]: ...
    # properties
    @property
    def at(self) -> _AtIndexerFrame: ...
    @property
    def columns(self) -> Index[str]: ...
    @columns.setter  # setter needs to be right next to getter; otherwise mypy complains
    def columns(
        self, cols: AnyArrayLike | SequenceNotStr[Hashable] | tuple[Hashable, ...]
    ) -> None: ...
    @property
    def dtypes(self) -> Series: ...
    @property
    def empty(self) -> _bool: ...
    @property
    def iat(self) -> _iAtIndexerFrame: ...
    @property
    def iloc(self) -> _iLocIndexerFrame[Self]: ...
    @property
    def index(self) -> Index: ...
    @index.setter
    def index(
        self, idx: AnyArrayLike | SequenceNotStr[Hashable] | tuple[Hashable, ...]
    ) -> None: ...
    @property
    def loc(self) -> _LocIndexerFrame[Self]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def values(self) -> np_2darray: ...
    # methods
    @final
    def abs(self) -> Self: ...
    def __add__(self, other: Any) -> Self: ...
    def add(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis | None = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def __radd__(self, other: Any) -> Self: ...
    def radd(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def __sub__(self, other: Any) -> Self: ...
    def sub(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    subtract = sub
    def __rsub__(self, other: Any) -> Self: ...
    def rsub(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def __mul__(self, other: Any) -> Self: ...
    def __rmul__(self, other: Any) -> Self: ...
    @final
    def add_prefix(self, prefix: _str, axis: Axis | None = None) -> Self: ...
    @final
    def add_suffix(self, suffix: _str, axis: Axis | None = None) -> Self: ...
    @overload
    def all(
        self,
        axis: None,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        **kwargs: Any,
    ) -> np.bool: ...
    @overload
    def all(
        self,
        axis: Axis = 0,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        **kwargs: Any,
    ) -> Series[_bool]: ...
    @overload
    def any(
        self,
        *,
        axis: None,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        **kwargs: Any,
    ) -> np.bool: ...
    @overload
    def any(
        self,
        *,
        axis: Axis = 0,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        **kwargs: Any,
    ) -> Series[_bool]: ...
    @final
    def asof(
        self,
        where: Scalar | AnyArrayLike | Sequence[Scalar],
        subset: Hashable | list[Hashable] | None = None,
    ) -> Self: ...
    @final
    def asfreq(
        self,
        freq: Frequency,
        method: FillnaOptions | None = None,
        how: Literal["start", "end"] | None = ...,
        normalize: _bool = False,
        fill_value: Scalar | None = ...,
    ) -> Self: ...
    @final
    def astype(
        self,
        dtype: AstypeArg | Mapping[Any, Dtype] | Series,
        copy: _bool = True,
        errors: IgnoreRaise = "raise",
    ) -> Self: ...
    @final
    def at_time(
        self,
        time: _str | dt.time,
        asof: _bool = False,
        axis: Axis | None = 0,
    ) -> Self: ...
    @final
    def between_time(
        self,
        start_time: _str | dt.time,
        end_time: _str | dt.time,
        inclusive: IntervalClosedType = "both",
        axis: Axis | None = 0,
    ) -> Self: ...
    @overload
    def bfill(
        self,
        *,
        axis: Axis | None = None,
        inplace: Literal[True],
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> None: ...
    @overload
    def bfill(
        self,
        *,
        axis: Axis | None = None,
        inplace: Literal[False] = False,
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: float | None = ...,
        upper: float | None = ...,
        *,
        axis: Axis | None = ...,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike = ...,
        upper: AnyArrayLike | None = ...,
        *,
        axis: Axis = ...,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike | None = ...,
        upper: AnyArrayLike = ...,
        *,
        axis: Axis = ...,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(  # pyright: ignore[reportOverlappingOverload]
        self,
        lower: None = None,
        upper: None = None,
        *,
        axis: Axis | None = ...,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: float | None = ...,
        upper: float | None = ...,
        *,
        axis: Axis | None = ...,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike = ...,
        upper: AnyArrayLike | None = ...,
        *,
        axis: Axis = ...,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike | None = ...,
        upper: AnyArrayLike = ...,
        *,
        axis: Axis = ...,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @final
    def copy(self, deep: _bool = True) -> Self: ...
    def cummax(
        self,
        axis: Axis | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    def cummin(
        self,
        axis: Axis | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    def cumprod(
        self,
        axis: Axis | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    def cumsum(
        self,
        axis: Axis | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    @final
    def describe(
        self,
        percentiles: list[float] | None = ...,
        include: Literal["all"] | list[Dtype] | None = ...,
        exclude: list[Dtype] | None = ...,
    ) -> Self: ...
    def truediv(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis | None = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    div = truediv
    divide = truediv
    @final
    def droplevel(self, level: Level | list[Level], axis: Axis = 0) -> Self: ...
    def eq(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
    ) -> Self: ...
    @final
    def equals(self, other: Series | DataFrame) -> _bool: ...
    @final
    def ewm(
        self,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | None = ...,
        alpha: float | None = ...,
        min_periods: int = 0,
        adjust: _bool = True,
        ignore_na: _bool = False,
        axis: Axis = 0,
    ) -> ExponentialMovingWindow[Self]: ...
    @final
    def expanding(
        self,
        min_periods: int = 1,
        axis: AxisIndex = 0,
        method: CalculationMethod = "single",
    ) -> Expanding[Self]: ...
    @overload
    def ffill(
        self,
        *,
        axis: Axis | None = ...,
        inplace: Literal[True],
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> None: ...
    @overload
    def ffill(
        self,
        *,
        axis: Axis | None = ...,
        inplace: Literal[False] = False,
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Self: ...
    def filter(
        self,
        items: ListLike | None = ...,
        like: _str | None = ...,
        regex: _str | None = ...,
        axis: Axis | None = None,
    ) -> Self: ...
    @final
    def first_valid_index(self) -> Scalar: ...
    def floordiv(
        self,
        other: float | ListLike | DataFrame,
        axis: Axis | None = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def ge(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
    ) -> Self: ...
    @overload
    def get(self, key: Hashable, default: None = None) -> Series | None: ...
    @overload
    def get(self, key: Hashable, default: _T) -> Series | _T: ...
    @overload
    def get(self, key: list[Hashable], default: None = None) -> Self | None: ...
    @overload
    def get(self, key: list[Hashable], default: _T) -> Self | _T: ...
    def gt(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
    ) -> Self: ...
    @final
    def head(self, n: int = 5) -> Self: ...
    @final
    def infer_objects(self, copy: _bool | None = ...) -> Self: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: Axis = 0,
        limit: int | None = ...,
        limit_direction: Literal["forward", "backward", "both"] = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: Axis = 0,
        limit: int | None = ...,
        limit_direction: Literal["forward", "backward", "both"] = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...
    def keys(self) -> Index: ...
    def kurt(
        self,
        axis: Axis | None = ...,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def kurtosis(
        self,
        axis: Axis | None = ...,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @final
    def last_valid_index(self) -> Scalar: ...
    def le(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
    ) -> Self: ...
    def lt(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
    ) -> Self: ...
    @overload
    def mask(
        self,
        cond: (
            Series
            | DataFrame
            | np_ndarray_bool
            | Callable[[DataFrame], DataFrame]
            | Callable[[Any], _bool]
        ),
        other: Scalar | Series | DataFrame | Callable | NAType | None = ...,
        *,
        inplace: Literal[True],
        axis: Axis | None = ...,
        level: Level | None = None,
    ) -> None: ...
    @overload
    def mask(
        self,
        cond: (
            Series
            | DataFrame
            | np_ndarray_bool
            | Callable[[DataFrame], DataFrame]
            | Callable[[Any], _bool]
        ),
        other: Scalar | Series | DataFrame | Callable | NAType | None = ...,
        *,
        inplace: Literal[False] = False,
        axis: Axis | None = ...,
        level: Level | None = None,
    ) -> Self: ...
    @overload
    def max(
        self,
        axis: None,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    @overload
    def max(
        self,
        axis: Axis = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def mean(
        self,
        axis: None,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    @overload
    def mean(
        self,
        axis: Axis = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def median(
        self,
        axis: None,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    @overload
    def median(
        self,
        axis: Axis = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def min(
        self,
        axis: None,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    @overload
    def min(
        self,
        axis: Axis = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def mod(
        self,
        other: float | ListLike | DataFrame,
        axis: Axis | None = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def mul(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis | None = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    multiply = mul
    def ne(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
    ) -> Self: ...
    @final
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq: Frequency | dt.timedelta | None = ...,
        fill_value: Scalar | NAType | None = ...,
        axis: Axis | None = ...,
    ) -> Self: ...
    def pop(self, item: _str) -> Series: ...
    def pow(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis | None = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def prod(
        self,
        axis: Axis | None = "columns",
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Series: ...
    product = prod
    @final
    def rank(
        self,
        axis: Axis = 0,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        numeric_only: _bool = False,
        na_option: Literal["keep", "top", "bottom"] = "keep",
        ascending: _bool = True,
        pct: _bool = False,
    ) -> Self: ...
    @final
    def reindex_like(
        self,
        other: DataFrame,
        method: FillnaOptions | Literal["nearest"] | None = ...,
        copy: _bool = True,
        limit: int | None = None,
        tolerance: Scalar | AnyArrayLike | Sequence[Scalar] = ...,
    ) -> Self: ...
    # Rename axis with `mapper`, `axis`, and `inplace=True`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: Axis | None = ...,
        copy: _bool = ...,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `mapper`, `axis`, and `inplace=False`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: Axis | None = ...,
        copy: _bool = ...,
        inplace: Literal[False] = False,
    ) -> Self: ...
    # Rename axis with `index` and/or `columns` and `inplace=True`
    @overload
    def rename_axis(
        self,
        *,
        index: _str | Sequence[_str] | dict[_str | int, _str] | Callable | None = ...,
        columns: _str | Sequence[_str] | dict[_str | int, _str] | Callable | None = ...,
        copy: _bool = ...,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `index` and/or `columns` and `inplace=False`
    @overload
    def rename_axis(
        self,
        *,
        index: _str | Sequence[_str] | dict[_str | int, _str] | Callable | None = ...,
        columns: _str | Sequence[_str] | dict[_str | int, _str] | Callable | None = ...,
        copy: _bool = ...,
        inplace: Literal[False] = False,
    ) -> Self: ...
    def rfloordiv(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def rmod(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def rmul(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    @overload
    def rolling(
        self,
        window: int | Frequency | dt.timedelta | BaseIndexer,
        min_periods: int | None = ...,
        center: _bool = ...,
        on: Hashable | None = ...,
        axis: AxisIndex = ...,
        closed: IntervalClosedType | None = ...,
        step: int | None = ...,
        method: CalculationMethod = ...,
        *,
        win_type: _str,
    ) -> Window[Self]: ...
    @overload
    def rolling(
        self,
        window: int | Frequency | dt.timedelta | BaseIndexer,
        min_periods: int | None = ...,
        center: _bool = ...,
        on: Hashable | None = ...,
        axis: AxisIndex = ...,
        closed: IntervalClosedType | None = ...,
        step: int | None = ...,
        method: CalculationMethod = ...,
        *,
        win_type: None = None,
    ) -> Rolling[Self]: ...
    def rpow(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def rtruediv(
        self,
        other: complex | ListLike | DataFrame,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    rdiv = rtruediv
    @final
    def sample(
        self,
        n: int | None = ...,
        frac: float | None = ...,
        replace: _bool = False,
        weights: _str | ListLike | None = ...,
        random_state: RandomState | None = ...,
        axis: Axis | None = None,
        ignore_index: _bool = False,
    ) -> Self: ...
    def sem(
        self,
        axis: Axis | None = ...,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    # Not actually positional, but used to handle removal of deprecated
    def set_axis(self, labels: AxesData, *, axis: Axis = 0) -> Self: ...
    def skew(
        self,
        axis: Axis | None = ...,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @final
    def squeeze(self, axis: Axis | None = None) -> DataFrame | Series | Scalar: ...
    def std(
        self,
        axis: Axis | None = 0,
        skipna: _bool = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def sum(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Series: ...
    @final
    def swapaxes(self, axis1: Axis, axis2: Axis, copy: _bool = ...) -> Self: ...
    @final
    def tail(self, n: int = 5) -> Self: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[str],
        *,
        orient: Literal["records"],
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: Literal[True],
        compression: CompressionOptions = ...,
        index: _bool | None = ...,
        indent: int | None = ...,
        storage_options: dict | None = ...,
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
        index: _bool | None = ...,
        indent: int | None = ...,
        storage_options: dict | None = ...,
        mode: Literal["a"],
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: None = None,
        *,
        orient: JsonFrameOrient | None = ...,
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: _bool = ...,
        compression: CompressionOptions = ...,
        index: _bool | None = ...,
        indent: int | None = ...,
        storage_options: dict | None = ...,
        mode: Literal["w"] = ...,
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes],
        *,
        orient: JsonFrameOrient | None = ...,
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: _bool = ...,
        compression: CompressionOptions = ...,
        index: _bool | None = ...,
        indent: int | None = ...,
        storage_options: dict | None = ...,
        mode: Literal["w"] = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        columns: SequenceNotStr[Hashable] | Index | Series | None = ...,
        col_space: int | list[int] | dict[HashableT, int] | None = ...,
        header: _bool | list[_str] | tuple[str, ...] = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        justify: _str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        line_width: int | None = ...,
        min_rows: int | None = ...,
        max_colwidth: int | None = ...,
        encoding: _str | None = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: None = None,
        *,
        columns: Sequence[Hashable] | Index | Series | None = ...,
        col_space: int | list[int] | dict[Hashable, int] | None = ...,
        header: _bool | Sequence[_str] = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        justify: _str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        line_width: int | None = ...,
        min_rows: int | None = ...,
        max_colwidth: int | None = ...,
        encoding: _str | None = ...,
    ) -> _str: ...
    @final
    def to_xarray(self) -> xr.Dataset: ...
    @final
    def truncate(
        self,
        before: dt.date | _str | int | None = ...,
        after: dt.date | _str | int | None = ...,
        axis: Axis | None = ...,
        copy: _bool = ...,
    ) -> Self: ...
    @final
    def tz_convert(
        self,
        tz: TimeZones,
        axis: Axis = 0,
        level: Level | None = None,
        copy: _bool = True,
    ) -> Self: ...
    @final
    def tz_localize(
        self,
        tz: TimeZones,
        axis: Axis = 0,
        level: Level | None = None,
        copy: _bool = True,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self: ...
    def var(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def where(
        self,
        cond: (
            Series
            | DataFrame
            | np_ndarray_bool
            | Callable[[DataFrame], DataFrame]
            | Callable[[Any], _bool]
        ),
        other: Scalar | Self | Callable[..., Scalar | Self] = ...,
        *,
        inplace: Literal[True],
        axis: Axis | None = ...,
        level: Level | None = None,
    ) -> None: ...
    @overload
    def where(
        self,
        cond: (
            Series
            | DataFrame
            | np_ndarray_bool
            | Callable[[DataFrame], DataFrame]
            | Callable[[Any], _bool]
        ),
        other: Scalar | Self | Callable[..., Scalar | Self] = ...,
        *,
        inplace: Literal[False] = False,
        axis: Axis | None = ...,
        level: Level | None = None,
    ) -> Self: ...
    # Move from generic because Series is Generic and it returns Series[bool] there
    @final
    def __invert__(self) -> Self: ...
    # floordiv overload
    def __floordiv__(
        self, other: float | DataFrame | Series[int] | Series[float] | Sequence[float]
    ) -> Self: ...
    def __rfloordiv__(
        self, other: float | DataFrame | Series[int] | Series[float] | Sequence[float]
    ) -> Self: ...
    def __truediv__(self, other: float | DataFrame | Series | Sequence) -> Self: ...
    def __rtruediv__(self, other: float | DataFrame | Series | Sequence) -> Self: ...
    @final
    def __bool__(self) -> NoReturn: ...

class _PandasNamedTuple(tuple[Any, ...]):
    def __getattr__(self, field: str) -> Scalar: ...
