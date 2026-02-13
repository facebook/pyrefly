from builtins import (
    bool as _bool,
    str as _str,
)
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
import datetime as dt
import sqlite3
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Literal,
    final,
    overload,
)

from pandas import Index
import pandas.core.indexing as indexing
from pandas.core.resample import DatetimeIndexResampler
from pandas.core.series import Series
import sqlalchemy.engine
from typing_extensions import (
    Never,
    Self,
)

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    Axis,
    CompressionOptions,
    CSVQuoting,
    DtypeArg,
    DtypeBackend,
    ExcelWriterMergeCells,
    FilePath,
    FileWriteMode,
    Frequency,
    HashableT1,
    HashableT2,
    HDFCompLib,
    IgnoreRaise,
    IndexLabel,
    Level,
    OpenFileErrors,
    P,
    StorageOptions,
    T,
    TakeIndexer,
    TimedeltaConvertibleTypes,
    TimeGrouperOrigin,
    TimestampConvertibleTypes,
    WriteBuffer,
    np_ndarray,
)

from pandas.io.pytables import HDFStore
from pandas.io.sql import SQLTable

class NDFrame(indexing.IndexingMixin):
    __hash__: ClassVar[None]  # type: ignore[assignment] # pyright: ignore[reportIncompatibleMethodOverride]

    @final
    def set_flags(
        self,
        *,
        copy: _bool = ...,
        allows_duplicate_labels: _bool | None = ...,
    ) -> Self: ...
    @property
    def attrs(self) -> dict[Hashable | None, Any]: ...
    @attrs.setter
    def attrs(self, value: Mapping[Hashable | None, Any]) -> None: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    def equals(self, other: Series) -> _bool: ...
    @final
    def __neg__(self) -> Self: ...
    @final
    def __pos__(self) -> Self: ...
    @final
    def __nonzero__(self) -> None: ...
    @final
    def bool(self) -> _bool: ...
    def __abs__(self) -> Self: ...
    @final
    def __round__(self, decimals: int = ...) -> Self: ...
    @final
    def __contains__(self, key) -> _bool: ...
    @property
    def empty(self) -> _bool: ...
    __array_priority__: int = ...
    def __array__(self, dtype=...) -> np_ndarray: ...
    @final
    def to_excel(
        self,
        excel_writer,
        sheet_name: _str = "Sheet1",
        na_rep: _str = "",
        float_format: _str | None = ...,
        columns: _str | Sequence[_str] | None = ...,
        header: _bool | list[_str] = True,
        index: _bool = True,
        index_label: _str | Sequence[_str] | None = ...,
        startrow: int = 0,
        startcol: int = 0,
        engine: _str | None = ...,
        merge_cells: ExcelWriterMergeCells = True,
        inf_rep: _str = "inf",
        freeze_panes: tuple[int, int] | None = ...,
    ) -> None: ...
    @final
    def to_hdf(
        self,
        path_or_buf: FilePath | HDFStore,
        *,
        key: _str,
        mode: Literal["a", "w", "r+"] = ...,
        complevel: int | None = ...,
        complib: HDFCompLib | None = ...,
        append: _bool = ...,
        format: Literal["t", "table", "f", "fixed"] | None = ...,
        index: _bool = ...,
        min_itemsize: int | dict[HashableT1, int] | None = ...,
        nan_rep: _str | None = ...,
        dropna: _bool | None = ...,
        data_columns: Literal[True] | list[HashableT2] | None = ...,
        errors: OpenFileErrors = ...,
        encoding: _str = ...,
    ) -> None: ...
    @overload
    def to_markdown(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        mode: FileWriteMode = ...,
        index: _bool = ...,
        storage_options: StorageOptions = ...,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def to_markdown(
        self,
        buf: None = None,
        *,
        mode: FileWriteMode | None = ...,
        index: _bool = ...,
        storage_options: StorageOptions = ...,
        **kwargs: Any,
    ) -> _str: ...
    @final
    def to_sql(
        self,
        name: _str,
        con: str | sqlalchemy.engine.Connectable | sqlite3.Connection,
        *,
        schema: _str | None = None,
        if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
        index: _bool = True,
        index_label: IndexLabel = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: (
            Literal["multi"]
            | Callable[
                [SQLTable, Any, list[str], Iterable[tuple[Any, ...]]],
                int | None,
            ]
            | None
        ) = None,
    ) -> int | None: ...
    @final
    def to_pickle(
        self,
        path: FilePath | WriteBuffer[bytes],
        compression: CompressionOptions = "infer",
        protocol: int = 5,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @final
    def to_clipboard(
        self,
        excel: _bool = True,
        sep: _str | None = None,
        *,
        na_rep: _str = ...,
        float_format: _str | Callable[[object], _str] | None = ...,
        columns: list[HashableT1] | None = ...,
        header: _bool | list[_str] = ...,
        index: _bool = ...,
        index_label: Literal[False] | _str | list[HashableT2] | None = ...,
        mode: FileWriteMode = ...,
        encoding: _str | None = ...,
        compression: CompressionOptions = ...,
        quoting: CSVQuoting = ...,
        quotechar: _str = ...,
        lineterminator: _str | None = ...,
        chunksize: int | None = ...,
        date_format: _str | None = ...,
        doublequote: _bool = ...,
        escapechar: _str | None = ...,
        decimal: _str = ...,
        errors: _str = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        columns: list[_str] | None = ...,
        header: _bool | list[_str] = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        bold_rows: _bool = ...,
        column_format: _str | None = ...,
        longtable: _bool | None = ...,
        escape: _bool | None = ...,
        encoding: _str | None = ...,
        decimal: _str = ...,
        multicolumn: _bool | None = ...,
        multicolumn_format: _str | None = ...,
        multirow: _bool | None = ...,
        caption: _str | tuple[_str, _str] | None = ...,
        label: _str | None = ...,
        position: _str | None = ...,
    ) -> None: ...
    @overload
    def to_latex(
        self,
        buf: None = None,
        columns: list[_str] | None = ...,
        header: _bool | list[_str] = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        bold_rows: _bool = ...,
        column_format: _str | None = ...,
        longtable: _bool | None = ...,
        escape: _bool | None = ...,
        encoding: _str | None = ...,
        decimal: _str = ...,
        multicolumn: _bool | None = ...,
        multicolumn_format: _str | None = ...,
        multirow: _bool | None = ...,
        caption: _str | tuple[_str, _str] | None = ...,
        label: _str | None = ...,
        position: _str | None = ...,
    ) -> _str: ...
    @overload
    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str],
        sep: _str = ...,
        na_rep: _str = ...,
        float_format: _str | Callable[[object], _str] | None = ...,
        columns: list[HashableT1] | None = ...,
        header: _bool | list[_str] = ...,
        index: _bool = ...,
        index_label: Literal[False] | _str | list[HashableT2] | None = ...,
        mode: FileWriteMode = ...,
        encoding: _str | None = ...,
        compression: CompressionOptions = ...,
        quoting: CSVQuoting = ...,
        quotechar: _str = ...,
        lineterminator: _str | None = ...,
        chunksize: int | None = ...,
        date_format: _str | None = ...,
        doublequote: _bool = ...,
        escapechar: _str | None = ...,
        decimal: _str = ...,
        errors: OpenFileErrors = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @overload
    def to_csv(
        self,
        path_or_buf: None = None,
        sep: _str = ...,
        na_rep: _str = ...,
        float_format: _str | Callable[[object], _str] | None = ...,
        columns: list[HashableT1] | None = ...,
        header: _bool | list[_str] = ...,
        index: _bool = ...,
        index_label: Literal[False] | _str | list[HashableT2] | None = ...,
        mode: FileWriteMode = ...,
        encoding: _str | None = ...,
        compression: CompressionOptions = ...,
        quoting: CSVQuoting = ...,
        quotechar: _str = ...,
        lineterminator: _str | None = ...,
        chunksize: int | None = ...,
        date_format: _str | None = ...,
        doublequote: _bool = ...,
        escapechar: _str | None = ...,
        decimal: _str = ...,
        errors: OpenFileErrors = ...,
        storage_options: StorageOptions = ...,
    ) -> _str: ...
    @final
    def __delitem__(self, idx: Hashable) -> None: ...
    @overload
    def drop(
        self,
        labels=...,
        *,
        axis=...,
        index: None,
        columns=...,
        level=...,
        inplace=...,
        errors=...,
    ) -> Never: ...
    @overload
    def drop(
        self,
        labels=...,
        *,
        axis=...,
        index=...,
        columns: None,
        level=...,
        inplace=...,
        errors=...,
    ) -> Never: ...
    @overload
    def drop(
        self,
        labels: None,
        *,
        axis=...,
        index=...,
        columns=...,
        level=...,
        inplace=...,
        errors=...,
    ) -> Never: ...
    @overload
    def drop(
        self,
        labels: None = None,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] | Index = ...,
        columns: Hashable | Iterable[Hashable],
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self,
        labels: None = None,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] | Index,
        columns: Hashable | Iterable[Hashable] = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self,
        labels: Hashable | Sequence[Hashable] | Index,
        *,
        axis: Axis = ...,
        index: None = None,
        columns: None = None,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self,
        labels: None = None,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] | Index = ...,
        columns: Hashable | Iterable[Hashable],
        level: Level | None = ...,
        inplace: Literal[False] = False,
        errors: IgnoreRaise = ...,
    ) -> Self: ...
    @overload
    def drop(
        self,
        labels: None = None,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] | Index,
        columns: Hashable | Iterable[Hashable] = ...,
        level: Level | None = ...,
        inplace: Literal[False] = False,
        errors: IgnoreRaise = ...,
    ) -> Self: ...
    @overload
    def drop(
        self,
        labels: Hashable | Sequence[Hashable] | Index,
        *,
        axis: Axis = ...,
        index: None = None,
        columns: None = None,
        level: Level | None = ...,
        inplace: Literal[False] = False,
        errors: IgnoreRaise = ...,
    ) -> Self: ...
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
    def __finalize__(self, other, method=..., **kwargs: Any) -> Self: ...
    @final
    def __setattr__(self, name: _str, value) -> None: ...
    @final
    def __copy__(self, deep: _bool = ...) -> Self: ...
    @final
    def __deepcopy__(self, memo: MutableMapping[int, Any] | None = None) -> Self: ...
    @final
    def convert_dtypes(
        self,
        infer_objects: _bool = True,
        convert_string: _bool = True,
        convert_integer: _bool = True,
        convert_boolean: _bool = True,
        convert_floating: _bool = True,
        dtype_backend: DtypeBackend = "numpy_nullable",
    ) -> Self: ...
    @final
    def resample(
        self,
        rule: Frequency | dt.timedelta,
        axis: Axis | _NoDefaultDoNotUse = 0,
        closed: Literal["right", "left"] | None = None,
        label: Literal["right", "left"] | None = None,
        on: Level | None = None,
        level: Level | None = None,
        origin: TimeGrouperOrigin | TimestampConvertibleTypes = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        group_keys: _bool = False,
    ) -> DatetimeIndexResampler[Self]: ...  # pyrefly: ignore[bad-specialization]
    @final
    def take(self, indices: TakeIndexer, axis: Axis = 0, **kwargs: Any) -> Self: ...
    def xs(
        self,
        key: IndexLabel,
        axis: Axis = 0,
        level: IndexLabel | None = None,
        drop_level: _bool = True,
    ) -> Self | Series: ...
