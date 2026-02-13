from collections import (
    abc,
    defaultdict,
)
from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
)
import csv
from types import TracebackType
from typing import (
    Any,
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
from typing_extensions import Self

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    CSVQuoting,
    DtypeArg,
    DtypeBackend,
    FilePath,
    HashableT,
    ListLikeHashable,
    ReadCsvBuffer,
    StorageOptions,
    UsecolsArgType,
)

@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = None,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = None,
    usecols: UsecolsArgType[HashableT] = None,
    dtype: DtypeArg | defaultdict | None = None,
    engine: CSVEngine | None = None,
    converters: (
        Mapping[int | str, Callable[[str], Any]]
        | Mapping[int, Callable[[str], Any]]
        | Mapping[str, Callable[[str], Any]]
        | None
    ) = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: (
        bool
        | list[int]
        | list[str]
        | Sequence[Sequence[int]]
        | Mapping[str, Sequence[int | str]]
        | None
    ) = None,
    keep_date_col: bool = True,
    date_format: dict[Hashable, str] | str | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: Literal[True],
    chunksize: int | None = None,
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: CSVQuoting = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> TextFileReader: ...
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = None,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = None,
    usecols: UsecolsArgType[HashableT] = None,
    dtype: DtypeArg | defaultdict | None = None,
    engine: CSVEngine | None = None,
    converters: (
        Mapping[int | str, Callable[[str], Any]]
        | Mapping[int, Callable[[str], Any]]
        | Mapping[str, Callable[[str], Any]]
        | None
    ) = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: (
        bool
        | list[int]
        | list[str]
        | Sequence[Sequence[int]]
        | Mapping[str, Sequence[int | str]]
        | None
    ) = None,
    keep_date_col: bool = False,
    date_format: dict[Hashable, str] | str | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: bool = False,
    chunksize: int,
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: CSVQuoting = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> TextFileReader: ...
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = ...,
    usecols: UsecolsArgType[HashableT] = ...,
    dtype: DtypeArg | defaultdict | None = ...,
    engine: CSVEngine | None = ...,
    converters: (
        Mapping[int | str, Callable[[str], Any]]
        | Mapping[int, Callable[[str], Any]]
        | Mapping[str, Callable[[str], Any]]
        | None
    ) = ...,
    true_values: list[str] | None = ...,
    false_values: list[str] | None = ...,
    skipinitialspace: bool = ...,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates: (
        bool
        | list[int]
        | list[str]
        | Sequence[Sequence[int]]
        | Mapping[str, Sequence[int | str]]
        | None
    ) = ...,
    keep_date_col: bool = False,
    date_format: dict[Hashable, str] | str | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[False] = False,
    chunksize: None = None,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: CSVQuoting = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = ...,
    low_memory: bool = True,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy", "round_trip"] | None = ...,
    storage_options: StorageOptions | None = ...,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> DataFrame: ...
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = None,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = None,
    usecols: UsecolsArgType[HashableT] = None,
    dtype: DtypeArg | defaultdict | None = None,
    engine: CSVEngine | None = None,
    converters: (
        Mapping[int | str, Callable[[str], Any]]
        | Mapping[int, Callable[[str], Any]]
        | Mapping[str, Callable[[str], Any]]
        | None
    ) = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: (
        bool
        | list[int]
        | list[str]
        | Sequence[Sequence[int]]
        | Mapping[str, Sequence[int | str]]
        | None
    ) = False,
    keep_date_col: bool = False,
    date_format: dict[Hashable, str] | str | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: Literal[True],
    chunksize: int | None = None,
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: CSVQuoting = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> TextFileReader: ...
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = None,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = None,
    usecols: UsecolsArgType[HashableT] = None,
    dtype: DtypeArg | defaultdict | None = None,
    engine: CSVEngine | None = None,
    converters: (
        Mapping[int | str, Callable[[str], Any]]
        | Mapping[int, Callable[[str], Any]]
        | Mapping[str, Callable[[str], Any]]
        | None
    ) = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: (
        bool
        | list[int]
        | list[str]
        | Sequence[Sequence[int]]
        | Mapping[str, Sequence[int | str]]
        | None
    ) = False,
    keep_date_col: bool = False,
    date_format: dict[Hashable, str] | str | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: bool = False,
    chunksize: int,
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: CSVQuoting = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> TextFileReader: ...
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = None,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = None,
    usecols: UsecolsArgType[HashableT] = None,
    dtype: DtypeArg | defaultdict | None = None,
    engine: CSVEngine | None = None,
    converters: (
        Mapping[int | str, Callable[[str], Any]]
        | Mapping[int, Callable[[str], Any]]
        | Mapping[str, Callable[[str], Any]]
        | None
    ) = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: (
        bool
        | list[int]
        | list[str]
        | Sequence[Sequence[int]]
        | Mapping[str, Sequence[int | str]]
        | None
    ) = False,
    keep_date_col: bool = False,
    date_format: dict[Hashable, str] | str | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: Literal[False] = False,
    chunksize: None = None,
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: CSVQuoting = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> DataFrame: ...
@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | Literal["infer"] | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwds: Any,
) -> TextFileReader: ...
@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | Literal["infer"] | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    iterator: bool = ...,
    chunksize: int,
    **kwds: Any,
) -> TextFileReader: ...
@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | Literal["infer"] | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    iterator: Literal[False] = False,
    chunksize: None = None,
    **kwds: Any,
) -> DataFrame: ...

class TextFileReader(abc.Iterator):
    engine: CSVEngine
    orig_options: Mapping[str, Any]
    chunksize: int | None
    nrows: int | None
    squeeze: bool
    def __init__(
        self,
        f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list,
        engine: CSVEngine | None = ...,
        **kwds: Any,
    ) -> None: ...
    def close(self) -> None: ...
    def read(self, nrows: int | None = ...) -> DataFrame: ...
    def get_chunk(self, size: int | None = ...) -> DataFrame: ...
    def __next__(self) -> DataFrame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
