from collections import defaultdict
from collections.abc import (
    Callable,
    Sequence,
)
import csv
from typing import (
    Any,
    Literal,
    overload,
)

from pandas.core.frame import DataFrame

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    CSVQuoting,
    DtypeArg,
    DtypeBackend,
    ListLikeHashable,
    StorageOptions,
    UsecolsArgType,
)

from pandas.io.parsers import TextFileReader

@overload
def read_clipboard(
    sep: str | None = ...,
    *,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | defaultdict | None = ...,
    engine: CSVEngine | None = ...,
    converters: dict[int | str, Callable[[str], Any]] = ...,
    true_values: list[str] = ...,
    false_values: list[str] = ...,
    skipinitialspace: bool = ...,
    skiprows: int | Sequence[int] | Callable[[int], bool] = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | dict[str, Sequence[str]] = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates: (
        bool
        | Sequence[int]
        | list[str]
        | Sequence[Sequence[int]]
        | dict[str, Sequence[int]]
    ) = ...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser: Callable = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
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
    dialect: str | csv.Dialect = ...,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = ...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy", "round_trip"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> TextFileReader: ...
@overload
def read_clipboard(
    sep: str | None = ...,
    *,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | defaultdict | None = ...,
    engine: CSVEngine | None = ...,
    converters: dict[int | str, Callable[[str], Any]] = ...,
    true_values: list[str] = ...,
    false_values: list[str] = ...,
    skipinitialspace: bool = ...,
    skiprows: int | Sequence[int] | Callable[[int], bool] = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | dict[str, Sequence[str]] = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates: (
        bool
        | Sequence[int]
        | list[str]
        | Sequence[Sequence[int]]
        | dict[str, Sequence[int]]
    ) = ...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser: Callable = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int,
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
    dialect: str | csv.Dialect = ...,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = ...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy", "round_trip"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> TextFileReader: ...
@overload
def read_clipboard(
    sep: str | None = ...,
    *,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | defaultdict | None = ...,
    engine: CSVEngine | None = ...,
    converters: dict[int | str, Callable[[str], Any]] = ...,
    true_values: list[str] = ...,
    false_values: list[str] = ...,
    skipinitialspace: bool = ...,
    skiprows: int | Sequence[int] | Callable[[int], bool] = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | dict[str, Sequence[str]] = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates: (
        bool
        | Sequence[int]
        | list[str]
        | Sequence[Sequence[int]]
        | dict[str, Sequence[int]]
    ) = ...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser: Callable = ...,
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
    dialect: str | csv.Dialect = ...,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = ...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy", "round_trip"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> DataFrame: ...
