from collections import abc
from collections.abc import Sequence
import datetime
from io import BytesIO
from types import TracebackType
from typing import (
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
from typing_extensions import Self

from pandas._typing import (
    CompressionOptions,
    FilePath,
    HashableT,
    HashableT1,
    HashableT2,
    HashableT3,
    ReadBuffer,
    StataDateFormat,
    StorageOptions,
    WriteBuffer,
)

@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = ...,
    convert_categoricals: bool = ...,
    index_col: str | None = ...,
    convert_missing: bool = ...,
    preserve_dtypes: bool = ...,
    columns: list[HashableT] | None = ...,
    order_categoricals: bool = ...,
    chunksize: int | None = ...,
    iterator: Literal[True],
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> StataReader: ...
@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = ...,
    convert_categoricals: bool = ...,
    index_col: str | None = ...,
    convert_missing: bool = ...,
    preserve_dtypes: bool = ...,
    columns: list[HashableT] | None = ...,
    order_categoricals: bool = ...,
    chunksize: int,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> StataReader: ...
@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = ...,
    convert_categoricals: bool = ...,
    index_col: str | None = ...,
    convert_missing: bool = ...,
    preserve_dtypes: bool = ...,
    columns: list[HashableT] | None = ...,
    order_categoricals: bool = ...,
    chunksize: None = None,
    iterator: Literal[False] = False,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...

class StataParser:
    def __init__(self) -> None: ...

class StataReader(StataParser, abc.Iterator):
    col_sizes: list[int] = ...
    path_or_buf: BytesIO = ...
    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        convert_dates: bool = ...,
        convert_categoricals: bool = ...,
        index_col: str | None = ...,
        convert_missing: bool = ...,
        preserve_dtypes: bool = ...,
        columns: Sequence[str] | None = ...,
        order_categoricals: bool = ...,
        chunksize: int | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def __next__(self) -> DataFrame: ...
    @property
    def data_label(self) -> str: ...
    def variable_labels(self) -> dict[str, str]: ...
    def value_labels(self) -> dict[str, dict[float, str]]: ...

class StataWriter(StataParser):
    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[HashableT1, StataDateFormat] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[HashableT2, str] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
        *,
        value_labels: dict[HashableT3, dict[float, str]] | None = ...,
    ) -> None: ...
    def write_file(self) -> None: ...
