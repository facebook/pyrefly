from typing import Any

from pandas import DataFrame

from pandas._typing import (
    DtypeBackend,
    FilePath,
    ParquetEngine,
    ReadBuffer,
    StorageOptions,
)

def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: ParquetEngine = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend = "numpy_nullable",
    filesystem: Any = None,
    filters: list[tuple] | list[list[tuple]] | None = None,
    **kwargs: Any,
) -> DataFrame: ...
