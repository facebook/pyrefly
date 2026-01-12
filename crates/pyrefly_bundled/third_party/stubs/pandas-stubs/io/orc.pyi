from typing import Any

from fsspec.spec import AbstractFileSystem  # pyright: ignore[reportMissingTypeStubs]
from pandas import DataFrame
from pyarrow.fs import FileSystem

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT,
    ReadBuffer,
)

def read_orc(
    path: FilePath | ReadBuffer[bytes],
    columns: list[HashableT] | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = "numpy_nullable",
    filesystem: FileSystem | AbstractFileSystem | None = None,
    **kwargs: Any,
) -> DataFrame: ...
