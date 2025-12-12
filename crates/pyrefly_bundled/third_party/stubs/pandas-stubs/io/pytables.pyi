from collections.abc import (
    Generator,
    Iterator,
    Sequence,
)
from types import TracebackType
from typing import (
    Any,
    Literal,
    overload,
)

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.computation.pytables import PyTablesExpr
from pandas.core.generic import NDFrame
from typing_extensions import Self

from pandas._typing import (
    BaseBuffer,
    FilePath,
    HashableT,
    HashableT1,
    HashableT2,
    HashableT3,
    HDFCompLib,
)

Term = PyTablesExpr

@overload
def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key: Any | None = ...,
    mode: Literal["r", "r+", "a"] = ...,
    errors: Literal[
        "strict",
        "ignore",
        "replace",
        "surrogateescape",
        "xmlcharrefreplace",
        "backslashreplace",
        "namereplace",
    ] = ...,
    where: str | Term | Sequence[Term] | None = ...,
    start: int | None = ...,
    stop: int | None = ...,
    columns: list[HashableT] | None = ...,
    *,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwargs: Any,
) -> TableIterator: ...
@overload
def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key: Any | None = ...,
    mode: Literal["r", "r+", "a"] = ...,
    errors: Literal[
        "strict",
        "ignore",
        "replace",
        "surrogateescape",
        "xmlcharrefreplace",
        "backslashreplace",
        "namereplace",
    ] = ...,
    where: str | Term | Sequence[Term] | None = ...,
    start: int | None = ...,
    stop: int | None = ...,
    columns: list[HashableT] | None = ...,
    iterator: bool = ...,
    *,
    chunksize: int,
    **kwargs: Any,
) -> TableIterator: ...
@overload
def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key: Any | None = ...,
    mode: Literal["r", "r+", "a"] = ...,
    errors: Literal[
        "strict",
        "ignore",
        "replace",
        "surrogateescape",
        "xmlcharrefreplace",
        "backslashreplace",
        "namereplace",
    ] = ...,
    where: str | Term | Sequence[Term] | None = ...,
    start: int | None = ...,
    stop: int | None = ...,
    columns: list[HashableT] | None = ...,
    iterator: Literal[False] = False,
    chunksize: None = None,
    **kwargs: Any,
) -> DataFrame | Series: ...

class HDFStore:
    def __init__(
        self,
        path: FilePath | BaseBuffer,
        mode: Literal["a", "w", "r", "r+"] = ...,
        complevel: int | None = ...,
        complib: HDFCompLib | None = ...,
        fletcher32: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def __fspath__(self) -> str: ...
    def __getitem__(self, key: str) -> DataFrame | Series: ...
    def __setitem__(self, key: str, value: DataFrame | Series) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __getattr__(self, name: str) -> DataFrame | Series: ...
    def __contains__(self, key: str) -> bool: ...
    def __len__(self) -> int: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def keys(self, include: Literal["pandas", "native"] = "pandas") -> list[str]: ...
    def __iter__(self) -> Iterator[str]: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...
    def get(self, key: str) -> DataFrame | Series: ...
    @overload
    def select(
        self,
        key: str,
        where: str | Term | Sequence[Term] | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
        columns: list[HashableT] | None = ...,
        *,
        iterator: Literal[True],
        chunksize: int | None = ...,
        auto_close: bool = ...,
    ) -> TableIterator: ...
    @overload
    def select(
        self,
        key: str,
        where: str | Term | Sequence[Term] | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
        columns: list[HashableT] | None = ...,
        iterator: bool = ...,
        *,
        chunksize: int,
        auto_close: bool = ...,
    ) -> TableIterator: ...
    @overload
    def select(
        self,
        key: str,
        where: str | Term | Sequence[Term] | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
        columns: list[HashableT] | None = ...,
        iterator: Literal[False] = False,
        chunksize: None = None,
        auto_close: bool = ...,
    ) -> DataFrame | Series: ...
    def put(
        self,
        key: str,
        value: NDFrame,
        format: Literal["t", "table", "f", "fixed"] | None = None,
        index: bool = True,
        append: bool = False,
        complib: HDFCompLib | None = None,
        complevel: int | None = None,
        min_itemsize: int | dict[HashableT1, int] | None = None,
        nan_rep: str | None = None,
        data_columns: Literal[True] | list[HashableT2] | None = None,
        encoding: str | None = None,
        errors: Literal[
            "strict",
            "ignore",
            "replace",
            "surrogateescape",
            "xmlcharrefreplace",
            "backslashreplace",
            "namereplace",
        ] = "strict",
        track_times: bool = True,
        dropna: bool = False,
    ) -> None: ...
    def append(
        self,
        key: str,
        value: NDFrame,
        format: Literal["t", "table", "f", "fixed"] | None = None,
        axes: int | None = None,
        index: bool = True,
        append: bool = True,
        complib: HDFCompLib | None = None,
        complevel: int | None = None,
        columns: list[HashableT1] | None = None,
        min_itemsize: int | dict[HashableT2, int] | None = None,
        nan_rep: str | None = None,
        chunksize: int | None = None,
        expectedrows: int | None = None,
        dropna: bool | None = False,
        data_columns: Literal[True] | list[HashableT3] | None = None,
        encoding: str | None = None,
        errors: Literal[
            "strict",
            "ignore",
            "replace",
            "surrogateescape",
            "xmlcharrefreplace",
            "backslashreplace",
            "namereplace",
        ] = "strict",
    ) -> None: ...
    def groups(self) -> list: ...
    def walk(
        self, where: str = "/"
    ) -> Generator[tuple[str, list, list[str]], None, None]: ...
    def info(self) -> str: ...

class TableIterator:
    def __iter__(self) -> Iterator[DataFrame | Series]: ...
    def close(self) -> None: ...
