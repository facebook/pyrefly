from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
from typing import (
    Any,
    final,
    overload,
)

import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index
from typing_extensions import Self

from pandas._typing import (
    AnyAll,
    Axes,
    DropKeep,
    Dtype,
    HashableT,
    IndexLabel,
    Level,
    MaskType,
    NaPosition,
    SequenceNotStr,
    np_1darray_bool,
    np_ndarray_anyint,
)

class MultiIndex(Index):
    def __new__(
        cls,
        levels: Sequence[SequenceNotStr[Hashable]] = ...,
        codes: Sequence[Sequence[int]] = ...,
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
        copy: bool = ...,
        name: SequenceNotStr[Hashable] = ...,
        verify_integrity: bool = ...,
    ) -> Self: ...
    @classmethod
    def from_arrays(
        cls,
        arrays: Sequence[Axes],
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
    ) -> Self: ...
    @classmethod
    def from_tuples(
        cls,
        tuples: Iterable[tuple[Hashable, ...]],
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
    ) -> Self: ...
    @classmethod
    def from_product(
        cls,
        iterables: Sequence[SequenceNotStr[Hashable] | pd.Series | pd.Index | range],
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
    ) -> Self: ...
    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
    ) -> Self: ...
    @property
    def shape(self): ...
    @property  # Should be read-only
    def levels(self) -> list[Index]: ...
    def set_levels(self, levels, *, level=..., verify_integrity: bool = ...): ...
    @property
    def codes(self): ...
    def set_codes(self, codes, *, level=..., verify_integrity: bool = ...): ...
    def copy(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore
        self, names: SequenceNotStr[Hashable] = ..., deep: bool = False
    ) -> Self: ...
    def view(self, cls=...): ...
    def __contains__(self, key) -> bool: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def dtypes(self) -> pd.Series[Dtype]: ...
    def memory_usage(self, deep: bool = False) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def format(
        self,
        name: bool | None = ...,
        formatter: Callable | None = ...,
        na_rep: str | None = ...,
        names: bool = ...,
        space: int = ...,
        sparsify: bool | None = ...,
        adjoin: bool = ...,
    ) -> list: ...
    def __len__(self) -> int: ...
    @property
    def values(self): ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    def duplicated(self, keep: DropKeep = "first"): ...
    def dropna(self, how: AnyAll = "any") -> Self: ...
    def droplevel(self, level: Level | Sequence[Level] = 0) -> MultiIndex | Index: ...  # type: ignore[override]
    def get_level_values(self, level: str | int) -> Index: ...
    def unique(self, level=...): ...
    def to_frame(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        index: bool = True,
        name: list[HashableT] = ...,
        allow_duplicates: bool = False,
    ) -> pd.DataFrame: ...
    def to_flat_index(self) -> Index: ...
    def remove_unused_levels(self): ...
    @property
    def nlevels(self) -> int: ...
    @property
    def levshape(self): ...
    def __reduce__(self): ...
    @overload  # type: ignore[override]
    # pyrefly: ignore  # bad-override
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index | MaskType,
    ) -> Self: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
        self, key: int
    ) -> tuple: ...
    def append(self, other): ...
    def repeat(self, repeats, axis=...): ...
    def drop(self, codes, level: Level | None = None, errors: str = "raise") -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def swaplevel(self, i: int = -2, j: int = -1) -> Self: ...
    def reorder_levels(self, order): ...
    def sortlevel(
        self,
        level: Level | Sequence[Level] = 0,
        ascending: bool = True,
        sort_remaining: bool = True,
        na_position: NaPosition = "first",
    ): ...
    @final
    def get_indexer(self, target, method=..., limit=..., tolerance=...): ...
    def get_indexer_non_unique(self, target): ...
    def reindex(self, target, method=..., level=..., limit=..., tolerance=...): ...
    def get_slice_bound(
        self, label: Hashable | Sequence[Hashable], side: str
    ) -> int: ...
    def get_loc_level(
        self, key, level: Level | list[Level] | None = None, drop_level: bool = True
    ): ...
    def get_locs(self, seq): ...
    def truncate(
        self, before: IndexLabel | None = None, after: IndexLabel | None = None
    ): ...
    def equals(self, other) -> bool: ...
    def equal_levels(self, other): ...
    def insert(self, loc, item): ...
    def delete(self, loc): ...
    @overload  # type: ignore[override]
    def isin(  # pyrefly: ignore[bad-override]
        self, values: Iterable[Any], level: Level
    ) -> np_1darray_bool: ...
    @overload
    def isin(  # ty: ignore[invalid-method-override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self, values: Collection[Iterable[Any]], level: None = None
    ) -> np_1darray_bool: ...
    def set_names(
        self,
        names: Hashable | Sequence[Hashable] | Mapping[Any, Hashable],
        *,
        level: Level | Sequence[Level] | None = None,
        inplace: bool = False,
    ) -> Self: ...
