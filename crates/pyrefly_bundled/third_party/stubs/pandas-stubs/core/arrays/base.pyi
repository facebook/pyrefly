from collections.abc import (
    Iterator,
    Sequence,
)
from typing import (
    Any,
    Literal,
    overload,
)

import numpy as np
from typing_extensions import Self

from pandas._typing import (
    AnyArrayLikeInt,
    ArrayLike,
    AstypeArg,
    Dtype,
    ListLike,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    TakeIndexer,
    np_1darray,
    np_1darray_intp,
    np_ndarray,
    npt,
)

from pandas.core.dtypes.dtypes import ExtensionDtype as ExtensionDtype

class ExtensionArray:
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any: ...
    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    def __setitem__(self, key: int | slice | np_ndarray, value: Any) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __contains__(self, item: object) -> bool | np.bool_: ...
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = ...,
        copy: bool = False,
        na_value: Scalar = ...,
    ) -> np_1darray: ...
    @property
    def dtype(self) -> ExtensionDtype: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @overload
    def astype(self, dtype: np.dtype, copy: bool = True) -> np_1darray: ...
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = True) -> ExtensionArray: ...
    @overload
    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike: ...
    def isna(self) -> ArrayLike: ...
    def argsort(
        self, *, ascending: bool = ..., kind: str = ..., **kwargs: Any
    ) -> np_1darray: ...
    def fillna(
        self, value: object | ArrayLike, limit: int | None = None, copy: bool = True
    ) -> Self: ...
    def dropna(self) -> Self: ...
    def shift(self, periods: int = 1, fill_value: object = ...) -> Self: ...
    def unique(self) -> Self: ...
    @overload
    def searchsorted(
        self,
        value: ListLike,
        side: Literal["left", "right"] = ...,
        sorter: ListLike | None = ...,
    ) -> np_1darray_intp: ...
    @overload
    def searchsorted(
        self,
        value: Scalar,
        side: Literal["left", "right"] = ...,
        sorter: ListLike | None = ...,
    ) -> np.intp: ...
    def factorize(self, use_na_sentinel: bool = True) -> tuple[np_1darray, Self]: ...
    def repeat(
        self, repeats: int | AnyArrayLikeInt | Sequence[int], axis: None = None
    ) -> Self: ...
    def take(
        self,
        indexer: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> Self: ...
    def copy(self) -> Self: ...
    @overload
    def view(self, dtype: None = None) -> Self: ...
    @overload
    def view(self, dtype: Dtype) -> ArrayLike: ...
    def ravel(self, order: Literal["C", "F", "A", "K"] | None = "C") -> Self: ...
    def tolist(self) -> list[Any]: ...
    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any
    ) -> object: ...
    def _accumulate(
        self,
        name: Literal["cummin", "cummax", "cumsum", "cumprod"],
        *,
        skipna: bool = True,
        **kwargs: Any,
    ) -> Self: ...

class ExtensionOpsMixin:
    @classmethod
    def _add_arithmetic_ops(cls) -> None: ...
    @classmethod
    def _add_comparison_ops(cls) -> None: ...
    @classmethod
    def _add_logical_ops(cls) -> None: ...

class ExtensionScalarOpsMixin(ExtensionOpsMixin): ...
