from collections.abc import (
    Hashable,
    Iterator,
    Sequence,
)
from datetime import timedelta
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    final,
    overload,
    type_check_only,
)

from _typeshed import _T_contra
import numpy as np
import numpy.typing as npt
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.indexes.accessors import ArrayDescriptor
from pandas.core.indexes.base import Index
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._typing import (
    S1,
    S2,
    AxisIndex,
    DropKeep,
    DTypeLike,
    GenericT,
    GenericT_co,
    Just,
    ListLike,
    NDFrameT,
    Scalar,
    SupportsDType,
    np_1darray,
    np_1darray_intp,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_float,
    np_ndarray_td,
)
from pandas.util._decorators import cache_readonly

T_INTERVAL_NP = TypeVar("T_INTERVAL_NP", bound=np.bytes_ | np.str_)

class NoNewAttributesMixin:
    def __setattr__(self, key: str, value: Any) -> None: ...

class SelectionMixin(Generic[NDFrameT]):
    obj: NDFrameT
    exclusions: frozenset[Hashable]
    @final
    @cache_readonly
    def ndim(self) -> int: ...
    def __getitem__(self, key): ...
    def aggregate(self, func, *args: Any, **kwargs: Any): ...

class IndexOpsMixin(OpsMixin, Generic[S1, GenericT_co]):
    __array_priority__: int = ...
    @property
    def T(self) -> Self: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    def item(self) -> S1: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    array = ArrayDescriptor()
    @overload
    def to_numpy(
        self,
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT_co]: ...
    @overload
    def to_numpy(
        self,
        dtype: np.dtype[GenericT] | SupportsDType[GenericT] | type[GenericT],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT]: ...
    @overload
    def to_numpy(
        self,
        dtype: DTypeLike,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    @property
    def empty(self) -> bool: ...
    def max(
        self, axis: AxisIndex | None = ..., skipna: bool = ..., **kwargs: Any
    ) -> S1: ...
    def min(
        self, axis: AxisIndex | None = ..., skipna: bool = ..., **kwargs: Any
    ) -> S1: ...
    def argmax(
        self,
        axis: AxisIndex | None = ...,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    def argmin(
        self,
        axis: AxisIndex | None = ...,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    def tolist(self) -> list[S1]: ...
    def to_list(self) -> list[S1]: ...
    def __iter__(self) -> Iterator[S1]: ...
    @property
    def hasnans(self) -> bool: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = False,
        sort: bool = ...,
        ascending: bool = ...,
        bins: int | None = ...,
        dropna: bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        bins: int | None = ...,
        dropna: bool = ...,
    ) -> Series[float]: ...
    def nunique(self, dropna: bool = True) -> int: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    def factorize(
        self, sort: bool = False, use_na_sentinel: bool = True
    ) -> tuple[np_1darray, np_1darray | Index | Categorical]: ...
    @overload
    def searchsorted(
        self,
        value: ListLike,
        side: Literal["left", "right"] = ...,
        sorter: ListLike | None = None,
    ) -> np_1darray_intp: ...
    @overload
    def searchsorted(
        self,
        value: Scalar,
        side: Literal["left", "right"] = ...,
        sorter: ListLike | None = None,
    ) -> np.intp: ...
    def drop_duplicates(self, *, keep: DropKeep = ...) -> Self: ...

ScalarArrayIndexJustInt: TypeAlias = (
    Just[int]
    | np.integer
    | Sequence[Just[int] | np.integer]
    | np_ndarray_anyint
    | IntegerArray
    | Index[int]
)
ScalarArrayIndexSeriesJustInt: TypeAlias = ScalarArrayIndexJustInt | Series[int]
ScalarArrayIndexJustFloat: TypeAlias = (
    Just[float]
    | np.floating
    | Sequence[Just[float] | np.floating]
    | np_ndarray_float
    # | FloatingArray  # TODO: after pandas-dev/pandas-stubs#1469
    | Index[float]
)
ScalarArrayIndexSeriesJustFloat: TypeAlias = ScalarArrayIndexJustFloat | Series[float]
ScalarArrayIndexJustComplex: TypeAlias = (
    Just[complex]
    | np.complexfloating
    | Sequence[Just[complex] | np.complexfloating]
    | np_ndarray_complex
    | Index[complex]
)
ScalarArrayIndexSeriesJustComplex: TypeAlias = (
    ScalarArrayIndexJustComplex | Series[complex]
)

ScalarArrayIndexIntNoBool: TypeAlias = (
    Just[int]
    | np.integer
    | Sequence[int | np.integer]
    | np_ndarray_anyint
    | IntegerArray
    | Index[int]
)
ScalarArrayIndexSeriesIntNoBool: TypeAlias = ScalarArrayIndexIntNoBool | Series[int]

NumpyRealScalar: TypeAlias = np.bool | np.integer | np.floating
IndexReal: TypeAlias = Index[bool] | Index[int] | Index[float]
ScalarArrayIndexReal: TypeAlias = (
    float
    | Sequence[float | NumpyRealScalar]
    | NumpyRealScalar
    | npt.NDArray[NumpyRealScalar]
    | ExtensionArray
    | IndexReal
)
SeriesReal: TypeAlias = Series[bool] | Series[int] | Series[float]
ScalarArrayIndexSeriesReal: TypeAlias = ScalarArrayIndexReal | SeriesReal

NumpyComplexScalar: TypeAlias = NumpyRealScalar | np.complexfloating
IndexComplex: TypeAlias = IndexReal | Index[complex]
ScalarArrayIndexComplex: TypeAlias = (
    complex
    | Sequence[complex | NumpyComplexScalar]
    | NumpyComplexScalar
    | npt.NDArray[NumpyComplexScalar]
    | ExtensionArray
    | IndexComplex
)
SeriesComplex: TypeAlias = SeriesReal | Series[complex]
ScalarArrayIndexSeriesComplex: TypeAlias = ScalarArrayIndexComplex | SeriesComplex

ArrayIndexTimedeltaNoSeq: TypeAlias = np_ndarray_td | TimedeltaArray | TimedeltaIndex
ScalarArrayIndexTimedelta: TypeAlias = (
    timedelta
    | np.timedelta64
    | Sequence[timedelta | np.timedelta64]
    | ArrayIndexTimedeltaNoSeq
)
ArrayIndexSeriesTimedeltaNoSeq: TypeAlias = ArrayIndexTimedeltaNoSeq | Series[Timedelta]
ScalarArrayIndexSeriesTimedelta: TypeAlias = (
    ScalarArrayIndexTimedelta | Series[Timedelta]
)

NumListLike: TypeAlias = (  # TODO: pandas-dev/pandas-stubs#1474 deprecated, do not use
    ExtensionArray
    | np_ndarray_bool
    | np_ndarray_anyint
    | np_ndarray_float
    | np_ndarray_complex
    | dict[str, np_ndarray]
    | Sequence[complex]
)

@type_check_only
class ElementOpsMixin(Generic[S2]):
    @overload
    def _proto_add(
        self: ElementOpsMixin[bool], other: bool | np.bool_
    ) -> ElementOpsMixin[bool]: ...
    @overload
    def _proto_add(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_add(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_add(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_add(self: ElementOpsMixin[str], other: str) -> ElementOpsMixin[str]: ...
    @overload
    def _proto_radd(
        self: ElementOpsMixin[bool], other: bool | np.bool_
    ) -> ElementOpsMixin[bool]: ...
    @overload
    def _proto_radd(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_radd(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_radd(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_radd(self: ElementOpsMixin[str], other: str) -> ElementOpsMixin[str]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[bool], other: bool | np.bool_
    ) -> ElementOpsMixin[bool]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[Timedelta],
        other: Just[int] | Just[float] | np.integer | np.floating,
    ) -> ElementOpsMixin[Timedelta]: ...
    @overload
    def _proto_mul(
        self: ElementOpsMixin[str], other: Just[int] | np.integer
    ) -> ElementOpsMixin[str]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[bool], other: bool | np.bool_
    ) -> ElementOpsMixin[bool]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[Timedelta],
        other: Just[int] | Just[float] | np.integer | np.floating,
    ) -> ElementOpsMixin[Timedelta]: ...
    @overload
    def _proto_rmul(
        self: ElementOpsMixin[str], other: Just[int] | np.integer
    ) -> ElementOpsMixin[str]: ...
    @overload
    def _proto_truediv(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_truediv(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_truediv(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_truediv(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rtruediv(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rtruediv(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rtruediv(
        self: ElementOpsMixin[complex], other: complex | np.complexfloating
    ) -> ElementOpsMixin[complex]: ...
    @overload
    def _proto_rtruediv(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_floordiv(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_floordiv(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_floordiv(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_rfloordiv(
        self: ElementOpsMixin[int], other: int | np.integer
    ) -> ElementOpsMixin[int]: ...
    @overload
    def _proto_rfloordiv(
        self: ElementOpsMixin[float], other: float | np.floating
    ) -> ElementOpsMixin[float]: ...
    @overload
    def _proto_rfloordiv(
        self: ElementOpsMixin[Timedelta], other: timedelta | np.timedelta64 | Timedelta
    ) -> ElementOpsMixin[int]: ...

@type_check_only
class Supports_ProtoAdd(Protocol[_T_contra, S2]):
    def _proto_add(self, other: _T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRAdd(Protocol[_T_contra, S2]):
    def _proto_radd(self, other: _T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoMul(Protocol[_T_contra, S2]):
    def _proto_mul(self, other: _T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRMul(Protocol[_T_contra, S2]):
    def _proto_rmul(self, other: _T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoTrueDiv(Protocol[_T_contra, S2]):
    def _proto_truediv(self, other: _T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRTrueDiv(Protocol[_T_contra, S2]):
    def _proto_rtruediv(self, other: _T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoFloorDiv(Protocol[_T_contra, S2]):
    def _proto_floordiv(self, other: _T_contra, /) -> ElementOpsMixin[S2]: ...

@type_check_only
class Supports_ProtoRFloorDiv(Protocol[_T_contra, S2]):
    def _proto_rfloordiv(self, other: _T_contra, /) -> ElementOpsMixin[S2]: ...
