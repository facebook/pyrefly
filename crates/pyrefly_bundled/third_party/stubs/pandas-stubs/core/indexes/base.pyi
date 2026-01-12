from builtins import str as _str
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Sequence,
)
from datetime import (
    datetime,
    timedelta,
)
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeAlias,
    final,
    overload,
    type_check_only,
)

from _typeshed import (
    SupportsAdd,
    SupportsMul,
    SupportsRAdd,
    SupportsRMul,
    _T_contra,
)
import numpy as np
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.base import (
    T_INTERVAL_NP,
    ArrayIndexTimedeltaNoSeq,
    ElementOpsMixin,
    IndexComplex,
    IndexOpsMixin,
    IndexReal,
    ScalarArrayIndexComplex,
    ScalarArrayIndexIntNoBool,
    ScalarArrayIndexJustComplex,
    ScalarArrayIndexJustFloat,
    ScalarArrayIndexReal,
    ScalarArrayIndexTimedelta,
    Supports_ProtoAdd,
    Supports_ProtoFloorDiv,
    Supports_ProtoMul,
    Supports_ProtoRAdd,
    Supports_ProtoRFloorDiv,
    Supports_ProtoRMul,
    Supports_ProtoRTrueDiv,
    Supports_ProtoTrueDiv,
)
from pandas.core.frame import DataFrame
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from pandas.core.strings.accessor import StringMethods
from typing_extensions import (
    Never,
    Self,
)

from pandas._libs.interval import (
    Interval,
    _OrderableT,
)
from pandas._libs.tslibs.period import Period
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._typing import (
    C2,
    S1,
    S2,
    S2_NSDT,
    T_COMPLEX,
    AnyAll,
    AnyArrayLike,
    AnyArrayLikeInt,
    ArrayLike,
    AxesData,
    Axis,
    BuiltinFloatDtypeArg,
    CategoryDtypeArg,
    DropKeep,
    Dtype,
    DtypeArg,
    DTypeLike,
    DtypeObj,
    GenericT,
    GenericT_co,
    HashableT,
    IgnoreRaise,
    JoinHow,
    Just,
    Label,
    Level,
    MaskType,
    NaPosition,
    NumpyFloatNot16DtypeArg,
    PandasAstypeFloatDtypeArg,
    PandasFloatDtypeArg,
    PyArrowFloatDtypeArg,
    ReindexMethod,
    Renamer,
    S2_contra,
    Scalar,
    SequenceNotStr,
    SliceType,
    SupportsDType,
    TakeIndexer,
    TimedeltaDtypeArg,
    TimestampDtypeArg,
    np_1darray,
    np_1darray_bool,
    np_1darray_intp,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_dt,
    np_ndarray_float,
    np_ndarray_str,
    np_ndarray_td,
    type_t,
)

from pandas.core.dtypes.dtypes import PeriodDtype

FloatNotNumpy16DtypeArg: TypeAlias = (
    BuiltinFloatDtypeArg
    | PandasFloatDtypeArg
    | NumpyFloatNot16DtypeArg
    | PyArrowFloatDtypeArg
)

class InvalidIndexError(Exception): ...

class Index(IndexOpsMixin[S1], ElementOpsMixin[S1]):
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]
    # overloads with additional dtypes
    @overload
    def __new__(  # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[bool | np.bool_] | IndexOpsMixin[bool] | np_ndarray_bool,
        *,
        dtype: Literal["bool"] | type_t[bool | np.bool_] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[bool]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[int | np.integer] | IndexOpsMixin[int] | np_ndarray_anyint,
        *,
        dtype: Literal["int"] | type_t[int | np.integer] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[int]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["int"] | type_t[int | np.integer],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[int]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[float | np.floating] | np_ndarray_float | FloatingArray,
        dtype: None = None,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[float]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        dtype: FloatNotNumpy16DtypeArg,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[float]: ...
    @overload
    def __new__(
        cls,
        data: (
            Sequence[complex | np.complexfloating]
            | IndexOpsMixin[complex]
            | np_ndarray_complex
        ),
        *,
        dtype: Literal["complex"] | type_t[complex | np.complexfloating] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[complex]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["complex"] | type_t[complex | np.complexfloating],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Index[complex]: ...
    # special overloads with dedicated Index-subclasses
    @overload
    def __new__(
        cls,
        data: (
            Sequence[np.datetime64 | datetime] | IndexOpsMixin[datetime] | DatetimeIndex
        ),
        *,
        dtype: TimestampDtypeArg = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> DatetimeIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: TimestampDtypeArg,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> DatetimeIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[Period] | IndexOpsMixin[Period],
        *,
        dtype: PeriodDtype = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> PeriodIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: PeriodDtype,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> PeriodIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[np.timedelta64 | timedelta] | IndexOpsMixin[timedelta],
        *,
        dtype: TimedeltaDtypeArg = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> TimedeltaIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: TimedeltaDtypeArg,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> TimedeltaIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: CategoryDtypeArg,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> CategoricalIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[Interval[_OrderableT]] | IndexOpsMixin[Interval[_OrderableT]],
        *,
        dtype: Literal["Interval"] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> IntervalIndex[Interval[_OrderableT]]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["Interval"],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> IntervalIndex[Interval[Any]]: ...
    @overload
    def __new__(
        cls,
        data: DatetimeIndex,
        *,
        dtype: TimestampDtypeArg | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> DatetimeIndex: ...
    # generic overloads
    @overload
    def __new__(
        cls,
        data: Iterable[S1] | IndexOpsMixin[S1],
        *,
        dtype: type[S1] = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: AxesData = ...,
        *,
        dtype: type[S1],
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Self: ...
    # fallback overload
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Dtype = ...,
        copy: bool = ...,
        name: Hashable = ...,
        tupleize_cols: bool = ...,
    ) -> Self: ...
    @property
    def str(
        self,
    ) -> StringMethods[  # pyrefly: ignore[bad-specialization]
        Self,
        MultiIndex,
        np_1darray_bool,
        Index[list[_str]],
        Index[int],
        Index[bytes],
        Index[_str],
        Index,
    ]: ...
    @final
    def is_(self, other: Any) -> bool: ...
    def __len__(self) -> int: ...
    def __array__(
        self, dtype: _str | np.dtype = ..., copy: bool | None = ...
    ) -> np_1darray: ...
    @property
    def dtype(self) -> DtypeObj: ...
    @final
    def ravel(self, order: _str = "C") -> Self: ...
    def view(self, cls=...): ...
    @overload
    def astype(
        self,
        dtype: FloatNotNumpy16DtypeArg | PandasAstypeFloatDtypeArg,
        copy: bool = True,
    ) -> Index[float]: ...
    @overload
    def astype(self, dtype: DtypeArg, copy: bool = True) -> Index: ...
    def take(
        self,
        indices: TakeIndexer,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value: Scalar | None = None,
        **kwargs: Any,
    ) -> Self: ...
    def repeat(
        self, repeats: int | AnyArrayLikeInt | Sequence[int], axis: None = None
    ) -> Self: ...
    def copy(self, name: Hashable = ..., deep: bool = False) -> Self: ...
    def format(
        self, name: bool = ..., formatter: Callable | None = ..., na_rep: _str = ...
    ) -> list[_str]: ...
    def to_series(
        self, index: Index | None = None, name: Hashable | None = None
    ) -> Series[S1]: ...
    def to_frame(self, index: bool = True, name: Hashable = ...) -> DataFrame: ...
    @property
    def name(self) -> Hashable | None: ...
    @name.setter
    def name(self, value: Hashable) -> None: ...
    @property
    def names(self) -> list[Hashable | None]: ...
    @names.setter
    def names(self, names: SequenceNotStr[Hashable | None]) -> None: ...
    def set_names(
        self,
        names: Hashable | Sequence[Hashable],
        *,
        level: Level | Sequence[Level] | None = None,
        inplace: bool = False,
    ) -> Self: ...
    @overload
    def rename(self, name: Hashable, *, inplace: Literal[False] = False) -> Self: ...
    @overload
    def rename(self, name: Hashable, *, inplace: Literal[True]) -> None: ...
    @property
    def nlevels(self) -> int: ...
    def get_level_values(self, level: int | _str) -> Index: ...
    def droplevel(self, level: Level | Sequence[Level] = 0) -> Self: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def has_duplicates(self) -> bool: ...
    @property
    def inferred_type(self) -> _str: ...
    @property
    def hasnans(self) -> bool: ...
    @final
    def isna(self) -> Index[bool]: ...
    @final
    def notna(self) -> Index[bool]: ...
    def fillna(self, value: Scalar) -> Index: ...
    def dropna(self, how: AnyAll = "any") -> Self: ...
    def unique(self, level: Hashable | None = None) -> Self: ...
    def drop_duplicates(self, *, keep: DropKeep = ...) -> Self: ...
    def duplicated(self, keep: DropKeep = "first") -> np_1darray_bool: ...
    def __and__(self, other: Never) -> Never: ...
    def __rand__(self, other: Never) -> Never: ...
    def __or__(self, other: Never) -> Never: ...
    def __ror__(self, other: Never) -> Never: ...
    def __xor__(self, other: Never) -> Never: ...
    def __rxor__(self, other: Never) -> Never: ...
    def __neg__(self) -> Self: ...
    @final
    def __nonzero__(self) -> None: ...
    __bool__ = ...
    def union(
        self, other: list[HashableT] | Self, sort: bool | None = None
    ) -> Self: ...
    def intersection(
        self, other: list[S1] | Self, sort: bool | None = False
    ) -> Self: ...
    def difference(self, other: list | Self, sort: bool | None = None) -> Self: ...
    def symmetric_difference(
        self,
        other: list[S1] | Self,
        result_name: Hashable = ...,
        sort: bool | None = None,
    ) -> Self: ...
    def get_loc(self, key: Label) -> int | slice | np_1darray_bool: ...
    def get_indexer(
        self,
        target: Index,
        method: ReindexMethod | None = None,
        limit: int | None = None,
        tolerance: Scalar | AnyArrayLike | Sequence[Scalar] | None = None,
    ) -> np_1darray_intp: ...
    def reindex(
        self,
        target: Iterable[Any],
        method: ReindexMethod | None = None,
        level: int | None = None,
        limit: int | None = None,
        tolerance: Scalar | AnyArrayLike | Sequence[Scalar] | None = None,
    ) -> tuple[Index, np_1darray_intp | None]: ...
    @overload
    def join(
        self,
        other: Index,
        *,
        how: JoinHow = "left",
        level: Level | None = None,
        return_indexers: Literal[True],
        sort: bool = False,
    ) -> tuple[Index, np_1darray_intp | None, np_1darray_intp | None]: ...
    @overload
    def join(
        self,
        other: Index,
        *,
        how: JoinHow = "left",
        level: Level | None = None,
        return_indexers: Literal[False] = False,
        sort: bool = False,
    ) -> Index: ...
    @property
    def values(self) -> np_1darray: ...
    def memory_usage(self, deep: bool = False) -> int: ...
    @overload
    def where(
        self,
        cond: Sequence[bool] | np_ndarray_bool | BooleanArray | IndexOpsMixin[bool],
        other: S1 | Series[S1] | Self,
    ) -> Self: ...
    @overload
    def where(
        self,
        cond: Sequence[bool] | np_ndarray_bool | BooleanArray | IndexOpsMixin[bool],
        other: Scalar | AnyArrayLike | None = None,
    ) -> Index: ...
    def __contains__(self, key: Hashable) -> bool: ...
    @overload
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index | MaskType,
    ) -> Self: ...
    @overload
    def __getitem__(self, idx: int | tuple[np_ndarray_anyint, ...]) -> S1: ...
    @overload
    def append(
        self: Index[C2], other: Index[C2] | Sequence[Index[C2]]
    ) -> Index[C2]: ...
    @overload
    def append(self, other: Index | Sequence[Index]) -> Index: ...
    def putmask(
        self,
        mask: Sequence[bool] | np_ndarray_bool | BooleanArray | IndexOpsMixin[bool],
        value: Scalar,
    ) -> Index: ...
    def equals(self, other: Any) -> bool: ...
    @final
    def identical(self, other: Any) -> bool: ...
    @final
    def asof(self, label: Scalar) -> Scalar: ...
    def asof_locs(
        self, where: DatetimeIndex, mask: np_ndarray_bool
    ) -> np_1darray_intp: ...
    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[False] = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Callable[[Index], Index] | None = None,
    ) -> Self: ...
    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Callable[[Index], Index] | None = None,
    ) -> tuple[Self, np_1darray_intp]: ...
    @final
    def sort(self, *args: Any, **kwargs: Any) -> None: ...
    def argsort(self, *args: Any, **kwargs: Any) -> np_1darray_intp: ...
    def get_indexer_non_unique(
        self, target: Index
    ) -> tuple[np_1darray_intp, np_1darray_intp]: ...
    @final
    def get_indexer_for(self, target: Index) -> np_1darray_intp: ...
    def map(
        self, mapper: Renamer, na_action: Literal["ignore"] | None = None
    ) -> Index: ...
    def isin(
        self, values: Iterable[Any], level: Level | None = None
    ) -> np_1darray_bool: ...
    def slice_indexer(
        self,
        start: Label | None = None,
        end: Label | None = None,
        step: int | None = None,
    ) -> slice: ...
    def get_slice_bound(self, label: Scalar, side: Literal["left", "right"]) -> int: ...
    def slice_locs(
        self, start: SliceType = None, end: SliceType = None, step: int | None = None
    ) -> tuple[int | np.intp, int | np.intp]: ...
    def delete(
        self, loc: np.integer | int | AnyArrayLikeInt | Sequence[int]
    ) -> Self: ...
    @overload
    def insert(self, loc: int, item: S1) -> Self: ...
    @overload
    def insert(self, loc: int, item: object) -> Index: ...
    def drop(self, labels, errors: IgnoreRaise = "raise") -> Self: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    # Extra methods from old stubs
    def __eq__(self, other: object) -> np_1darray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __ne__(self, other: object) -> np_1darray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __le__(self, other: Self | S1) -> np_1darray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __ge__(self, other: Self | S1) -> np_1darray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __lt__(self, other: Self | S1) -> np_1darray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __gt__(self, other: Self | S1) -> np_1darray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    @overload
    def __add__(self: Index[Never], other: _str) -> Never: ...
    @overload
    def __add__(
        self: Index[Never], other: complex | ArrayLike | SequenceNotStr[S1] | Index
    ) -> Index: ...
    @overload
    def __add__(self, other: Index[Never]) -> Index: ...
    @overload
    def __add__(self: Index[Never], other: Period) -> PeriodIndex: ...
    @overload
    def __add__(
        self: Supports_ProtoAdd[_T_contra, S2], other: _T_contra | Sequence[_T_contra]
    ) -> Index[S2]: ...
    @overload
    def __add__(
        self: Index[S2_contra],
        other: SupportsRAdd[S2_contra, S2] | Sequence[SupportsRAdd[S2_contra, S2]],
    ) -> Index[S2]: ...
    @overload
    def __add__(
        self: Index[T_COMPLEX], other: np_ndarray_bool | Index[bool]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(
        self: Index[bool], other: np_ndarray_anyint | Index[int]
    ) -> Index[int]: ...
    @overload
    def __add__(
        self: Index[T_COMPLEX], other: np_ndarray_anyint | Index[int]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(
        self: Index[bool] | Index[int], other: np_ndarray_float | Index[float]
    ) -> Index[float]: ...
    @overload
    def __add__(
        self: Index[T_COMPLEX], other: np_ndarray_float | Index[float]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(
        self: Index[T_COMPLEX], other: np_ndarray_complex | Index[complex]
    ) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __add__(
        self: Index[_str], other: np_ndarray_str | Index[_str]
    ) -> Index[_str]: ...
    @overload
    def __radd__(self: Index[Never], other: _str) -> Never: ...
    @overload
    def __radd__(
        self: Index[Never], other: complex | ArrayLike | SequenceNotStr[S1] | Index
    ) -> Index: ...
    @overload
    def __radd__(self: Index[Never], other: Period) -> PeriodIndex: ...
    @overload
    def __radd__(
        self: Supports_ProtoRAdd[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Index[S2]: ...
    @overload
    def __radd__(
        self: Index[S2_contra],
        other: SupportsAdd[S2_contra, S2] | Sequence[SupportsAdd[S2_contra, S2]],
    ) -> Index[S2]: ...
    @overload
    def __radd__(
        self: Index[T_COMPLEX], other: np_ndarray_bool | Index[bool]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Index[bool], other: np_ndarray_anyint | Index[int]
    ) -> Index[int]: ...
    @overload
    def __radd__(
        self: Index[T_COMPLEX], other: np_ndarray_anyint | Index[int]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Index[bool] | Index[int], other: np_ndarray_float | Index[float]
    ) -> Index[float]: ...
    @overload
    def __radd__(
        self: Index[T_COMPLEX], other: np_ndarray_float | Index[float]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Index[T_COMPLEX], other: np_ndarray_complex | Index[complex]
    ) -> Index[complex]: ...
    @overload
    def __radd__(
        self: Index[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __radd__(
        self: Index[_str], other: np_ndarray_str | Index[_str]
    ) -> Index[_str]: ...
    @overload
    def __sub__(self: Index[Never], other: DatetimeIndex) -> Never: ...
    @overload
    def __sub__(
        self: Index[Never], other: complex | ArrayLike | SequenceNotStr[S1] | Index
    ) -> Index: ...
    @overload
    def __sub__(self, other: Index[Never]) -> Index: ...
    @overload
    def __sub__(
        self: Index[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Index[int],
    ) -> Index[int]: ...
    @overload
    def __sub__(
        self: Index[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
        ),
    ) -> Index[int]: ...
    @overload
    def __sub__(
        self: Index[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Index[int]
            | Index[float]
        ),
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __sub__(
        self: Index[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
        ),
    ) -> Index[complex]: ...
    @overload
    def __rsub__(self: Index[Never], other: DatetimeIndex) -> Never: ...  # type: ignore[misc]
    @overload
    def __rsub__(
        self: Index[Never], other: complex | ArrayLike | SequenceNotStr[S1] | Index
    ) -> Index: ...
    @overload
    def __rsub__(self, other: Index[Never]) -> Index: ...
    @overload
    def __rsub__(
        self: Index[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Index[int],
    ) -> Index[int]: ...
    @overload
    def __rsub__(
        self: Index[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
        ),
    ) -> Index[int]: ...
    @overload
    def __rsub__(
        self: Index[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Index[int]
            | Index[float]
        ),
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __rsub__(
        self: Index[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
        ),
    ) -> Index[complex]: ...
    @overload
    def __mul__(
        self: Index[Never], other: complex | ArrayLike | SequenceNotStr[S1] | Index
    ) -> Index: ...
    @overload
    def __mul__(self, other: Index[Never]) -> Index: ...
    @overload
    def __mul__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __mul__(self: Index[bool] | Index[complex], other: np_ndarray_td) -> Never: ...
    @overload
    def __mul__(
        self: Index[int] | Index[float],
        other: timedelta | Sequence[Timedelta] | np.timedelta64 | np_ndarray_td,
    ) -> TimedeltaIndex: ...
    @overload
    def __mul__(
        self: Index[_str],
        other: (
            np_ndarray_bool
            | np_ndarray_float
            | np_ndarray_complex
            | np_ndarray_dt
            | np_ndarray_td
        ),
    ) -> Never: ...
    @overload
    def __mul__(
        self: Index[_str], other: np_ndarray_anyint | Index[int]
    ) -> Index[_str]: ...
    @overload
    def __mul__(
        self: Supports_ProtoMul[_T_contra, S2], other: _T_contra | Sequence[_T_contra]
    ) -> Index[S2]: ...
    @overload
    def __mul__(
        self: Index[S2_contra],
        other: (
            SupportsRMul[S2_contra, S2_NSDT]
            | Sequence[SupportsRMul[S2_contra, S2_NSDT]]
        ),
    ) -> Index[S2_NSDT]: ...
    @overload
    def __mul__(
        self: Index[T_COMPLEX], other: np_ndarray_bool | Index[bool]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __mul__(
        self: Index[bool], other: np_ndarray_anyint | Index[int]
    ) -> Index[int]: ...
    @overload
    def __mul__(
        self: Index[T_COMPLEX], other: np_ndarray_anyint | Index[int]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __mul__(
        self: Index[bool] | Index[int], other: np_ndarray_float | Index[float]
    ) -> Index[float]: ...
    @overload
    def __mul__(
        self: Index[T_COMPLEX], other: np_ndarray_float | Index[float]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __mul__(
        self: Index[T_COMPLEX], other: np_ndarray_complex | Index[complex]
    ) -> Index[complex]: ...
    @overload
    def __rmul__(
        self: Index[Never], other: complex | ArrayLike | SequenceNotStr[S1] | Index
    ) -> Index: ...
    @overload
    def __rmul__(self, other: Index[Never]) -> Index: ...
    @overload
    def __rmul__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __rmul__(self: Index[bool] | Index[complex], other: np_ndarray_td) -> Never: ...
    @overload
    def __rmul__(
        self: Index[int] | Index[float],
        other: timedelta | Sequence[Timedelta] | np.timedelta64 | np_ndarray_td,
    ) -> TimedeltaIndex: ...
    @overload
    def __rmul__(
        self: Index[_str],
        other: (
            np_ndarray_bool
            | np_ndarray_float
            | np_ndarray_complex
            | np_ndarray_dt
            | np_ndarray_td
        ),
    ) -> Never: ...
    @overload
    def __rmul__(
        self: Index[_str], other: np_ndarray_anyint | Index[int]
    ) -> Index[_str]: ...
    @overload
    def __rmul__(
        self: Supports_ProtoRMul[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Index[S2]: ...
    @overload
    def __rmul__(
        self: Index[S2_contra],
        other: (
            SupportsMul[S2_contra, S2_NSDT] | Sequence[SupportsMul[S2_contra, S2_NSDT]]
        ),
    ) -> Index[S2_NSDT]: ...
    @overload
    def __rmul__(
        self: Index[T_COMPLEX], other: np_ndarray_bool | Index[bool]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __rmul__(
        self: Index[bool], other: np_ndarray_anyint | Index[int]
    ) -> Index[int]: ...
    @overload
    def __rmul__(
        self: Index[T_COMPLEX], other: np_ndarray_anyint | Index[int]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __rmul__(
        self: Index[bool] | Index[int], other: np_ndarray_float | Index[float]
    ) -> Index[float]: ...
    @overload
    def __rmul__(
        self: Index[T_COMPLEX], other: np_ndarray_float | Index[float]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __rmul__(
        self: Index[T_COMPLEX], other: np_ndarray_complex | Index[complex]
    ) -> Index[complex]: ...
    @overload
    def __truediv__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __truediv__(  # type: ignore[overload-overlap]
        self: Index[Never], other: ScalarArrayIndexComplex
    ) -> Index: ...
    @overload
    def __truediv__(self: Index[Never], other: ArrayIndexTimedeltaNoSeq) -> Never: ...
    @overload
    def __truediv__(self: Index[T_COMPLEX], other: np_ndarray_td) -> Never: ...
    @overload
    def __truediv__(self: Index[bool], other: np_ndarray_bool) -> Never: ...
    @overload
    def __truediv__(self: IndexComplex, other: Index[Never]) -> Index: ...
    @overload
    def __truediv__(
        self: Supports_ProtoTrueDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Index[S2]: ...
    @overload
    def __truediv__(
        self: Index[int], other: np_ndarray_bool | Index[bool]
    ) -> Index[float]: ...
    @overload
    def __truediv__(
        self: Index[bool] | Index[int], other: ScalarArrayIndexIntNoBool
    ) -> Index[float]: ...
    @overload
    def __truediv__(
        self: Index[float],
        other: np_ndarray_bool | np_ndarray_anyint | Index[bool] | Index[int],
    ) -> Index[float]: ...
    @overload
    def __truediv__(
        self: Index[complex],
        other: np_ndarray_bool | np_ndarray_anyint | Index[bool] | Index[int],
    ) -> Index[complex]: ...
    @overload
    def __truediv__(
        self: Index[bool] | Index[int], other: ScalarArrayIndexJustFloat
    ) -> Index[float]: ...
    @overload
    def __truediv__(
        self: Index[T_COMPLEX], other: ScalarArrayIndexJustFloat
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __truediv__(
        self: IndexComplex, other: ScalarArrayIndexJustComplex
    ) -> Index[complex]: ...
    @overload
    def __truediv__(self: Index[_str], other: Path) -> Index: ...
    @overload
    def __rtruediv__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __rtruediv__(  # type: ignore[overload-overlap]
        self: Index[Never], other: ScalarArrayIndexComplex | ScalarArrayIndexTimedelta
    ) -> Index: ...
    @overload
    def __rtruediv__(  # type: ignore[overload-overlap]
        self: IndexComplex, other: Index[Never]
    ) -> Index: ...
    @overload
    def __rtruediv__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        self: Index[int] | Index[float], other: Sequence[timedelta | np.timedelta64]
    ) -> Index: ...
    @overload
    def __rtruediv__(
        self: Supports_ProtoRTrueDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Index[S2]: ...
    @overload
    def __rtruediv__(
        self: Index[int], other: np_ndarray_bool | Index[bool]
    ) -> Index[float]: ...
    @overload
    def __rtruediv__(
        self: Index[bool] | Index[int], other: ScalarArrayIndexIntNoBool
    ) -> Index[float]: ...
    @overload
    def __rtruediv__(  # type: ignore[misc]
        self: Index[float],
        other: np_ndarray_bool | np_ndarray_anyint | Index[bool] | Index[int],
    ) -> Index[float]: ...
    @overload
    def __rtruediv__(
        self: Index[complex],
        other: np_ndarray_bool | np_ndarray_anyint | Index[bool] | Index[int],
    ) -> Index[complex]: ...
    @overload
    def __rtruediv__(
        self: Index[bool] | Index[int], other: ScalarArrayIndexJustFloat
    ) -> Index[float]: ...
    @overload
    def __rtruediv__(
        self: Index[T_COMPLEX], other: ScalarArrayIndexJustFloat
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __rtruediv__(
        self: IndexComplex, other: ScalarArrayIndexJustComplex
    ) -> Index[complex]: ...
    @overload
    def __rtruediv__(
        self: Index[int] | Index[float], other: ScalarArrayIndexTimedelta
    ) -> TimedeltaIndex: ...
    @overload
    def __rtruediv__(self: Index[_str], other: Path) -> Index: ...
    @overload
    def __floordiv__(self, other: np_ndarray_dt) -> Never: ...
    @overload
    def __floordiv__(self: Index[Never], other: np_ndarray_td) -> Never: ...
    @overload
    def __floordiv__(
        self: Index[int] | Index[float], other: np_ndarray_complex | np_ndarray_td
    ) -> Never: ...
    @overload
    def __floordiv__(self: Index[Never], other: ScalarArrayIndexReal) -> Index: ...
    @overload
    def __floordiv__(self: IndexReal, other: Index[Never]) -> Index: ...
    @overload
    def __floordiv__(
        self: Index[bool] | Index[complex], other: np_ndarray
    ) -> Never: ...
    @overload
    def __floordiv__(
        self: Supports_ProtoFloorDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Index[S2]: ...
    @overload
    def __floordiv__(
        self: Index[int], other: np_ndarray_bool | Index[bool]
    ) -> Index[int]: ...
    @overload
    def __floordiv__(
        self: Index[float], other: np_ndarray_bool | Index[bool]
    ) -> Index[float]: ...
    @overload
    def __floordiv__(
        self: Index[bool] | Index[int], other: np_ndarray_anyint | Index[int]
    ) -> Index[int]: ...
    @overload
    def __floordiv__(
        self: Index[float], other: np_ndarray_anyint | Index[int]
    ) -> Index[float]: ...
    @overload
    def __floordiv__(
        self: Index[int] | Index[float],
        other: float | Sequence[float] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __rfloordiv__(  # type: ignore[overload-overlap]
        self: Index[Never], other: ScalarArrayIndexReal
    ) -> Index: ...
    @overload
    def __rfloordiv__(self, other: np_ndarray_complex | np_ndarray_dt) -> Never: ...
    @overload
    def __rfloordiv__(
        self: Index[int] | Index[float], other: np_ndarray_td
    ) -> Never: ...
    @overload
    def __rfloordiv__(
        self: Index[bool] | Index[complex], other: np_ndarray
    ) -> Never: ...
    @overload
    def __rfloordiv__(  # type: ignore[overload-overlap]
        self: IndexReal, other: Index[Never]
    ) -> Index: ...
    @overload
    def __rfloordiv__(
        self: Supports_ProtoRFloorDiv[_T_contra, S2],
        other: _T_contra | Sequence[_T_contra],
    ) -> Index[S2]: ...
    @overload
    def __rfloordiv__(
        self: Index[int], other: np_ndarray_bool | Index[bool]
    ) -> Index[int]: ...
    @overload
    def __rfloordiv__(
        self: Index[float], other: np_ndarray_bool | Index[bool]
    ) -> Index[float]: ...
    @overload
    def __rfloordiv__(
        self: Index[bool] | Index[int], other: np_ndarray_anyint | Index[int]
    ) -> Index[int]: ...
    @overload
    def __rfloordiv__(
        self: Index[float], other: np_ndarray_anyint | Index[int]
    ) -> Index[float]: ...
    @overload
    def __rfloordiv__(
        self: Index[int] | Index[float],
        other: float | Sequence[float] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __rfloordiv__(
        self: Index[int] | Index[float],
        other: timedelta | np.timedelta64 | ArrayIndexTimedeltaNoSeq,
    ) -> TimedeltaIndex: ...
    @overload
    def __rfloordiv__(
        self: Index[int] | Index[float], other: Sequence[timedelta | np.timedelta64]
    ) -> Index: ...
    def infer_objects(self, copy: bool = True) -> Self: ...

@type_check_only
class _IndexSubclassBase(Index[S1], Generic[S1, GenericT_co]):
    @overload
    def to_numpy(
        self: _IndexSubclassBase[Interval],
        dtype: type[T_INTERVAL_NP],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
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
    def to_numpy(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dtype: DTypeLike,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
