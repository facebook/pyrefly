from collections.abc import (
    Container,
    Generator,
    Iterable,
)
from contextlib import contextmanager
from typing import (
    Any,
    Literal,
    overload,
)
import warnings

from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
)
from pandas.arrays import (
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.arrays.base import ExtensionArray

from pandas._typing import (
    AnyArrayLike,
    T,
    np_ndarray,
)

def assert_almost_equal(
    left: T,
    right: T,
    check_dtype: bool | Literal["equiv"] = "equiv",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    **kwargs: Any,
) -> None: ...
def assert_dict_equal(left: dict, right: dict, compare_keys: bool = True) -> None: ...
def assert_index_equal(
    left: Index,
    right: Index,
    exact: bool | Literal["equiv"] = "equiv",
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    obj: str = "Index",
) -> None: ...
def assert_class_equal(
    left: T, right: T, exact: bool | Literal["equiv"] = True, obj: str = "Input"
) -> None: ...
def assert_attr_equal(
    attr: str, left: object, right: object, obj: str = "Attributes"
) -> None: ...
def assert_is_sorted(seq: AnyArrayLike) -> None: ...
def assert_categorical_equal(
    left: Categorical,
    right: Categorical,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = "Categorical",
) -> None: ...
def assert_interval_array_equal(
    left: IntervalArray,
    right: IntervalArray,
    exact: bool | Literal["equiv"] = "equiv",
    obj: str = "IntervalArray",
) -> None: ...
def assert_period_array_equal(
    left: PeriodArray, right: PeriodArray, obj: str = "PeriodArray"
) -> None: ...
def assert_datetime_array_equal(
    left: DatetimeArray,
    right: DatetimeArray,
    obj: str = "DatetimeArray",
    check_freq: bool = True,
) -> None: ...
def assert_timedelta_array_equal(
    left: TimedeltaArray,
    right: TimedeltaArray,
    obj: str = "TimedeltaArray",
    check_freq: bool = True,
) -> None: ...
def assert_extension_array_equal(
    left: ExtensionArray,
    right: ExtensionArray,
    check_dtype: bool | Literal["equiv"] = True,
    index_values: Index | np_ndarray | None = None,
    check_exact: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    obj: str = "ExtensionArray",
) -> None: ...
@overload
def assert_series_equal(
    left: Series,
    right: Series,
    check_dtype: bool | Literal["equiv"] = ...,
    check_index_type: bool | Literal["equiv"] = ...,
    check_series_type: bool = ...,
    check_names: bool = ...,
    check_exact: bool = ...,
    check_datetimelike_compat: bool = ...,
    check_categorical: bool = ...,
    check_category_order: bool = ...,
    check_freq: bool = ...,
    check_flags: bool = ...,
    rtol: float = ...,
    atol: float = ...,
    obj: str = ...,
    *,
    check_index: Literal[False],
    check_like: Literal[False] = False,
) -> None: ...
@overload
def assert_series_equal(
    left: Series,
    right: Series,
    check_dtype: bool | Literal["equiv"] = ...,
    check_index_type: bool | Literal["equiv"] = ...,
    check_series_type: bool = ...,
    check_names: bool = ...,
    check_exact: bool = ...,
    check_datetimelike_compat: bool = ...,
    check_categorical: bool = ...,
    check_category_order: bool = ...,
    check_freq: bool = ...,
    check_flags: bool = ...,
    rtol: float = ...,
    atol: float = ...,
    obj: str = ...,
    *,
    check_index: Literal[True] = True,
    check_like: bool = ...,
) -> None: ...
def assert_frame_equal(
    left: DataFrame,
    right: DataFrame,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_column_type: bool | Literal["equiv"] = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    obj: str = "DataFrame",
) -> None: ...
def assert_sp_array_equal(left: SparseArray, right: SparseArray) -> None: ...
def assert_contains_all(iterable: Iterable[T], dic: Container[T]) -> None: ...
def assert_copy(iter1: Iterable[T], iter2: Iterable[T], **eql_kwargs: Any) -> None: ...
@contextmanager
def assert_produces_warning(
    expected_warning: (
        type[Warning] | Literal[False] | tuple[type[Warning], ...] | None
    ) = ...,
    filter_level: Literal[
        "error", "ignore", "always", "default", "module", "once"
    ] = "always",
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: str | None = None,
) -> Generator[list[warnings.WarningMessage], None, None]: ...
@contextmanager
def ensure_clean(filename: str | None = None) -> Generator[str, None, None]: ...
