from collections.abc import Sequence
from typing import Any

from pandas.core.indexes.datetimes import DatetimeIndex

from pandas._libs.tslibs import BaseOffset
from pandas._typing import (
    np_1darray_intp,
    np_ndarray_intp,
)

class BaseIndexer:
    def __init__(
        self,
        index_array: Sequence[float] | np_ndarray_intp | None = None,
        window_size: int = 0,
        **kwargs: Any,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int,
        min_periods: int | None,
        center: bool | None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np_1darray_intp, np_1darray_intp]: ...

class FixedForwardWindowIndexer(BaseIndexer): ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: np_ndarray_intp | None = None,
        window_size: int = 0,
        index: DatetimeIndex | None = None,
        offset: BaseOffset | None = None,
        **kwargs: Any,
    ) -> None: ...
