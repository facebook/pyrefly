from typing import Any

import numpy as np
from pandas.core.indexes.extension import ExtensionIndex
from typing_extensions import (
    Never,
    Self,
)

from pandas._libs.tslibs import BaseOffset
from pandas._typing import (
    S1,
    AxisIndex,
    GenericT_co,
    TimeUnit,
    np_ndarray_complex,
)

class DatetimeIndexOpsMixin(ExtensionIndex[S1, GenericT_co]):
    @property
    def freq(self) -> BaseOffset | None: ...
    @property
    def freqstr(self) -> str | None: ...
    @property
    def is_all_dates(self) -> bool: ...
    def min(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> S1: ...
    def argmin(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    def max(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> S1: ...
    def argmax(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    def __mul__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: np_ndarray_complex
    ) -> Never: ...
    def __rmul__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: np_ndarray_complex
    ) -> Never: ...

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin[S1, GenericT_co]):
    @property
    def unit(self) -> TimeUnit: ...
    def as_unit(self, unit: TimeUnit) -> Self: ...
