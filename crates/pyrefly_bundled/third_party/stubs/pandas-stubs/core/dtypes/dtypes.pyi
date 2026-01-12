import datetime as dt
from typing import (
    Any,
    Literal,
)

import numpy as np
from pandas.core.indexes.base import Index
from pandas.core.series import Series

from pandas._libs import NaTType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.offsets import (
    RelativeDeltaOffset,
    SingleConstructorOffset,
)
from pandas._typing import (
    Ordered,
    TimeZones,
    npt,
)

from pandas.core.dtypes.base import (
    ExtensionDtype as ExtensionDtype,
    register_extension_dtype as register_extension_dtype,
)

class BaseMaskedDtype(ExtensionDtype): ...
class PandasExtensionDtype(ExtensionDtype): ...

class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    def __init__(
        self,
        categories: Series | Index | list[Any] | None = ...,
        ordered: Ordered = ...,
    ) -> None: ...
    @property
    def categories(self) -> Index: ...
    @property
    def ordered(self) -> Ordered: ...

class DatetimeTZDtype(PandasExtensionDtype):
    def __init__(self, unit: Literal["ns"] = ..., tz: TimeZones = ...) -> None: ...
    @property
    def unit(self) -> Literal["ns"]: ...
    @property
    def tz(self) -> dt.tzinfo: ...
    @property
    def na_value(self) -> NaTType: ...

class PeriodDtype(PandasExtensionDtype):
    def __init__(
        self, freq: str | SingleConstructorOffset | RelativeDeltaOffset = ...
    ) -> None: ...
    @property
    def freq(self) -> BaseOffset: ...
    @property
    def na_value(self) -> NaTType: ...

class IntervalDtype(PandasExtensionDtype):
    def __init__(self, subtype: str | npt.DTypeLike | None = ...) -> None: ...
    @property
    def subtype(self) -> np.dtype | None: ...
