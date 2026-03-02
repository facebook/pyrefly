from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from typing import (
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.series import Series
from typing_extensions import Never

from pandas._typing import (
    S2,
    Axis,
    AxisIndex,
    HashableT1,
    HashableT2,
    HashableT3,
    HashableT4,
)

@overload
def concat(
    objs: Iterable[None] | Mapping[HashableT1, None],
    *,
    axis: Axis = 0,
    join: Literal["inner", "outer"] = "outer",
    ignore_index: bool = False,
    keys: Iterable[HashableT2] | None = None,
    levels: Sequence[list[HashableT3] | tuple[HashableT3, ...]] | None = None,
    names: list[HashableT4] | None = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = True,
) -> Never: ...
@overload
def concat(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    objs: Iterable[Series[S2] | None] | Mapping[HashableT1, Series[S2] | None],
    *,
    axis: AxisIndex = 0,
    join: Literal["inner", "outer"] = "outer",
    ignore_index: bool = False,
    keys: Iterable[HashableT2] | None = None,
    levels: Sequence[list[HashableT3] | tuple[HashableT3, ...]] | None = None,
    names: list[HashableT4] | None = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = True,
) -> Series[S2]: ...
@overload
def concat(  # type: ignore[overload-overlap]
    objs: Iterable[Series | None] | Mapping[HashableT1, Series | None],
    *,
    axis: AxisIndex = 0,
    join: Literal["inner", "outer"] = "outer",
    ignore_index: bool = False,
    keys: Iterable[HashableT2] | None = None,
    levels: Sequence[list[HashableT3] | tuple[HashableT3, ...]] | None = None,
    names: list[HashableT4] | None = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = True,
) -> Series: ...
@overload
def concat(
    objs: Iterable[NDFrame | None] | Mapping[HashableT1, NDFrame | None],
    *,
    axis: Axis = 0,
    join: Literal["inner", "outer"] = "outer",
    ignore_index: bool = False,
    keys: Iterable[HashableT2] | None = None,
    levels: Sequence[list[HashableT3] | tuple[HashableT3, ...]] | None = None,
    names: list[HashableT4] | None = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = True,
) -> DataFrame: ...
