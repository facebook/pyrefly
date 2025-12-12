from collections.abc import (
    Iterator,
)
from typing import Generic

from pandas._typing import (
    AxisInt,
    NDFrameT,
    np_ndarray_intp,
)

class DataSplitter(Generic[NDFrameT]):
    data: NDFrameT
    labels: np_ndarray_intp
    ngroups: int
    axis: AxisInt
    def __iter__(self) -> Iterator[NDFrameT]: ...
