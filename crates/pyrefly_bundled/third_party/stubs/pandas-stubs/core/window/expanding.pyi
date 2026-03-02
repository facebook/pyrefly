from pandas.core.window.rolling import (
    BaseWindowGroupby,
    RollingAndExpandingMixin,
)

from pandas._typing import NDFrameT

class Expanding(RollingAndExpandingMixin[NDFrameT]): ...
class ExpandingGroupby(BaseWindowGroupby[NDFrameT], Expanding[NDFrameT]): ...
