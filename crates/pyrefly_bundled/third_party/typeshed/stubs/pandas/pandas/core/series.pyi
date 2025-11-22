# Minimal Series stub
from typing import Any

from pandas._typing import Axes, Dtype

class Series:
    def __init__(
        self,
        data: Any = None,
        index: Axes | None = None,
        dtype: Dtype | None = None,
        name: Any = None,
        copy: bool | None = None,
    ) -> None: ...
