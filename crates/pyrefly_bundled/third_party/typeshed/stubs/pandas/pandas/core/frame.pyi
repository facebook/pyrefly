# DataFrame stub with corrected type annotations
from typing import Any

from pandas._typing import Axes, Dtype

class DataFrame:
    def __init__(
        self,
        data: Any = None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
    ) -> None: ...
