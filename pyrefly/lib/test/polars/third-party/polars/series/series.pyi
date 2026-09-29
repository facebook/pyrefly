from typing import Any, overload

class Series:
    def __init__(
        self,
        name: object = None,
        values: object = None,
        dtype: object = None,
        *,
        strict: bool = True,
        nan_to_null: bool = False,
    ) -> None: ...
    @overload
    def __getitem__(self, key: int) -> Any: ...
    @overload
    def __getitem__(self, key: slice) -> Series: ...
    def __or__(self, other: Series) -> Series: ...
