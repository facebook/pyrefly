from collections.abc import (
    Hashable,
    Iterable,
)

from pandas import DataFrame

from pandas._typing import (
    AnyArrayLike,
    Dtype,
    HashableT1,
    HashableT2,
)

def get_dummies(
    data: AnyArrayLike | DataFrame,
    prefix: str | Iterable[str] | dict[HashableT1, str] | None = None,
    prefix_sep: str = "_",
    dummy_na: bool = False,
    columns: list[HashableT2] | None = None,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: Dtype | None = None,
) -> DataFrame: ...
def from_dummies(
    data: DataFrame,
    sep: str | None = None,
    default_category: Hashable | dict[str, Hashable] | None = None,
) -> DataFrame: ...
