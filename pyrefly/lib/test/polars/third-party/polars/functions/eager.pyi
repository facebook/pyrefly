from typing import Iterable

from polars.dataframe.frame import DataFrame

def concat(
    items: Iterable[DataFrame],
    *,
    how: str = "vertical",
    rechunk: bool = False,
    parallel: bool = True,
) -> DataFrame: ...
