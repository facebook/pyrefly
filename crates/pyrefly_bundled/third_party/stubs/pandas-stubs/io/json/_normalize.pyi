from pandas import DataFrame

from pandas._typing import IgnoreRaise

def json_normalize(
    data: dict | list[dict],
    record_path: str | list | None = None,
    meta: str | list[str | list[str]] | None = None,
    meta_prefix: str | None = None,
    record_prefix: str | None = None,
    errors: IgnoreRaise = "raise",
    sep: str = ".",
    max_level: int | None = None,
) -> DataFrame: ...
