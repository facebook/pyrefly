from pandas import (
    DataFrame,
    Series,
)

from pandas._typing import JSONSerializable

def build_table_schema(
    data: DataFrame | Series,
    index: bool = True,
    primary_key: bool | None = True,
    version: bool = True,
) -> dict[str, JSONSerializable]: ...
