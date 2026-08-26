from polars.dataframe.frame import DataFrame
from polars.lazyframe.frame import LazyFrame

def read_csv(
    source: object,
    *,
    schema: object = None,
    schema_overrides: object = None,
    columns: object = None,
    new_columns: object = None,
    row_index_name: str | None = None,
    **kwargs: object,
) -> DataFrame: ...
def scan_csv(
    source: object,
    *,
    schema: object = None,
    schema_overrides: object = None,
    new_columns: object = None,
    row_index_name: str | None = None,
    with_column_names: object = None,
    include_file_paths: str | None = None,
    **kwargs: object,
) -> LazyFrame: ...
