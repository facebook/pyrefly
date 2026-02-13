from collections.abc import Sequence
from gzip import GzipFile
from typing import Any, Literal

from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import spmatrix

def load_arff_from_gzip_file(
    gzip_file: GzipFile,
    parser: Literal["pandas", "liac-arff"],
    output_type: Literal["numpy", "sparse", "pandas"],
    openml_columns_info: dict,
    feature_names_to_select: list[str] | Sequence[str],
    target_names_to_select: Sequence[str] | list[Any | str],
    shape: None | tuple[int, int] = None,
) -> tuple[
    DataFrame | ndarray | spmatrix,
    DataFrame | ndarray | Series,
    DataFrame | None,
    list[str] | None,
]: ...
