from collections.abc import Mapping
from typing import (
    Any,
    Literal,
)

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.computation.ops import BinOp

from pandas._typing import (
    Scalar,
    np_ndarray,
)

def eval(
    expr: str | BinOp,
    parser: Literal["pandas", "python"] = "pandas",
    engine: Literal["python", "numexpr"] | None = ...,
    local_dict: dict[str, Any] | None = None,
    global_dict: dict[str, Any] | None = None,
    resolvers: list[Mapping] | None = ...,
    level: int = 0,
    target: object | None = None,
    inplace: bool = False,
) -> np_ndarray | Scalar | DataFrame | Series | None: ...
