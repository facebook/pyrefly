import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)

from pandas._typing import (
    ArrayLike,
    np_1darray,
)

def hash_pandas_object(
    obj: Index | Series | DataFrame,
    index: bool = True,
    encoding: str = "utf8",
    hash_key: str | None = "0123456789123456",
    categorize: bool = True,
) -> Series[int]: ...
def hash_array(
    vals: ArrayLike,
    encoding: str = "utf8",
    hash_key: str = "0123456789123456",
    categorize: bool = True,
) -> np_1darray[np.uint64]: ...
