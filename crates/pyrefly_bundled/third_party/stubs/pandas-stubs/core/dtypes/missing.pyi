from typing import (
    Any,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.arrays import ExtensionArray
from typing_extensions import TypeIs

from pandas._libs.missing import NAType
from pandas._libs.tslibs import NaTType
from pandas._typing import (
    Scalar,
    ScalarT,
    ShapeT,
    np_1darray_bool,
    np_ndarray,
    np_ndarray_bool,
)

isposinf_scalar = ...
isneginf_scalar = ...

@overload
def isna(obj: DataFrame) -> DataFrame: ...
@overload
def isna(obj: Series) -> Series[bool]: ...
@overload
def isna(obj: Index | ExtensionArray | list[ScalarT]) -> np_1darray_bool: ...
@overload
def isna(obj: np_ndarray[ShapeT]) -> np_ndarray[ShapeT, np.bool]: ...
@overload
def isna(obj: list[Any]) -> np_ndarray_bool: ...
@overload
def isna(
    obj: Scalar | NaTType | NAType | None,
) -> TypeIs[NaTType | NAType | None]: ...

isnull = isna

@overload
def notna(obj: DataFrame) -> DataFrame: ...
@overload
def notna(obj: Series) -> Series[bool]: ...
@overload
def notna(obj: Index | ExtensionArray | list[ScalarT]) -> np_1darray_bool: ...
@overload
def notna(obj: np_ndarray[ShapeT]) -> np_ndarray[ShapeT, np.bool]: ...
@overload
def notna(obj: list[Any]) -> np_ndarray_bool: ...
@overload
def notna(obj: ScalarT | NaTType | NAType | None) -> TypeIs[ScalarT]: ...

notnull = notna
