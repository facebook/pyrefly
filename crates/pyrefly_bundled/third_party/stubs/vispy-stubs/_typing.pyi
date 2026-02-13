import decimal
import io
import typing_extensions

import numpy.typing
import pandas as pd

Decimal = decimal.Decimal
PythonScalar: typing_extensions.TypeAlias = str | int | float | bool

ArrayLike = numpy.typing.ArrayLike
FileLike = io.IOBase
PathLike = str

PandasScalar: typing_extensions.TypeAlias = pd.Period | pd.Timestamp | pd.Timedelta | pd.Interval
Scalar: typing_extensions.TypeAlias = PythonScalar | PandasScalar

Color: typing_extensions.TypeAlias = tuple[float, float, float] | str

__all__ = [
    "ArrayLike",
    "Color",
    "Decimal",
    "FileLike",
    "PathLike",
    "Scalar",
]
