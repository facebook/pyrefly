from builtins import type as type_t
from collections.abc import (
    Callable,
    Hashable,
    Iterator,
    KeysView,
    Mapping,
    MutableSequence,
    Sequence,
)
import datetime
from datetime import tzinfo
from os import PathLike
from re import Pattern
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    SupportsIndex,
    TypeAlias,
    TypedDict,
    Union,
    overload,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
from pandas.core.arrays import (
    ExtensionArray,
    IntegerArray,
)
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from pandas.core.tools.datetimes import FulldatetimeDict
from typing_extensions import (
    ParamSpec,
    TypeVar,
    override,
)

from pandas._libs.interval import Interval
from pandas._libs.missing import NAType
from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    Timedelta,
    Timestamp,
)

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)

from pandas.io.formats.format import EngFormatter
from pandas.tseries.offsets import (
    Day,
    Hour,
    Micro,
    Milli,
    Minute,
    MonthEnd,
    Nano,
    QuarterEnd,
    Second,
    Week,
    YearEnd,
)

P = ParamSpec("P")

HashableT = TypeVar("HashableT", bound=Hashable)
HashableT1 = TypeVar("HashableT1", bound=Hashable)
HashableT2 = TypeVar("HashableT2", bound=Hashable)
HashableT3 = TypeVar("HashableT3", bound=Hashable)
HashableT4 = TypeVar("HashableT4", bound=Hashable)
HashableT5 = TypeVar("HashableT5", bound=Hashable)

# array-like

ArrayLike: TypeAlias = ExtensionArray | npt.NDArray[Any]
AnyArrayLike: TypeAlias = ArrayLike | Index | Series
if TYPE_CHECKING:  # noqa: PYI002
    AnyArrayLikeInt: TypeAlias = (
        IntegerArray | Index[int] | Series[int] | npt.NDArray[np.integer]
    )

# list-like

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)

class SequenceNotStr(Protocol[_T_co]):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T_co]: ...
    def __contains__(self, value: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def index(self, value: Any, start: int = ..., stop: int = ..., /) -> int: ...
    def count(self, value: Any, /) -> int: ...
    def __reversed__(self) -> Iterator[_T_co]: ...

ListLike: TypeAlias = AnyArrayLike | SequenceNotStr[Any] | range

# scalars

PythonScalar: TypeAlias = str | bool | complex
DatetimeLikeScalar: TypeAlias = Period | Timestamp | Timedelta
PandasScalar: TypeAlias = Period | Timestamp | Timedelta | Interval

_IndexIterScalar: TypeAlias = (
    str
    | bytes
    | datetime.date
    | datetime.datetime
    | datetime.timedelta
    | np.datetime64
    | np.timedelta64
    | bool
    | int
    | float
    | Timestamp
    | Timedelta
)
# This is wider than what is in pandas
Scalar: TypeAlias = (
    _IndexIterScalar | complex | np.integer | np.floating | np.complexfloating
)
IntStrT = TypeVar("IntStrT", int, str)

# timestamp and timedelta convertible types

TimestampConvertibleTypes: TypeAlias = (
    Timestamp
    | datetime.datetime
    | datetime.date
    | np.datetime64
    | np.integer
    | float
    | str
)
TimestampNonexistent: TypeAlias = (
    Literal["shift_forward", "shift_backward", "NaT", "raise"]
    | Timedelta
    | datetime.timedelta
)
TimedeltaConvertibleTypes: TypeAlias = (
    Timedelta | datetime.timedelta | np.timedelta64 | np.integer | float | str
)

Timezone: TypeAlias = str | tzinfo  # Not used in pandas or the stubs

ToTimestampHow: TypeAlias = Literal["s", "e", "start", "end"]

# NDFrameT is stricter and ensures that the same subclass of NDFrame always is
# used. E.g. `def func(a: NDFrameT) -> NDFrameT: ...` means that if a
# Series is passed into a function, a Series is always returned and if a DataFrame is
# passed in, a DataFrame is always returned.
NDFrameT = TypeVar("NDFrameT", bound=NDFrame)

IndexT = TypeVar("IndexT", bound=Index)
T_EXTENSION_ARRAY = TypeVar("T_EXTENSION_ARRAY", bound=ExtensionArray)

# From _typing.py, not used here:
# FreqIndexT = TypeVar("FreqIndexT", "DatetimeIndex", "PeriodIndex", "TimedeltaIndex")
# NumpyIndexT = TypeVar("NumpyIndexT", np.ndarray, "Index")

AxisInt: TypeAlias = int
AxisIndex: TypeAlias = Literal["index", 0]
AxisColumn: TypeAlias = Literal["columns", 1]
Axis: TypeAlias = AxisIndex | AxisColumn  # slight difference from pandas
IndexLabel: TypeAlias = Hashable | Sequence[Hashable]
Level: TypeAlias = Hashable
Shape: TypeAlias = tuple[int, ...]
Suffixes: TypeAlias = tuple[str | None, str | None] | list[str | None]
Ordered: TypeAlias = bool | None
JSONSerializable: TypeAlias = PythonScalar | list | dict
Frequency: TypeAlias = str | BaseOffset
PeriodFrequency: TypeAlias = (
    str
    | Day
    | Hour
    | Minute
    | Second
    | Milli
    | Micro
    | Nano
    | YearEnd
    | QuarterEnd
    | MonthEnd
    | Week
)
Axes: TypeAlias = ListLike

RandomState: TypeAlias = (
    int
    | ArrayLike
    | np.random.Generator
    | np.random.BitGenerator
    | np.random.RandomState
)

# dtypes
NpDtypeNoStr: TypeAlias = np.dtype[np.generic] | type[complex | bool | object]
NpDtype: TypeAlias = str | NpDtypeNoStr | type[str]
Dtype: TypeAlias = ExtensionDtype | NpDtype

# AstypeArg is more carefully defined here as compared to pandas

# NOTE: we want to catch all the possible dtypes from np.sctypeDict
# timedelta64
# M
# m8
# M8
# object_
# object0
# m
# datetime64

# Builtin bool type and its string alias
BuiltinBooleanDtypeArg: TypeAlias = type[bool] | Literal["bool"]
# Pandas nullable boolean type and its string alias
PandasBooleanDtypeArg: TypeAlias = pd.BooleanDtype | Literal["boolean"]
# Numpy bool type
# https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_
NumpyBooleanDtypeArg: TypeAlias = type[np.bool_] | Literal["?", "b1", "bool_"]
# PyArrow boolean type and its string alias
PyArrowBooleanDtypeArg: TypeAlias = Literal["bool[pyarrow]", "boolean[pyarrow]"]
BooleanDtypeArg: TypeAlias = (
    BuiltinBooleanDtypeArg
    | PandasBooleanDtypeArg
    | NumpyBooleanDtypeArg
    | PyArrowBooleanDtypeArg
)
# Builtin integer type and its string alias
BuiltinIntDtypeArg: TypeAlias = type[int] | Literal["int"]
# Pandas nullable integer types and their string aliases
PandasIntDtypeArg: TypeAlias = (
    pd.Int8Dtype
    | pd.Int16Dtype
    | pd.Int32Dtype
    | pd.Int64Dtype
    | Literal["Int8", "Int16", "Int32", "Int64"]
)
# Numpy signed integer types and their string aliases
NumpyIntDtypeArg: TypeAlias = (
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.byte
    type[np.byte]  # noqa: PYI030
    | Literal["b", "i1", "int8", "byte"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.short
    | type[np.short]
    | Literal["h", "i2", "int16", "short"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.intc
    | type[np.intc]
    | Literal["i", "i4", "int32", "intc"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int_
    | type[np.int_]
    | Literal["l", "i8", "int64", "int_", "long"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.longlong
    | type[np.longlong]
    | Literal["q", "longlong"]  # NOTE: int128 not assigned
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.intp
    | type[np.intp]  # signed pointer (=`intptr_t`, platform dependent)
    | Literal["p", "intp"]
)
# PyArrow integer types and their string aliases
PyArrowIntDtypeArg: TypeAlias = Literal[
    "int8[pyarrow]", "int16[pyarrow]", "int32[pyarrow]", "int64[pyarrow]"
]
IntDtypeArg: TypeAlias = (
    BuiltinIntDtypeArg | PandasIntDtypeArg | NumpyIntDtypeArg | PyArrowIntDtypeArg
)
# Pandas nullable unsigned integer types and their string aliases
PandasUIntDtypeArg: TypeAlias = (
    pd.UInt8Dtype
    | pd.UInt16Dtype
    | pd.UInt32Dtype
    | pd.UInt64Dtype
    | Literal["UInt8", "UInt16", "UInt32", "UInt64"]
)
# Numpy unsigned integer types and their string aliases
NumpyUIntDtypeArg: TypeAlias = (
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.ubyte
    type[np.ubyte]  # noqa: PYI030
    | Literal["B", "u1", "uint8", "ubyte"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.ushort
    | type[np.ushort]
    | Literal["H", "u2", "uint16", "ushort"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uintc
    | type[np.uintc]
    | Literal["I", "u4", "uint32", "uintc"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint
    | type[np.uint]
    | Literal["L", "u8", "uint", "ulong", "uint64"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.ulonglong
    | type[np.ulonglong]
    | Literal["Q", "ulonglong"]  # NOTE: uint128 not assigned
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uintp
    | type[np.uintp]  # unsigned pointer (=`uintptr_t`, platform dependent)
    | Literal["P", "uintp"]
)
# PyArrow unsigned integer types and their string aliases
PyArrowUIntDtypeArg: TypeAlias = Literal[
    "uint8[pyarrow]", "uint16[pyarrow]", "uint32[pyarrow]", "uint64[pyarrow]"
]
UIntDtypeArg: TypeAlias = PandasUIntDtypeArg | NumpyUIntDtypeArg | PyArrowUIntDtypeArg
# Builtin float type and its string alias
BuiltinFloatDtypeArg: TypeAlias = type[float] | Literal["float"]
# Pandas nullable float types and their string aliases
PandasFloatDtypeArg: TypeAlias = (
    pd.Float32Dtype | pd.Float64Dtype | Literal["Float32", "Float64"]
)
PandasAstypeFloatDtypeArg: TypeAlias = Literal["float_", "longfloat"]
# Numpy float types and their string aliases
NumpyFloat16DtypeArg: TypeAlias = (
    # NOTE: Alias np.float16 only on Linux x86_64, use np.half instead
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.half
    type[np.half]
    | Literal["e", "f2", "<f2", "float16", "half"]
)
NumpyFloatNot16DtypeArg: TypeAlias = (
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.single
    type[np.single]  # noqa: PYI030
    | Literal["f", "f4", "float32", "single"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double
    | type[np.double]
    | Literal["d", "f8", "float64", "double"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.longdouble
    | type[np.longdouble]
    | Literal["g", "f16", "float128", "longdouble"]
)
# PyArrow floating point types and their string aliases
PyArrowFloatDtypeArg: TypeAlias = Literal[
    "float[pyarrow]",
    "double[pyarrow]",
    "float16[pyarrow]",
    "float32[pyarrow]",
    "float64[pyarrow]",
]
FloatDtypeArg: TypeAlias = (
    BuiltinFloatDtypeArg
    | PandasFloatDtypeArg
    | NumpyFloat16DtypeArg
    | NumpyFloatNot16DtypeArg
    | PyArrowFloatDtypeArg
)
# Builtin complex type and its string alias
BuiltinComplexDtypeArg: TypeAlias = type[complex] | Literal["complex"]
PandasAstypeComplexDtypeArg: TypeAlias = (
    Literal["singlecomplex"]  # noqa: PYI030
    | Literal["cfloat", "complex_"]
    | Literal["c32", "complex256", "clongfloat", "longcomplex"]
)
# Numpy complex types and their aliases
NumpyComplexDtypeArg: TypeAlias = (
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.csingle
    type[np.csingle]  # noqa: PYI030
    | Literal["F", "c8", "complex64", "csingle"]
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.cdouble
    | type[np.cdouble]
    | Literal["D", "c16", "complex128", "cdouble"]
    #  https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.clongdouble
    # NOTE: Alias np.complex256 only on Linux x86_64, use np.clongdouble instead
    | type[np.clongdouble]
    | Literal["G", "clongdouble"]
)
ComplexDtypeArg: TypeAlias = BuiltinComplexDtypeArg | NumpyComplexDtypeArg
PandasAstypeTimedeltaDtypeArg: TypeAlias = Literal[
    "timedelta64[Y]",
    "timedelta64[M]",
    "timedelta64[W]",
    "timedelta64[D]",
    "timedelta64[h]",
    "timedelta64[m]",
    "timedelta64[μs]",
    "timedelta64[ps]",
    "timedelta64[fs]",
    "timedelta64[as]",
    # numpy type codes
    "m8[Y]",
    "m8[M]",
    "m8[W]",
    "m8[D]",
    "m8[h]",
    "m8[m]",
    "m8[μs]",
    "m8[ps]",
    "m8[fs]",
    "m8[as]",
    # little endian
    "<m8[Y]",
    "<m8[M]",
    "<m8[W]",
    "<m8[D]",
    "<m8[h]",
    "<m8[m]",
    "<m8[μs]",
    "<m8[ps]",
    "<m8[fs]",
    "<m8[as]",
]
# Refer to https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
NumpyTimedeltaDtypeArg: TypeAlias = Literal[
    "timedelta64[s]",
    "timedelta64[ms]",
    "timedelta64[us]",
    "timedelta64[ns]",
    # numpy type codes
    "m8[s]",
    "m8[ms]",
    "m8[us]",
    "m8[ns]",
    # little endian
    "<m8[s]",
    "<m8[ms]",
    "<m8[us]",
    "<m8[ns]",
]
# PyArrow duration type and its string alias
PyArrowTimedeltaDtypeArg: TypeAlias = Literal[
    "duration[s][pyarrow]",
    "duration[ms][pyarrow]",
    "duration[us][pyarrow]",
    "duration[ns][pyarrow]",
]
TimedeltaDtypeArg: TypeAlias = NumpyTimedeltaDtypeArg | PyArrowTimedeltaDtypeArg
# Pandas timestamp type and its string alias
# Not comprehensive
PandasTimestampDtypeArg: TypeAlias = (
    pd.DatetimeTZDtype
    | Literal[
        "datetime64[s, UTC]",
        "datetime64[ms, UTC]",
        "datetime64[us, UTC]",
        "datetime64[ns, UTC]",
    ]
)
PandasAstypeTimestampDtypeArg: TypeAlias = Literal[
    # numpy datetime64
    "datetime64[Y]",
    "datetime64[M]",
    "datetime64[W]",
    "datetime64[D]",
    "datetime64[h]",
    "datetime64[m]",
    "datetime64[μs]",
    "datetime64[ps]",
    "datetime64[fs]",
    "datetime64[as]",
    # numpy datetime64 type codes
    "M8[Y]",
    "M8[M]",
    "M8[W]",
    "M8[D]",
    "M8[h]",
    "M8[m]",
    "M8[μs]",
    "M8[ps]",
    "M8[fs]",
    "M8[as]",
    # little endian
    "<M8[Y]",
    "<M8[M]",
    "<M8[W]",
    "<M8[D]",
    "<M8[h]",
    "<M8[m]",
    "<M8[μs]",
    "<M8[ps]",
    "<M8[fs]",
    "<M8[as]",
]
# Numpy timestamp type and its string alias
NumpyTimestampDtypeArg: TypeAlias = Literal[
    "datetime64[s]",
    "datetime64[ms]",
    "datetime64[us]",
    "datetime64[ns]",
    # numpy type codes
    "M8[s]",
    "M8[ms]",
    "M8[us]",
    "M8[ns]",
    # little endian
    "<M8[s]",
    "<M8[ms]",
    "<M8[us]",
    "<M8[ns]",
]
# PyArrow timestamp type and its string alias
PyArrowTimestampDtypeArg: TypeAlias = Literal[
    "date32[pyarrow]",
    "date64[pyarrow]",
    "timestamp[s][pyarrow]",
    "timestamp[ms][pyarrow]",
    "timestamp[us][pyarrow]",
    "timestamp[ns][pyarrow]",
]
TimestampDtypeArg: TypeAlias = (
    PandasTimestampDtypeArg | NumpyTimestampDtypeArg | PyArrowTimestampDtypeArg
)
# Builtin str type and its string alias
BuiltinStrDtypeArg: TypeAlias = type[str] | Literal["str"]
# Pandas nullable string type and its string alias
PandasStrDtypeArg: TypeAlias = pd.StringDtype | Literal["string"]
# Numpy string type and its string alias
# https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.str_
NumpyStrDtypeArg: TypeAlias = type[np.str_] | Literal["U", "str_", "unicode"]
# PyArrow string type and its string alias
PyArrowStrDtypeArg: TypeAlias = Literal["string[pyarrow]"]
StrDtypeArg: TypeAlias = (
    BuiltinStrDtypeArg | PandasStrDtypeArg | NumpyStrDtypeArg | PyArrowStrDtypeArg
)
# Builtin bytes type and its string alias
BuiltinBytesDtypeArg: TypeAlias = type[bytes] | Literal["bytes"]
# Numpy bytes type and its string alias
# https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bytes_
NumpyBytesDtypeArg: TypeAlias = type[np.bytes_] | Literal["S", "bytes_"]
# PyArrow binary type and its string alias
PyArrowBytesDtypeArg: TypeAlias = Literal["binary[pyarrow]"]
BytesDtypeArg: TypeAlias = (
    BuiltinBytesDtypeArg | NumpyBytesDtypeArg | PyArrowBytesDtypeArg
)
CategoryDtypeArg: TypeAlias = CategoricalDtype | Literal["category"]

# Builtin object type and its string alias
BuiltinObjectDtypeArg: TypeAlias = type[object] | Literal["object"]
# Numpy object type and its string alias
# https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.object_
# NOTE: "object_" not assigned
NumpyObjectDtypeArg: TypeAlias = type[np.object_] | Literal["O"]

ObjectDtypeArg: TypeAlias = BuiltinObjectDtypeArg | NumpyObjectDtypeArg

# Numpy void type and its string alias
# https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.void
NumpyVoidDtypeArg: TypeAlias = type[np.void] | Literal["V", "void"]
VoidDtypeArg: TypeAlias = NumpyVoidDtypeArg

# DtypeArg specifies all allowable dtypes in a functions its dtype argument
DtypeArg: TypeAlias = Dtype | Mapping[Hashable, Dtype]
DtypeObj: TypeAlias = np.dtype[np.generic] | ExtensionDtype

AstypeArg: TypeAlias = (
    BooleanDtypeArg
    | IntDtypeArg
    | UIntDtypeArg
    | StrDtypeArg
    | BytesDtypeArg
    | FloatDtypeArg
    | ComplexDtypeArg
    | TimedeltaDtypeArg
    | TimestampDtypeArg
    | CategoryDtypeArg
    | ObjectDtypeArg
    | VoidDtypeArg
    | DtypeObj
)

# converters
ConvertersArg: TypeAlias = Mapping[Hashable, Callable[[Dtype], Dtype]]

# parse_dates
ParseDatesArg: TypeAlias = (
    bool | list[Hashable] | list[list[Hashable]] | Mapping[HashableT, list[HashableT2]]
)

# Not in pandas
Label: TypeAlias = Hashable | None

# For functions like rename that convert one label to another
Renamer: TypeAlias = Mapping[Any, Label] | Callable[[Any], Label]

# to maintain type information across generic functions and parametrization
T = TypeVar("T")

# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType: TypeAlias = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
TypeT = TypeVar("TypeT", bound=type)

# types of vectorized key functions for DataFrame::sort_values and
# DataFrame::sort_index, among others
ValueKeyFunc: TypeAlias = Callable[[Series], Series | AnyArrayLike] | None
IndexKeyFunc: TypeAlias = Callable[[Index], Index | AnyArrayLike] | None

# types of `func` kwarg for DataFrame.aggregate and Series.aggregate
# More specific than what is in pandas
# following Union is here to make it ty compliant https://github.com/astral-sh/ty/issues/591
AggFuncTypeBase: TypeAlias = Union[Callable, str, np.ufunc]  # noqa: UP007
AggFuncTypeDictSeries: TypeAlias = Mapping[HashableT, AggFuncTypeBase]
AggFuncTypeDictFrame: TypeAlias = Mapping[
    HashableT, AggFuncTypeBase | list[AggFuncTypeBase]
]
AggFuncTypeSeriesToFrame: TypeAlias = list[AggFuncTypeBase] | AggFuncTypeDictSeries
AggFuncTypeFrame: TypeAlias = (
    AggFuncTypeBase | list[AggFuncTypeBase] | AggFuncTypeDictFrame
)
AggFuncTypeDict: TypeAlias = AggFuncTypeDictSeries | AggFuncTypeDictFrame
AggFuncType: TypeAlias = AggFuncTypeBase | list[AggFuncTypeBase] | AggFuncTypeDict

# Not used in stubs
# AggObjType = Union[
#     "Series",
#     "DataFrame",
#     "GroupBy",
#     "SeriesGroupBy",
#     "DataFrameGroupBy",
#     "BaseWindow",
#     "Resampler",
# ]

# filenames and file-like-objects
AnyStr_co = TypeVar("AnyStr_co", str, bytes, covariant=True)
AnyStr_contra = TypeVar("AnyStr_contra", str, bytes, contravariant=True)

class BaseBuffer(Protocol):
    @property
    def mode(self) -> str:
        # for _get_filepath_or_buffer
        ...

    def seek(self, offset: int, whence: int = ..., /) -> int:
        # with one argument: gzip.GzipFile, bz2.BZ2File
        # with two arguments: zip.ZipFile, read_sas
        ...

    def seekable(self) -> bool:
        # for bz2.BZ2File
        ...

    def tell(self) -> int:
        # for zip.ZipFile, read_stata, to_stata
        ...

class ReadBuffer(BaseBuffer, Protocol[AnyStr_co]):
    def read(self, n: int = ..., /) -> AnyStr_co:
        # for BytesIOWrapper, gzip.GzipFile, bz2.BZ2File
        ...

class WriteBuffer(BaseBuffer, Protocol[AnyStr_contra]):
    def write(self, b: AnyStr_contra, /) -> Any:
        # for gzip.GzipFile, bz2.BZ2File
        ...

    def flush(self) -> Any:
        # for gzip.GzipFile, bz2.BZ2File
        ...

class ReadPickleBuffer(ReadBuffer[bytes], Protocol):
    def readline(self) -> bytes: ...

class WriteExcelBuffer(WriteBuffer[bytes], Protocol):
    def truncate(self, size: int | None = ..., /) -> int: ...

class ReadCsvBuffer(ReadBuffer[AnyStr_co], Protocol):
    def __iter__(self) -> Iterator[AnyStr_co]:
        # for engine=python
        ...

    def fileno(self) -> int:
        # for _MMapWrapper
        ...

    def readline(self) -> AnyStr_co:
        # for engine=python
        ...

    @property
    def closed(self) -> bool:
        # for engine=pyarrow
        ...

FilePath: TypeAlias = str | PathLike[str]

# for arbitrary kwargs passed during reading/writing files
StorageOptions: TypeAlias = dict[str, Any] | None

# compression keywords and compression
CompressionDict: TypeAlias = dict[str, Any]
CompressionOptions: TypeAlias = (
    None | Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"] | CompressionDict
)

# types in DataFrameFormatter
FormattersType: TypeAlias = (
    list[Callable] | tuple[Callable, ...] | Mapping[str | int, Callable]
)
# ColspaceType = Mapping[Hashable, Union[str, int]] not used in stubs
FloatFormatType: TypeAlias = str | Callable[[float], str] | EngFormatter
ColspaceArgType: TypeAlias = (
    str | int | Sequence[int | str] | Mapping[Hashable, str | int]
)

# Arguments for fillna()
FillnaOptions: TypeAlias = Literal["backfill", "bfill", "ffill", "pad"]
InterpolateOptions: TypeAlias = Literal[
    "linear",
    "time",
    "index",
    "values",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
    "from_derivatives",
]

# internals
# Manager = Union["BlockManager", "SingleBlockManager"] not used in stubs

# indexing
# PositionalIndexer -> valid 1D positional indexer, e.g. can pass
# to ndarray.__getitem__
# ScalarIndexer is for a single value as the index
# SequenceIndexer is for list like or slices (but not tuples)
# PositionalIndexerTuple is extends the PositionalIndexer for 2D arrays
# These are used in various __getitem__ overloads
# TODO(typing#684): add Ellipsis, see
# https://github.com/python/typing/issues/684#issuecomment-548203158
# https://bugs.python.org/issue41810
# Using List[int] here rather than Sequence[int] to disallow tuples.

ScalarIndexer: TypeAlias = int | np.integer
SequenceIndexer: TypeAlias = slice | list[int] | npt.NDArray[np.integer | np.bool]
PositionalIndexer: TypeAlias = ScalarIndexer | SequenceIndexer
PositionalIndexerTuple: TypeAlias = tuple[PositionalIndexer, PositionalIndexer]
# PositionalIndexer2D = Union[PositionalIndexer, PositionalIndexerTuple] Not used in stubs
TakeIndexer: TypeAlias = Sequence[int | np.integer] | npt.NDArray[np.integer | np.bool]

# Shared by functions such as drop and astype
IgnoreRaise: TypeAlias = Literal["ignore", "raise"]

# Windowing rank methods
WindowingRankType: TypeAlias = Literal["average", "min", "max"]

# read_csv engines
CSVEngine: TypeAlias = Literal["c", "python", "pyarrow", "python-fwf"]

# read_json engines
JSONEngine: TypeAlias = Literal["ujson", "pyarrow"]

# read_xml parsers
XMLParsers: TypeAlias = Literal["lxml", "etree"]

# read_html flavors
HTMLFlavors: TypeAlias = Literal["lxml", "html5lib", "bs4"]

# Interval closed type
IntervalT = TypeVar("IntervalT", bound=Interval)
IntervalLeftRight: TypeAlias = Literal["left", "right"]
IntervalClosedType: TypeAlias = IntervalLeftRight | Literal["both", "neither"]

# datetime and NaTType
# DatetimeNaTType = datetime |  NaTType not used in stubs
RaiseCoerce: TypeAlias = Literal["raise", "coerce"]
DateTimeErrorChoices: TypeAlias = RaiseCoerce

# sort_index
SortKind: TypeAlias = Literal["quicksort", "mergesort", "heapsort", "stable"]
NaPosition: TypeAlias = Literal["first", "last"]

# Arguments for nsmallest and nlargest
NsmallestNlargestKeep: TypeAlias = Literal["first", "last", "all"]

# quantile interpolation
QuantileInterpolation: TypeAlias = Literal[
    "linear", "lower", "higher", "midpoint", "nearest"
]

# plotting
# PlottingOrientation = Literal["horizontal", "vertical"] not used in stubs

# dropna
AnyAll: TypeAlias = Literal["any", "all"]

# merge
# defined in a different manner, but equivalent to pandas
JoinHow: TypeAlias = Literal["left", "right", "outer", "inner"]
MergeHow: TypeAlias = JoinHow | Literal["cross", "left_anti", "right_anti"]
ValidationOptions: TypeAlias = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]
MergeValidate: TypeAlias = ValidationOptions
JoinValidate: TypeAlias = ValidationOptions

# reindex
ReindexMethod: TypeAlias = FillnaOptions | Literal["nearest"]

# MatplotlibColor = Union[str, Sequence[float]] # not used in stubs
TimeGrouperOrigin: TypeAlias = (
    Timestamp | Literal["epoch", "start", "start_day", "end", "end_day"]
)
TimeAmbiguous: TypeAlias = Literal["infer", "NaT", "raise"] | npt.NDArray[np.bool_]
# Note this is same as TimestampNonexistent - defined both ways in pandas
TimeNonexistent: TypeAlias = (
    Literal["shift_forward", "shift_backward", "NaT", "raise"]
    | Timedelta
    | datetime.timedelta
)
DropKeep: TypeAlias = Literal["first", "last", False]
CorrelationMethod: TypeAlias = (
    Literal["pearson", "kendall", "spearman"]
    | Callable[[npt.NDArray[Any], npt.NDArray[Any]], float]
)
AlignJoin: TypeAlias = Literal["outer", "inner", "left", "right"]
DtypeBackend: TypeAlias = Literal["pyarrow", "numpy_nullable"]

TimeUnit: TypeAlias = Literal["s", "ms", "us", "ns"]
OpenFileErrors: TypeAlias = Literal[
    "strict",
    "ignore",
    "replace",
    "surrogateescape",
    "xmlcharrefreplace",
    "backslashreplace",
    "namereplace",
]

# update
UpdateJoin: TypeAlias = Literal["left"]

# applymap
# NaAction = Literal["ignore"] not used in stubs or pandas

# from_dict
FromDictOrient: TypeAlias = Literal["columns", "index", "tight"]

# to_stata
ToStataByteorder: TypeAlias = Literal[">", "<", "little", "big"]

# ExcelWriter
ExcelWriterIfSheetExists: TypeAlias = Literal["error", "new", "replace", "overlay"]
ExcelWriterMergeCells: TypeAlias = bool | Literal["columns"]

# Offsets
# OffsetCalendar = Union[np.busdaycalendar, "AbstractHolidayCalendar"] not used in stubs or pandas

# read_csv: usecols
UsecolsArgType: TypeAlias = (
    SequenceNotStr[Hashable] | range | AnyArrayLike | Callable[[HashableT], bool] | None
)

# maintain the sub-type of any hashable sequence
# SequenceT = TypeVar("SequenceT", bound=Sequence[Hashable]) not used in stubs

SliceType: TypeAlias = Hashable | None

######
## All types below this point are only used in pandas-stubs
######

BuiltinDtypeArg: TypeAlias = (
    BuiltinBooleanDtypeArg
    | BuiltinIntDtypeArg
    | BuiltinFloatDtypeArg
    | BuiltinComplexDtypeArg
    | BuiltinStrDtypeArg
    | BuiltinBytesDtypeArg
    | BuiltinObjectDtypeArg
)
NumpyNotTimeDtypeArg: TypeAlias = (
    NumpyBooleanDtypeArg
    | NumpyIntDtypeArg
    | NumpyUIntDtypeArg
    | NumpyFloat16DtypeArg
    | NumpyFloatNot16DtypeArg
    | NumpyComplexDtypeArg
    | NumpyStrDtypeArg
    | NumpyBytesDtypeArg
    | NumpyObjectDtypeArg
    | NumpyVoidDtypeArg
)
PyArrowNotStrDtypeArg: TypeAlias = (
    PyArrowBooleanDtypeArg
    | PyArrowIntDtypeArg
    | PyArrowUIntDtypeArg
    | PyArrowFloatDtypeArg
    | PyArrowTimedeltaDtypeArg
    | PyArrowTimestampDtypeArg
    | PyArrowBytesDtypeArg
)

StrLike: TypeAlias = str | np.str_

ScalarT = TypeVar("ScalarT", bound=Scalar)
# Refine the definitions below in 3.9 to use the specialized type.
np_num: TypeAlias = np.bool | np.integer | np.floating | np.complexfloating
np_ndarray_intp: TypeAlias = npt.NDArray[np.intp]
np_ndarray_int64: TypeAlias = npt.NDArray[np.int64]
np_ndarray_int: TypeAlias = npt.NDArray[np.signedinteger]
np_ndarray_anyint: TypeAlias = npt.NDArray[np.integer]
np_ndarray_float: TypeAlias = npt.NDArray[np.floating]
np_ndarray_complex: TypeAlias = npt.NDArray[np.complexfloating]
np_ndarray_bool: TypeAlias = npt.NDArray[np.bool_]
np_ndarray_str: TypeAlias = npt.NDArray[np.str_]
np_ndarray_dt: TypeAlias = npt.NDArray[np.datetime64]
np_ndarray_td: TypeAlias = npt.NDArray[np.timedelta64]

# Define shape and generic type variables with defaults similar to numpy
GenericT = TypeVar("GenericT", bound=np.generic, default=Any)
GenericT_co = TypeVar("GenericT_co", bound=np.generic, default=Any, covariant=True)
GenericT_contra = TypeVar(
    "GenericT_contra", bound=np.generic, default=Any, contravariant=True
)
NpNumT = TypeVar("NpNumT", bound=np_num, default=np_num)
ShapeT = TypeVar("ShapeT", bound=tuple[int, ...], default=tuple[Any, ...])
# Numpy ndarray with more ergonomic typevar
np_ndarray: TypeAlias = np.ndarray[ShapeT, np.dtype[GenericT]]
np_ndarray_num: TypeAlias = np_ndarray[ShapeT, NpNumT]

# Numpy arrays with known shape (Do not use as argument types, only as return types)
np_1darray: TypeAlias = np.ndarray[tuple[int], np.dtype[GenericT]]
np_1darray_str: TypeAlias = np_1darray[np.str_]
np_1darray_bytes: TypeAlias = np_1darray[np.bytes_]
np_1darray_complex: TypeAlias = np_1darray[np.complexfloating]
np_1darray_object: TypeAlias = np_1darray[np.object_]
np_1darray_bool: TypeAlias = np_1darray[np.bool]
np_1darray_intp: TypeAlias = np_1darray[np.intp]
np_1darray_int64: TypeAlias = np_1darray[np.int64]
np_1darray_anyint: TypeAlias = np_1darray[np.integer]
np_1darray_float: TypeAlias = np_1darray[np.floating]
np_1darray_dt: TypeAlias = np_1darray[np.datetime64]
np_1darray_td: TypeAlias = np_1darray[np.timedelta64]
np_2darray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[GenericT]]

DtypeNp = TypeVar("DtypeNp", bound=np.dtype[np.generic])
KeysArgType: TypeAlias = Any
ListLikeT = TypeVar("ListLikeT", bound=ListLike)
ListLikeExceptSeriesAndStr: TypeAlias = (
    MutableSequence[Any] | np_1darray | tuple[Any, ...] | Index
)
ListLikeU: TypeAlias = Sequence | np_1darray | Series | Index
ListLikeHashable: TypeAlias = (
    MutableSequence[HashableT] | np_1darray | tuple[HashableT, ...] | range
)

class SupportsDType(Protocol[GenericT_co]):
    @property
    def dtype(self) -> np.dtype[GenericT_co]: ...

# Similar to npt.DTypeLike but leaves out np.dtype and None for use in overloads
DTypeLike: TypeAlias = type[Any] | tuple[Any, Any] | list[Any] | str

if TYPE_CHECKING:  # noqa: PYI002
    IndexType: TypeAlias = slice | np_ndarray_anyint | Index | list[int] | Series[int]
    MaskType: TypeAlias = Series[bool] | np_ndarray_bool | list[bool]

# Scratch types for generics

T_INT = TypeVar("T_INT", bound=int)
T_COMPLEX = TypeVar("T_COMPLEX", bound=complex)
SeriesDTypeNoStrDateTime: TypeAlias = (
    bytes
    | bool
    | int
    | float
    | complex
    | NpDtypeNoStr
    | ExtensionDtype
    | Period
    | Interval
    | CategoricalDtype
    | BaseOffset
)
SeriesDTypeNoDateTime: TypeAlias = (
    str | SeriesDTypeNoStrDateTime | type[str] | list[str]
)
SeriesDType: TypeAlias = (
    SeriesDTypeNoDateTime
    | datetime.date
    | datetime.time
    | datetime.datetime  # includes pd.Timestamp
    | datetime.timedelta  # includes pd.Timedelta
)
S1 = TypeVar("S1", bound=SeriesDType, default=Any)
# Like S1, but without `default=Any`.
S2 = TypeVar("S2", bound=SeriesDType)
S2_contra = TypeVar("S2_contra", bound=SeriesDType, contravariant=True)
S2_NDT_contra = TypeVar(
    "S2_NDT_contra", bound=SeriesDTypeNoDateTime, contravariant=True
)
S2_NSDT = TypeVar("S2_NSDT", bound=SeriesDTypeNoStrDateTime)
S3 = TypeVar("S3", bound=SeriesDType)

# Constraint, instead of bound
C2 = TypeVar(
    "C2",
    str,
    bytes,
    datetime.date,
    datetime.time,
    bool,
    int,
    float,
    complex,
    Dtype,
    datetime.datetime,  # includes pd.Timestamp
    datetime.timedelta,  # includes pd.Timedelta
    Period,
    Interval,
    CategoricalDtype,
    BaseOffset,
)

IndexingInt: TypeAlias = (
    int | np.int_ | np.integer | np.unsignedinteger | np.signedinteger | np.int8
)

# AxesData is used for data for Index
AxesData: TypeAlias = Mapping[S3, Any] | Axes | KeysView[S3]

# Any plain Python or numpy function
Function: TypeAlias = np.ufunc | Callable[..., Any]
# Use a distinct HashableT in shared types to avoid conflicts with
# shared HashableT and HashableT#. This one can be used if the identical
# type is need in a function that uses GroupByObjectNonScalar
_HashableTa = TypeVar("_HashableTa", bound=Hashable)
if TYPE_CHECKING:  # noqa: PYI002
    ByT = TypeVar(
        "ByT",
        bound=str
        | bytes
        | datetime.date
        | datetime.datetime
        | datetime.timedelta
        | np.datetime64
        | np.timedelta64
        | bool
        | int
        | float
        | complex
        | Scalar
        | Period
        | Interval[int | float | Timestamp | Timedelta]
        | tuple,
    )
    # Use a distinct SeriesByT when using groupby with Series of known dtype.
    # Essentially, an intersection between Series S1 TypeVar, and ByT TypeVar
    SeriesByT = TypeVar(
        "SeriesByT",
        bound=str
        | bytes
        | datetime.date
        | bool
        | int
        | float
        | complex
        | datetime.datetime
        | datetime.timedelta
        | Period
        | Interval[int | float | Timestamp | Timedelta],
    )
GroupByObjectNonScalar: TypeAlias = (
    tuple
    | list[_HashableTa]
    | Function
    | list[Function]
    | list[Series]
    | np_ndarray
    | list[np_ndarray]
    | Mapping[Label, Any]
    | list[Mapping[Label, Any]]
    | list[Index]
    | Grouper
    | list[Grouper]
)
GroupByObject: TypeAlias = Scalar | Index | GroupByObjectNonScalar | Series

StataDateFormat: TypeAlias = Literal[
    "tc",
    "%tc",
    "td",
    "%td",
    "tw",
    "%tw",
    "tm",
    "%tm",
    "tq",
    "%tq",
    "th",
    "%th",
    "ty",
    "%ty",
]

# Can be passed to `to_replace`, `value`, or `regex` in `Series.replace`.
# `DataFrame.replace` also accepts mappings of these.
ReplaceValue: TypeAlias = (
    Scalar
    | Pattern
    | NAType
    | Sequence[Scalar | Pattern]
    | Mapping[HashableT, ScalarT]
    | Series
    | None
)

JsonFrameOrient: TypeAlias = Literal[
    "split", "records", "index", "columns", "values", "table"
]
JsonSeriesOrient: TypeAlias = Literal["split", "records", "index", "table"]

TimestampConvention: TypeAlias = Literal["start", "end", "s", "e"]

# [pandas-dev/pandas-stubs/991]
# Ref: https://github.com/python/cpython/blob/5a4fb7ea1c96f67dbb3df5d4ccaf3f66a1e19731/Modules/_csv.c#L88-L91
# QUOTE_MINIMAL = 0
# QUOTE_ALL = 1
# QUOTE_NONNUMERIC = 2
# QUOTE_NONE = 3
# Added in 3.12:
# QUOTE_STRINGS = 4
# QUOTE_NOTNULL = 5
CSVQuotingCompat: TypeAlias = Literal[0, 1, 2, 3]
if sys.version_info >= (3, 12):
    CSVQuoting: TypeAlias = CSVQuotingCompat | Literal[4, 5]
else:
    CSVQuoting: TypeAlias = CSVQuotingCompat

HDFCompLib: TypeAlias = Literal["zlib", "lzo", "bzip2", "blosc"]
ParquetEngine: TypeAlias = Literal["auto", "pyarrow", "fastparquet"]
FileWriteMode: TypeAlias = Literal[
    "a", "w", "x", "at", "wt", "xt", "ab", "wb", "xb", "w+", "w+b", "a+", "a+b"
]

WindowingEngine: TypeAlias = Literal["cython", "numba"] | None

class _WindowingNumbaKwargs(TypedDict, total=False):
    nopython: bool
    nogil: bool
    parallel: bool

WindowingEngineKwargs: TypeAlias = _WindowingNumbaKwargs | None

class StyleExportDict(TypedDict, total=False):
    apply: Any
    table_attributes: Any
    table_styles: Any
    hide_index: bool
    hide_columns: bool
    hide_index_names: bool
    hide_column_names: bool
    css: dict[str, str | int]

CalculationMethod: TypeAlias = Literal["single", "table"]

ExcelReadEngine: TypeAlias = Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"]
ExcelWriteEngine: TypeAlias = Literal["openpyxl", "odf", "xlsxwriter"]

# Repeated in `timestamps.pyi` so as to satisfy mixed strict / non-strict paths.
# https://github.com/pandas-dev/pandas-stubs/pull/1151#issuecomment-2715130190
TimeZones: TypeAlias = str | tzinfo | None | int

# Evaluates to a DataFrame column in DataFrame.assign context.
IntoColumn: TypeAlias = (
    AnyArrayLike
    | Scalar
    | Callable[[DataFrame], AnyArrayLike | Scalar | Sequence[Scalar] | range | None]
    | Sequence[Scalar]
    | range
    | None
)

DatetimeLike: TypeAlias = datetime.datetime | np.datetime64 | Timestamp
DateAndDatetimeLike: TypeAlias = datetime.date | DatetimeLike

DatetimeDictArg: TypeAlias = (
    Sequence[int] | Sequence[float] | list[str] | tuple[Scalar, ...] | AnyArrayLike
)
DictConvertible: TypeAlias = FulldatetimeDict | DataFrame

# `Incomplete` is equivalent to `Any`. Use it to annotate symbols that you don't
# know the type of yet and that should be changed in the future. Use `Any` only
# where it is the only acceptable type.
Incomplete: TypeAlias = Any

# differentiating between bool and int/float/complex
# https://github.com/pandas-dev/pandas-stubs/pull/1312#pullrequestreview-3126128971
class Just(Protocol, Generic[T]):
    @property  # type: ignore[override]
    @override
    def __class__(self, /) -> type[T]: ...  # pyrefly: ignore[bad-override]
    @__class__.setter
    @override
    def __class__(self, t: type[T], /) -> None: ...

class SupportsTrueDiv(Protocol[_T_contra, _T_co]):
    def __truediv__(self, x: _T_contra, /) -> _T_co: ...

class SupportsRTrueDiv(Protocol[_T_contra, _T_co]):
    def __rtruediv__(self, x: _T_contra, /) -> _T_co: ...

__all__ = ["npt", "type_t"]
