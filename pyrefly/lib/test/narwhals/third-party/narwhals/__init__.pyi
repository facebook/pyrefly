"""Minimal Narwhals stubs for schema-tracking tests."""

from narwhals.dataframe import DataFrame as DataFrame, LazyFrame as LazyFrame
from narwhals.dtypes import (
    Array as Array,
    Boolean as Boolean,
    Float32 as Float32,
    Float64 as Float64,
    Int8 as Int8,
    Int16 as Int16,
    Int32 as Int32,
    Int64 as Int64,
    Int128 as Int128,
    List as List,
    String as String,
    Struct as Struct,
    UInt8 as UInt8,
    UInt64 as UInt64,
    UInt128 as UInt128,
)
from narwhals.expr import Expr as Expr
from narwhals.functions import (
    col as col,
    concat as concat,
    len_ as len,
    lit as lit,
    when as when,
)
from narwhals.schema import Schema as Schema
from narwhals.series import Series as Series
from narwhals.translate import from_native as from_native, to_native as to_native
