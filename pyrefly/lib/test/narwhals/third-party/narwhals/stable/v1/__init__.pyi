"""The stable namespace re-declares the public API as subclasses of the main one."""

from typing import Iterable

from narwhals.dataframe import DataFrame as NwDataFrame, LazyFrame as NwLazyFrame
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
from narwhals.expr import Expr as NwExpr
from narwhals.functions import Then, When
from narwhals.schema import Schema as NwSchema
from narwhals.series import Series as NwSeries

class DataFrame(NwDataFrame): ...
class LazyFrame(NwLazyFrame): ...
class Series(NwSeries): ...
class Expr(NwExpr): ...
class Schema(NwSchema): ...

def col(*names: str | Iterable[str]) -> Expr: ...
def lit(value: object, dtype: object = None) -> Expr: ...
def len() -> Expr: ...
def concat(items: Iterable[DataFrame], *, how: str = "vertical") -> DataFrame: ...
def when(*predicates: object) -> When: ...
def from_native(native_object: object, *, eager_only: bool = False) -> DataFrame: ...
def to_native(narwhals_object: object) -> object: ...
