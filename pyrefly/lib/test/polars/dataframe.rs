/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

/// A minimal Polars stub: `DataFrame` is defined in `polars.dataframe.frame` and
/// re-exported from `polars`, and its column-access methods return an opaque type.
fn env_with_polars_stubs() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "polars.series.series",
        "polars/series/series.pyi",
        r#"
from typing import Any, overload
class Series:
    def __init__(self, name: object = None, values: object = None, dtype: object = None, *, strict: bool = True, nan_to_null: bool = False) -> None: ...
    @overload
    def __getitem__(self, key: int) -> Any: ...
    @overload
    def __getitem__(self, key: slice) -> "Series": ...
    def __or__(self, other: "Series") -> "Series": ...
"#,
    );
    env.add_with_path(
        "polars.dataframe.frame",
        "polars/dataframe/frame.pyi",
        r#"
from typing import Iterator, overload
from polars.series.series import Series
from polars.lazyframe.frame import LazyFrame
class DataFrame:
    columns: list[str]
    def __init__(self, data: object = None, schema: object = None, schema_overrides: object = None, strict: bool = True) -> None: ...
    @overload
    def __getitem__(self, key: str) -> Series: ...
    @overload
    def __getitem__(self, key: list[str] | list[int]) -> "DataFrame": ...
    def __iter__(self) -> Iterator[Series]: ...
    def __contains__(self, key: str) -> bool: ...
    def get_column(self, name: str, *, default: object = None) -> Series: ...
    def to_series(self, index: int = 0) -> Series: ...
    def head(self, n: int = 5) -> "DataFrame": ...
    def select(self, *exprs: object, **named_exprs: object) -> "DataFrame": ...
    def drop(self, *columns: object, strict: bool = True) -> "DataFrame": ...
    def rename(self, mapping: object, *, strict: bool = True) -> "DataFrame": ...
    def with_columns(self, *exprs: object, **named_exprs: object) -> "DataFrame": ...
    def filter(self, *predicates: object, **constraints: object) -> "DataFrame": ...
    def sort(self, by: object, *more: object, descending: bool = False) -> "DataFrame": ...
    def fill_null(self, value: object = None) -> "DataFrame": ...
    def slice(self, offset: int, length: int | None = None) -> "DataFrame": ...
    def unique(self, subset: object = None, *, keep: str = "any", maintain_order: bool = False) -> "DataFrame": ...
    def drop_nulls(self, subset: object = None) -> "DataFrame": ...
    def cast(self, dtypes: object, *, strict: bool = True) -> "DataFrame": ...
    def join(self, other: "DataFrame", on: object = None, how: str = "inner", *, left_on: object = None, right_on: object = None, suffix: str = "_right", coalesce: object = None) -> "DataFrame": ...
    def hstack(self, columns: object, *, in_place: bool = False) -> "DataFrame": ...
    def vstack(self, other: "DataFrame", *, in_place: bool = False) -> "DataFrame": ...
    def extend(self, other: "DataFrame") -> "DataFrame": ...
    def insert_column(self, index: int, column: object) -> "DataFrame": ...
    def replace_column(self, index: int, column: object) -> "DataFrame": ...
    def group_by(self, *by: object, maintain_order: bool = False, **named_by: object) -> "GroupBy": ...
    def lazy(self) -> "LazyFrame": ...
class GroupBy:
    def agg(self, *aggs: object, **named_aggs: object) -> DataFrame: ...
"#,
    );
    env.add_with_path(
        "polars.lazyframe.frame",
        "polars/lazyframe/frame.pyi",
        r#"
from typing import Literal
from polars.dataframe.frame import DataFrame
class LazyFrame:
    def select(self, *exprs: object, **named_exprs: object) -> "LazyFrame": ...
    def drop(self, *columns: object, strict: bool = True) -> "LazyFrame": ...
    def rename(self, mapping: object, *, strict: bool = True) -> "LazyFrame": ...
    def with_columns(self, *exprs: object, **named_exprs: object) -> "LazyFrame": ...
    def filter(self, *predicates: object, **constraints: object) -> "LazyFrame": ...
    def sort(self, by: object, *more: object, descending: bool = False) -> "LazyFrame": ...
    def collect(self, *, engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto") -> DataFrame: ...
"#,
    );
    env.add_with_path(
        "polars.functions.eager",
        "polars/functions/eager.pyi",
        r#"
from typing import Iterable
from polars.dataframe.frame import DataFrame
def concat(items: Iterable[DataFrame], *, how: str = "vertical", rechunk: bool = False, parallel: bool = True) -> DataFrame: ...
"#,
    );
    env.add_with_path(
        "polars.expr.expr",
        "polars/expr/expr.pyi",
        r#"
from typing import Any
class Expr:
    def __add__(self, other: Any) -> "Expr": ...
    def __radd__(self, other: Any) -> "Expr": ...
    def __sub__(self, other: Any) -> "Expr": ...
    def __mul__(self, other: Any) -> "Expr": ...
    def __truediv__(self, other: Any) -> "Expr": ...
    def __floordiv__(self, other: Any) -> "Expr": ...
    def __mod__(self, other: Any) -> "Expr": ...
    def __pow__(self, other: Any) -> "Expr": ...
    def __and__(self, other: Any) -> "Expr": ...
    def __or__(self, other: Any) -> "Expr": ...
    def __xor__(self, other: Any) -> "Expr": ...
    def __neg__(self) -> "Expr": ...
    def __pos__(self) -> "Expr": ...
    def __invert__(self) -> "Expr": ...
    def __gt__(self, other: Any) -> "Expr": ...
    def __ge__(self, other: Any) -> "Expr": ...
    def __lt__(self, other: Any) -> "Expr": ...
    def __le__(self, other: Any) -> "Expr": ...
    def alias(self, name: str) -> "Expr": ...
    def cast(self, dtype: Any, *, strict: bool = True) -> "Expr": ...
    def sum(self) -> "Expr": ...
    def mean(self) -> "Expr": ...
    def median(self) -> "Expr": ...
    def std(self, ddof: int = 1) -> "Expr": ...
    def var(self, ddof: int = 1) -> "Expr": ...
    def min(self) -> "Expr": ...
    def max(self) -> "Expr": ...
    def first(self) -> "Expr": ...
    def last(self) -> "Expr": ...
    def product(self) -> "Expr": ...
    def count(self) -> "Expr": ...
    def n_unique(self) -> "Expr": ...
"#,
    );
    env.add_with_path(
        "polars.functions.col",
        "polars/functions/col.pyi",
        r#"
from polars.expr.expr import Expr
def col(*names: str) -> Expr: ...
"#,
    );
    env.add_with_path(
        "polars.functions.lit",
        "polars/functions/lit.pyi",
        r#"
from polars.expr.expr import Expr
def lit(value: object, dtype: object = None) -> Expr: ...
"#,
    );
    env.add_with_path(
        "polars.functions.len",
        "polars/functions/len.pyi",
        r#"
from polars.expr.expr import Expr
def len() -> Expr: ...
"#,
    );
    env.add_with_path(
        "polars.schema",
        "polars/schema.pyi",
        r#"
class Schema:
    def __init__(self, schema: object = None) -> None: ...
"#,
    );
    env.add(
        "polars",
        r#"
from polars.dataframe.frame import DataFrame as DataFrame
from polars.series.series import Series as Series
from polars.functions.eager import concat as concat
from polars.functions.col import col as col
from polars.functions.lit import lit as lit
from polars.functions.len import len as len
from polars.expr.expr import Expr as Expr
from polars.schema import Schema as Schema
class Int8: ...
class Int16: ...
class Int32: ...
class Int64: ...
class Int128: ...
class UInt8: ...
class UInt64: ...
class UInt128: ...
class Float32: ...
class Float64: ...
class String: ...
class Boolean: ...
"#,
    );
    env.add(
        "mymod",
        r#"
class Schema:
    def __init__(self, schema: object = None) -> None: ...
"#,
    );
    env
}

/// Polars stubs plus a module whose top-level `df` carries an inferred schema, so
/// tests can pin that the schema survives the import boundary.
fn env_cross_file() -> TestEnv {
    let mut env = env_with_polars_stubs();
    env.add(
        "defs",
        r#"
import polars as pl
df = pl.DataFrame({"a": [1], "b": ["x"]})
df_kw = pl.DataFrame(data={"a": [1], "b": ["x"]})
df_schema = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
df_records = pl.DataFrame([{"a": 1}, {"b": 2}])
my_schema = pl.Schema({"a": pl.Int64})
s = pl.Series("a", [1])
"#,
    );
    env
}

/// A minimal pandas stub: `DataFrame` lives in `pandas.core.frame` and is re-exported
/// from `pandas`. A pandas frame is mutable, so its inferred schema is Partial.
fn env_with_pandas_stubs() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "pandas.core.frame",
        "pandas/core/frame.pyi",
        r#"
class DataFrame:
    def __init__(self, data: object = None, index: object = None, columns: object = None) -> None: ...
"#,
    );
    env.add(
        "pandas",
        r#"
from pandas.core.frame import DataFrame as DataFrame
"#,
    );
    env
}

/// Polars stubs plus a pandas `DataFrame` at its real qname, so tests can pin cross-library
/// behavior where a pandas frame is passed to a Polars method.
fn env_with_polars_and_pandas_stubs() -> TestEnv {
    let mut env = env_with_polars_stubs();
    env.add_with_path(
        "pandas.core.frame",
        "pandas/core/frame.pyi",
        r#"
class Series: ...
class DataFrame:
    columns: list[str]
    def __init__(self, data: object = None, columns: object = None, dtype: object = None) -> None: ...
    def __getitem__(self, key: str) -> Series: ...
"#,
    );
    env.add(
        "pandas",
        "from pandas.core.frame import DataFrame as DataFrame",
    );
    env
}

testcase!(
    test_construct_int_and_str_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_columns_in_source_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"b": ["x"], "a": [1]}))  # E: revealed type: DataFrame[b: String, a: Int64]
"#,
);

testcase!(
    test_non_polars_table_untouched,
    env_with_polars_stubs(),
    r#"
from typing import reveal_type
class DataFrame:
    def __init__(self, data: object = None) -> None: ...
reveal_type(DataFrame({"a": [1]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_non_string_key,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({1: [1]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_degrade_scalar_value,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": 1}))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_degrade_non_literal_element,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
x: int = 1
reveal_type(pl.DataFrame({"a": [x]}))  # E: revealed type: DataFrame[a: Unknown]
def g() -> int: ...
reveal_type(pl.DataFrame({"b": [g()]}))  # E: revealed type: DataFrame[b: Unknown]
"#,
);

testcase!(
    test_construct_incompatible_mix_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, "s"]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_construct_int_then_float_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2.0]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_construct_float_then_int_widens_to_float,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [2.0, 1]}))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_construct_float_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1.0, 2.0]}))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_construct_bool_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [True, False]}))  # E: revealed type: DataFrame[a: Boolean]
"#,
);

testcase!(
    test_construct_bytes_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [b"x", b"y"]}))  # E: revealed type: DataFrame[a: Binary]
"#,
);

testcase!(
    test_construct_date_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [date(2020, 1, 1)]}))  # E: revealed type: DataFrame[a: Date]
"#,
);

testcase!(
    test_construct_datetime_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import datetime
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [datetime(2020, 1, 1, 3, 4, 5)]}))  # E: revealed type: DataFrame[a: Datetime]
"#,
);

testcase!(
    test_construct_time_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import time
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [time(1, 2, 3)]}))  # E: revealed type: DataFrame[a: Time]
"#,
);

testcase!(
    test_construct_duration_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import timedelta
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [timedelta(days=1)]}))  # E: revealed type: DataFrame[a: Duration]
"#,
);

testcase!(
    test_construct_datetime_tz_drops_timezone,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import datetime, timezone
from typing import reveal_type
# Our model carries no time unit or timezone, so a tz-aware value still records plain `Datetime`.
reveal_type(pl.DataFrame({"a": [datetime(2020, 1, 1, tzinfo=timezone.utc)]}))  # E: revealed type: DataFrame[a: Datetime]
"#,
);

testcase!(
    test_construct_date_multi_element,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [date(2020, 1, 1), date(2021, 1, 1)]}))  # E: revealed type: DataFrame[a: Date]
"#,
);

testcase!(
    test_construct_temporal_and_plain_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date
from typing import reveal_type
reveal_type(pl.DataFrame({"d": [date(2020, 1, 1)], "n": [1]}))  # E: revealed type: DataFrame[d: Date, n: Int64]
"#,
);

testcase!(
    test_construct_date_then_datetime_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date, datetime
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [date(2020, 1, 1), datetime(2020, 1, 1)]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Date`
"#,
);

testcase!(
    test_construct_datetime_then_date_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date, datetime
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [datetime(2020, 1, 1), date(2020, 1, 1)]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Datetime`
"#,
);

testcase!(
    test_construct_date_then_int_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [date(2020, 1, 1), 5]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Date`
"#,
);

testcase!(
    test_construct_int_then_date_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [5, date(2020, 1, 1)]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_construct_temporal_strict_false_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date, datetime
from typing import reveal_type
# Mixed temporal supertypes are not modeled, so we do not guess the runtime `Datetime`.
reveal_type(pl.DataFrame({"a": [date(2020, 1, 1), datetime(2020, 1, 1)]}, strict=False))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_construct_date_then_none_keeps_date,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date
from typing import reveal_type
# A `None` contributes `Null`, which takes the other side, so the column stays `Date`.
reveal_type(pl.DataFrame({"a": [date(2020, 1, 1), None]}))  # E: revealed type: DataFrame[a: Date]
"#,
);

testcase!(
    test_construct_int_then_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, None]}))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_construct_none_then_int,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A leading `None` never anchors the column; the anchor is the first non-null element.
reveal_type(pl.DataFrame({"a": [None, 1]}))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_construct_single_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [None]}))  # E: revealed type: DataFrame[a: Null]
"#,
);

testcase!(
    test_construct_all_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [None, None]}))  # E: revealed type: DataFrame[a: Null]
"#,
);

testcase!(
    test_construct_float_then_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1.0, None]}))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_construct_string_then_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": ["x", None]}))  # E: revealed type: DataFrame[a: String]
"#,
);

testcase!(
    test_construct_bool_then_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [True, None]}))  # E: revealed type: DataFrame[a: Boolean]
"#,
);

testcase!(
    test_construct_none_then_bool,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [None, True]}))  # E: revealed type: DataFrame[a: Boolean]
"#,
);

testcase!(
    test_construct_bytes_then_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [b"x", None]}))  # E: revealed type: DataFrame[a: Binary]
"#,
);

testcase!(
    test_construct_none_then_date,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [None, date(2020, 1, 1)]}))  # E: revealed type: DataFrame[a: Date]
"#,
);

testcase!(
    test_construct_datetime_then_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import datetime
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [datetime(2020, 1, 1), None]}))  # E: revealed type: DataFrame[a: Datetime]
"#,
);

testcase!(
    test_construct_int_none_float_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, None, 2.0]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_construct_leading_none_then_int_float_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# The anchor comes from the first non-null element `1`, so the trailing float still does not fit.
reveal_type(pl.DataFrame({"a": [None, 1, 2.0]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_construct_int_none_string_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, None, "x"]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_construct_int_none_float_strict_false_widens,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, None, 2.0]}, strict=False))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_construct_leading_none_int_float_strict_false_widens,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [None, 1, 2.0]}, strict=False))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_construct_single_none_strict_false,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [None]}, strict=False))  # E: revealed type: DataFrame[a: Null]
"#,
);

testcase!(
    test_construct_none_columns_independent,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, None], "b": [None]}))  # E: revealed type: DataFrame[a: Int64, b: Null]
"#,
);

testcase!(
    test_construct_shadowed_date_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A shadowed `date` must not fabricate a temporal dtype.
def date() -> str: ...
reveal_type(pl.DataFrame({"a": [date()]}))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_construct_temporal_variable_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date, datetime
from typing import reveal_type
# A `date` variable may hold a `datetime` subclass, so only direct constructors are trusted.
def f(d: date) -> None:
    reveal_type(pl.DataFrame({"a": [d, datetime(2020, 1, 1)]}))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_construct_datetime_tz_mix_strict_true_reports_datetime,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import datetime, timezone
from typing import reveal_type
# Under the default strict=True Polars coerces a naive/tz-aware mix into one Datetime column, so
# reporting `Datetime` matches the runtime even though we do not model the timezone.
reveal_type(pl.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 1, tzinfo=timezone.utc)]}))  # E: revealed type: DataFrame[a: Datetime]
"#,
);

testcase!(
    test_construct_datetime_tz_mix_strict_false_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import datetime, timezone
from typing import reveal_type
# Static types do not distinguish naive and timezone-aware datetimes.
reveal_type(pl.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 1, tzinfo=timezone.utc)]}, strict=False))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_construct_datetime_multi_strict_false_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import datetime
from typing import reveal_type
# Static types cannot prove that every datetime shares a timezone.
reveal_type(pl.DataFrame({"a": [datetime(2020, 1, 1), datetime(2021, 1, 1)]}, strict=False))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_degrade_complex_not_modeled,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Polars stores complex values as `Object`.
reveal_type(pl.DataFrame({"a": [1j]}))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_construct_i64_max_is_int64,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [9223372036854775807]}))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_construct_int_above_i64_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Past i64 the runtime dtype is data-shape dependent (UInt64 or Int128), so we degrade rather than
# claim Int64.
reveal_type(pl.DataFrame({"a": [9223372036854775808]}))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_construct_int_then_bool_is_int,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, True]}))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_construct_bool_then_int_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [True, 1]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Boolean`
"#,
);

testcase!(
    test_construct_empty_list_unknown_element,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": []}))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_construct_multi_column_with_uncertain_elements,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": [], "c": [2.0, 1]}))  # E: revealed type: DataFrame[a: Int64, b: Unknown, c: Float64]
"#,
);

testcase!(
    test_degrade_mixed_literal_and_non_literal,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
x: int = 1
reveal_type(pl.DataFrame({"a": [1, x]}))  # E: revealed type: DataFrame[a: Unknown]
def g() -> int: ...
reveal_type(pl.DataFrame({"b": [2, g()]}))  # E: revealed type: DataFrame[b: Unknown]
"#,
);

testcase!(
    test_fallback_empty_dict,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_construct_from_data_keyword,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"a": [1]}))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_data_keyword_two_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"a": [1, 2], "b": ["x", "y"]}))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_data_keyword_source_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"b": ["x"], "a": [1]}))  # E: revealed type: DataFrame[b: String, a: Int64]
"#,
);

testcase!(
    test_data_keyword_with_schema_overrides,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"a": [1, 2], "b": [3, 4]}, schema_overrides={"a": pl.Int32}))  # E: revealed type: DataFrame[a: Int32, b: Int64]
"#,
);

testcase!(
    test_data_keyword_with_strict_false,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"a": [1, 2.0]}, strict=False))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_schema_overrides_before_data_keyword,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(schema_overrides={"a": pl.Int32}, data={"a": [1, 2]}))  # E: revealed type: DataFrame[a: Int32]
"#,
);

testcase!(
    test_data_keyword_strict_mismatch_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"a": [1, "s"]}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_fallback_data_keyword_empty_dict,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_data_keyword_list,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data=[1, 2]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_positional_and_data_keyword,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, data={"b": [2]}))  # E: revealed type: DataFrame # E: Multiple values for argument `data`
"#,
);

testcase!(
    test_data_keyword_and_schema_keyword,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"a": [1]}, schema={"a": pl.Int8}))  # E: revealed type: DataFrame[a: Int8]
"#,
);

testcase!(
    test_fallback_multiple_positional_args,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, None))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_schema_overrides_sets_column_dtype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": ["x"]}, schema_overrides={"a": pl.Int8}))  # E: revealed type: DataFrame[a: Int8, b: String]
"#,
);

testcase!(
    test_schema_overrides_suppresses_mismatch,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# The explicit dtype is authoritative, so an otherwise-incompatible mix coerces and does not error.
reveal_type(pl.DataFrame({"a": [1, 2.0]}, schema_overrides={"a": pl.Float64}))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_schema_overrides_ignores_non_polars_dtype_name,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
class Other:
    Int8 = int
# `Other.Int8` reuses a dtype name but is not a Polars dtype, so the override is not honored.
reveal_type(pl.DataFrame({"a": [1]}, schema_overrides={"a": Other.Int8}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_schema_keyword_with_matching_data,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, schema={"a": pl.Int8}))  # E: revealed type: DataFrame[a: Int8]
"#,
);

testcase!(
    test_schema_keyword_only_no_data,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(schema={"a": pl.Int64, "b": pl.String}))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_schema_dtype_coerces_data,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2], "b": [3, 4]}, schema={"a": pl.Int64, "b": pl.Float64}))  # E: revealed type: DataFrame[a: Int64, b: Float64]
"#,
);

testcase!(
    test_schema_none_value_defers_to_data,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2, 3]}, schema={"a": None}))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_schema_none_value_no_data_is_null,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(schema={"a": None, "b": pl.Int64}))  # E: revealed type: DataFrame[a: Null, b: Int64]
"#,
);

testcase!(
    test_schema_overrides_wins_over_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": [2]}, schema={"a": pl.Int64, "b": pl.Int64}, schema_overrides={"b": pl.Float64}))  # E: revealed type: DataFrame[a: Int64, b: Float64]
"#,
);

testcase!(
    test_schema_as_second_positional,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, {"a": pl.Float64}))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_schema_dtype_suppresses_mismatch,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# The authoritative schema dtype casts the mixed list, so no strict mismatch is reported.
reveal_type(pl.DataFrame({"a": [1, 2.0]}, schema={"a": pl.Float64}))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_schema_none_value_still_reports_mismatch,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, "s"]}, schema={"a": None}))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_schema_output_follows_schema_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"b": [1], "a": [2]}, schema={"a": pl.Int64, "b": pl.Int64}))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_schema_with_data_none_keyword,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data=None, schema={"a": pl.Int64, "b": pl.Float64}))  # E: revealed type: DataFrame[a: Int64, b: Float64]
"#,
);

testcase!(
    test_schema_with_empty_data_dict,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({}, schema={"a": pl.Int64}))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_schema_positional_with_none_data,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(None, {"a": pl.Int64, "b": pl.String}))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_schema_none_defers_to_data_inference,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, schema=None))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_row_transform_starred_arg_no_spurious_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
extra = [1]
# A `*` spread argument must not be treated as a type-form.
reveal_type(df.filter(*extra))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_sort_starred_arg_no_spurious_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
frame = pl.DataFrame({"a": [2, 1], "b": [1, 2]})
columns: list[str] = ["a", "b"]
# A `*` spread of column names into `sort` must not be treated as a type-form.
reveal_type(frame.sort(*columns))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_schema_rename_mismatch_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"x": [1, 2]}, schema={"a": pl.Int64}))  # E: revealed type: DataFrame # E: do not match the declared schema (missing `a`, unexpected `x`)
"#,
);

testcase!(
    test_schema_data_superset_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": [2]}, schema={"a": pl.Int64}))  # E: revealed type: DataFrame # E: do not match the declared schema (unexpected `b`)
"#,
);

testcase!(
    test_schema_data_subset_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, schema={"a": pl.Int64, "b": pl.String}))  # E: revealed type: DataFrame # E: do not match the declared schema (missing `b`)
"#,
);

testcase!(
    test_schema_class_inline_matching_data,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": ["x"]}, schema=pl.Schema({"a": pl.Int64, "b": pl.String})))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_schema_class_inline_no_data,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(schema=pl.Schema({"a": pl.Int64, "b": pl.String})))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_schema_class_imported_alias,
    env_with_polars_stubs(),
    r#"
import polars as pl
from polars import Schema as RenamedSchema
from typing import reveal_type
reveal_type(pl.DataFrame(schema=RenamedSchema({"a": pl.Int64})))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_unrelated_schema_attribute_falls_back,
    env_with_polars_stubs(),
    r#"
import mymod
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(schema=mymod.Schema({"a": pl.Int64})))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_schema_class_inline_mismatch_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"x": [1]}, schema=pl.Schema({"a": pl.Int64})))  # E: revealed type: DataFrame # E: do not match the declared schema (missing `a`, unexpected `x`)
"#,
);

testcase!(
    test_schema_class_matches_dict_literal,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, schema=pl.Schema({"a": pl.Int8})))  # E: revealed type: DataFrame[a: Int8]
reveal_type(pl.DataFrame({"a": [1]}, schema={"a": pl.Int8}))  # E: revealed type: DataFrame[a: Int8]
"#,
);

testcase!(
    test_schema_class_output_follows_schema_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"b": [1], "a": [2]}, schema=pl.Schema({"a": pl.Int64, "b": pl.Int64})))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_schema_class_none_value_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# pl.Schema forbids a None value and always raises at runtime, unlike a bare dict which defers to
# data inference, so we fall back rather than infer a concrete frame from broken code.
reveal_type(pl.DataFrame(schema=pl.Schema({"a": None})))  # E: revealed type: DataFrame # !E: DataFrame[
"#,
);

testcase!(
    test_schema_class_none_value_with_data_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, schema=pl.Schema({"a": None})))  # E: revealed type: DataFrame # !E: DataFrame[
"#,
);

testcase!(
    test_schema_class_mixed_none_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": [2]}, schema=pl.Schema({"a": pl.Int64, "b": None})))  # E: revealed type: DataFrame # !E: DataFrame[
"#,
);

testcase!(
    test_schema_dtype_coercion_not_validated,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Whether the data coerces to the pinned dtype is value-dependent, so no error is emitted here.
reveal_type(pl.DataFrame({"a": [1.5]}, schema={"a": pl.Int64}))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_schema_bound_name_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
MySchema = pl.Schema({"a": pl.Int64})
reveal_type(pl.DataFrame({"a": [1]}, schema=MySchema))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_schema_class_cross_file_falls_back,
    env_cross_file(),
    r#"
import polars as pl
from defs import my_schema
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, schema=my_schema))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_schema_list_form,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, schema=["a"]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_schema_non_dtype_value,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(schema={"a": 5}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_pandas_construct_infers_partial_schema,
    env_with_pandas_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
reveal_type(pd.DataFrame({"a": [1], "b": ["x"]}))  # E: revealed type: DataFrame[a: Int64, b: String, ...]
"#,
);

testcase!(
    test_pandas_columns_selects_and_orders,
    env_with_pandas_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
reveal_type(pd.DataFrame({"a": [1], "b": ["x"]}, columns=["b"]))  # E: revealed type: DataFrame[b: String, ...]
reveal_type(pd.DataFrame({"a": [1], "b": ["x"]}, columns=["b", "a"]))  # E: revealed type: DataFrame[b: String, a: Int64, ...]
"#,
);

testcase!(
    test_pandas_columns_missing_name_falls_back,
    env_with_pandas_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
reveal_type(pd.DataFrame({"a": [1]}, columns=["a", "c"]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_polars_columns_keyword_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, columns=["a"]))  # E: revealed type: DataFrame # E: Unexpected keyword argument `columns`
"#,
);

testcase!(
    test_strict_false_coerces_to_supertype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2.0]}, strict=False))  # E: revealed type: DataFrame[a: Float64]
reveal_type(pl.DataFrame({"a": [True, 1]}, strict=False))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_strict_false_incompatible_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Int64 and String have no modeled supertype; Polars coerces them only under strict=False.
reveal_type(pl.DataFrame({"a": [1, "s"]}, strict=False))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_strict_true_still_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2.0]}, strict=True))  # E: revealed type: DataFrame[a: Unknown] # E: Polars builds column `a` with type `Int64`
"#,
);

testcase!(
    test_degrade_non_list_value_keeps_good_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": 2}))  # E: revealed type: DataFrame[a: Int64, b: Unknown]
"#,
);

testcase!(
    test_degrade_series_value_keeps_good_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2], "b": pl.Series()}))  # E: revealed type: DataFrame[a: Int64, b: Unknown]
"#,
);

testcase!(
    test_degrade_range_value_keeps_good_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2], "b": range(2)}))  # E: revealed type: DataFrame[a: Int64, b: Unknown]
"#,
);

testcase!(
    test_degrade_per_column_order_preserved,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": [1j], "c": ["x"]}))  # E: revealed type: DataFrame[a: Int64, b: Unknown, c: String]
"#,
);

testcase!(
    test_degrade_column_read_consistency,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": 2})
reveal_type(df["b"])  # E: revealed type: Series
df["z"]  # E: Column `z` is not in the DataFrame schema
df.select("z")  # E: Column `z` is not in the DataFrame schema
"#,
);

testcase!(
    test_spread_key_still_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A spread makes the column name set unknown, so per-column degradation is unsafe.
reveal_type(pl.DataFrame({"a": [1], **{"b": [2]}}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_duplicate_key,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "a": ["x"]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_subclass_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
class MyFrame(pl.DataFrame): ...
reveal_type(MyFrame({"a": [1]}))  # E: revealed type: MyFrame
"#,
);

testcase!(
    test_element_type_error_reported_once,
    env_with_polars_stubs(),
    r#"
import polars as pl
pl.DataFrame({"a": [undefined_name]})  # E: Could not find name `undefined_name`
"#,
);

testcase!(
    test_schema_dataframe_assignable_to_underlying,
    env_with_polars_stubs(),
    r#"
import polars as pl
df: pl.DataFrame = pl.DataFrame({"a": [1]})
def f(x: pl.DataFrame) -> None: ...
f(pl.DataFrame({"a": [1]}))
"#,
);

testcase!(
    test_schema_dataframe_attribute_access,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.columns)  # E: revealed type: list[str]
reveal_type(df.head())  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_schema_dataframe_subscript,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df["a"])  # E: revealed type: Series[Int64]
"#,
);

testcase!(
    test_typed_series_is_subscriptable,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
s = df["a"]
reveal_type(s[0])  # E: revealed type: Any
reveal_type(s[0:2])  # E: revealed type: Series
reveal_type(df["a"][0])  # E: revealed type: Any
"#,
);

testcase!(
    test_typed_series_bitor_resolves_operator,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": [2]})
# `|` on Series values must resolve `__or__`, not be read as a PEP 604 type union.
reveal_type(df["a"] | df["b"])  # E: revealed type: Series
"#,
);

testcase!(
    test_known_column_read_no_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df["a"])  # E: revealed type: Series[Int64]
reveal_type(df["b"])  # E: revealed type: Series[String]
reveal_type(df["c"])  # E: revealed type: Series[Float64]
"#,
);

testcase!(
    test_column_read_unknown_dtype_is_typed_series,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A scalar column has no resolvable dtype, so it reads as Series[Unknown] rather than falling back.
df = pl.DataFrame({"a": 1})
reveal_type(df["a"])  # E: revealed type: Series[Unknown]
"#,
);

testcase!(
    test_partial_schema_column_read_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# `insert_column` degrades the frame to Partial, so a known column can no longer prove its dtype.
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
reveal_type(df["a"])  # E: revealed type: Series
"#,
);

testcase!(
    test_list_key_stays_dataframe,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df[["a"]])  # E: revealed type: DataFrame[a: Int64]
reveal_type(df[["a", "b"]])  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_typed_column_read_across_import,
    env_cross_file(),
    r#"
from defs import df
from typing import reveal_type
reveal_type(df["a"])  # E: revealed type: Series[Int64]
reveal_type(df["b"])  # E: revealed type: Series[String]
"#,
);

testcase!(
    test_unknown_column_read_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df["b"])  # E: Column `b` is not in the DataFrame schema # E: revealed type: Series
"#,
);

testcase!(
    test_non_literal_key_no_unknown_column_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
k = "b"
reveal_type(df[k])  # E: revealed type: Series
def key() -> str: ...
reveal_type(df[key()])  # E: revealed type: Series
"#,
);

testcase!(
    test_no_schema_no_unknown_column_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({})
reveal_type(df["missing"])  # E: revealed type: Series
"#,
);

testcase!(
    test_data_keyword_unknown_column_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame(data={"a": [1]})
df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_construct_from_data_keyword_across_import,
    env_cross_file(),
    r#"
from defs import df_kw
df_kw["a"]
df_kw["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_construct_from_schema_keyword_across_import,
    env_cross_file(),
    r#"
from defs import df_schema
df_schema["a"]
df_schema["b"]
df_schema["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_construct_from_records_across_import,
    env_cross_file(),
    r#"
from defs import df_records
df_records["a"]
df_records["b"]
df_records["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_unknown_column_is_suppressible,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df["b"]  # pyrefly: ignore[unknown-column]
"#,
);

testcase!(
    test_unknown_column_across_import,
    env_cross_file(),
    r#"
from defs import df
df["a"]
df["b"]
df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_schema_dataframe_iteration,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
for col in df:
    reveal_type(col)  # E: revealed type: Series
"#,
);

testcase!(
    test_schema_dataframe_membership,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type("a" in df)  # E: revealed type: bool
"#,
);

testcase!(
    test_select_list_narrows_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df[["c", "a"]])  # E: revealed type: DataFrame[c: Float64, a: Int64]
"#,
);

testcase!(
    test_select_list_unknown_column_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df[["a", "missing"]])  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_select_list_non_literal_element_delegates,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
k = "a"
reveal_type(df[[k]])  # E: revealed type: DataFrame
reveal_type(df[[1]])  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_list_unknown_column_suppressible,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df[["a", "b"]]  # pyrefly: ignore[unknown-column]
"#,
);

testcase!(
    test_select_list_duplicate_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df[["a", "a"]])  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_empty_list_narrows_to_empty,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df[[]])  # E: revealed type: DataFrame[]
"#,
);

testcase!(
    test_select_method_narrows_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.select("c", "a"))  # E: revealed type: DataFrame[c: Float64, a: Int64]
"#,
);

testcase!(
    test_select_method_literal_list_and_tuple_arguments,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.select(["c", "a"]))  # E: revealed type: DataFrame[c: Float64, a: Int64]
reveal_type(df.select((pl.col("b"), pl.col("a").alias("x"))))  # E: revealed type: DataFrame[b: String, x: Int64]
"#,
);

testcase!(
    test_select_method_leaves_original_schema_unchanged,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
df.select("a")
reveal_type(df)  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_select_method_non_literal_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
k = "a"
reveal_type(df.select(k))  # E: revealed type: DataFrame
reveal_type(df.select("a", k))  # E: revealed type: DataFrame
reveal_type(df.select([k]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_method_unknown_column_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select("a", "missing"))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_select_method_unknown_column_suppressible,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.select("b")  # pyrefly: ignore[unknown-column]
"#,
);

testcase!(
    test_select_method_duplicate_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.select("a", "a"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_method_empty_narrows_to_empty,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select())  # E: revealed type: DataFrame[]
"#,
);

testcase!(
    test_select_on_non_dataframe_falls_back,
    env_with_polars_stubs(),
    r#"
from typing import reveal_type
# A `select` method on an unrelated type is untouched; only Polars DataFrames are narrowed.
class NotAFrame:
    def select(self, x: int) -> int: ...
reveal_type(NotAFrame().select(1))  # E: revealed type: int
"#,
);

testcase!(
    test_select_on_non_dataframe_receiver_error_reported_once,
    env_with_polars_stubs(),
    r#"
# The receiver is inferred once, so an error inside it is not reported twice.
class NotAFrame:
    def select(self, x: int) -> int: ...
def f(n: NotAFrame) -> None:
    (n.missing).select(1)  # E: Object of class `NotAFrame` has no attribute `missing`
"#,
);

testcase!(
    test_select_wildcard_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.select("*"))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_select_wildcard_with_other_arg_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.select("*", "a"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_regex_selector_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.select("^a.*$"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_drop_wildcard_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.drop("*"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_method_keyword_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(b="x"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_expr_col_alias_renames,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.col("a").alias("c")))  # E: revealed type: DataFrame[c: Int64]
"#,
);

testcase!(
    test_select_expr_col_bare_keeps_name,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.col("a")))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_select_expr_cast_keeps_name_changes_dtype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.col("a").cast(pl.Float64)))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_select_expr_cast_then_alias,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.col("a").cast(pl.Float64).alias("c2")))  # E: revealed type: DataFrame[c2: Float64]
"#,
);

testcase!(
    test_select_expr_outer_alias_wins,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.col("a").alias("m").alias("n")))  # E: revealed type: DataFrame[n: Int64]
"#,
);

testcase!(
    test_select_expr_binop_takes_left_root,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "c": [1.0]})
reveal_type(df.select(pl.col("a") + pl.col("c")))  # E: revealed type: DataFrame[a: Float64]
reveal_type(df.select(pl.col("c") + pl.col("a")))  # E: revealed type: DataFrame[c: Float64]
"#,
);

testcase!(
    test_select_expr_binop_scalar_literal_left_names_literal,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(1 + pl.col("a")))  # E: revealed type: DataFrame[literal: Int64]
"#,
);

testcase!(
    test_select_expr_left_alias_propagates,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "c": [1.0]})
reveal_type(df.select(pl.col("a").alias("z") + pl.col("c")))  # E: revealed type: DataFrame[z: Float64]
"#,
);

testcase!(
    test_select_expr_right_alias_ignored,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "c": [1.0]})
reveal_type(df.select(pl.col("a") + pl.col("c").alias("z")))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_select_expr_comparison_is_boolean_under_left_name,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.col("a") > 0))  # E: revealed type: DataFrame[a: Boolean]
"#,
);

testcase!(
    test_select_expr_comparison_scalar_left_reflects_to_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# Python reflects `0 < col` to `col > 0`, so Polars names the output after the column, not the scalar.
reveal_type(df.select(0 < pl.col("a")))  # E: revealed type: DataFrame[a: Boolean]
reveal_type(df.select(5 >= pl.col("a")))  # E: revealed type: DataFrame[a: Boolean]
df.select(0 < pl.col("a")).select("a")
"#,
);

testcase!(
    test_select_expr_comparison_variable_scalar_left_reflects,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# A scalar held in a variable reflects at runtime just like a literal, so reflection is decided by the
# operand's type, not its syntax: the output is named after the column, not left as an opaque frame.
x = 0
reveal_type(df.select(x < pl.col("a")))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_select_expr_comparison_variable_expr_left_no_reflection,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# The left operand is a Polars expression in a variable, so there is no reflection; its output name
# cannot be resolved from the variable, so the whole select falls back.
e = pl.col("a")
reveal_type(df.select(e > 0))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_expr_comparison_lit_left_stays_literal,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# `pl.lit(5)` is already an expression, so there is no reflection and the output stays `literal`.
reveal_type(df.select(pl.lit(5) < pl.col("a")))  # E: revealed type: DataFrame[literal: Boolean]
"#,
);

testcase!(
    test_select_expr_lit_series_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# `pl.lit(series)` takes the series name, which is not statically knowable, so fall back to the opaque frame.
res = df.select(pl.lit(pl.Series("foo", [1])))
reveal_type(res)  # E: revealed type: DataFrame
res["foo"]
"#,
);

testcase!(
    test_select_expr_lit_names_literal,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.lit(5)))  # E: revealed type: DataFrame[literal: Int32]
reveal_type(df.select(pl.lit(5).alias("z")))  # E: revealed type: DataFrame[z: Int32]
"#,
);

testcase!(
    test_select_expr_bare_scalar_names_literal,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(5))  # E: revealed type: DataFrame[literal: Int32]
"#,
);

testcase!(
    test_select_expr_mixed_string_and_expr,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.select("a", pl.col("b").alias("bb")))  # E: revealed type: DataFrame[a: Int64, bb: String]
"#,
);

testcase!(
    test_select_expr_unknown_column_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.col("missing")))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[missing: Unknown]
"#,
);

testcase!(
    test_select_expr_duplicate_output_name_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.select(pl.col("a"), pl.col("b").alias("a")))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_expr_multi_name_col_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.select(pl.col("a", "b")))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_expr_wildcard_col_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.select(pl.col("*")))  # E: revealed type: DataFrame
reveal_type(df.select(pl.col("^a$")))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_expr_non_literal_alias_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
k = "c"
reveal_type(df.select(pl.col("a").alias(k)))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_expr_unmodeled_method_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.select(pl.col("a").sum()))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_drop_method_removes_column_preserves_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.drop("b"))  # E: revealed type: DataFrame[a: Int64, c: Float64]
"#,
);

testcase!(
    test_drop_method_multi_column_removes_both,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.drop("a", "c"))  # E: revealed type: DataFrame[b: String]
"#,
);

testcase!(
    test_drop_method_non_literal_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
k = "a"
reveal_type(df.drop(k))  # E: revealed type: DataFrame
reveal_type(df.drop("a", k))  # E: revealed type: DataFrame
reveal_type(df.drop((k,)))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_drop_method_unknown_and_non_literal_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
k = "a"
reveal_type(df.drop("missing", k))  # E: revealed type: DataFrame
reveal_type(df.drop(k, "missing"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_drop_method_duplicate_dedups,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.drop("a", "a"))  # E: revealed type: DataFrame[b: String]
"#,
);

testcase!(
    test_drop_method_unknown_column_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.drop("missing"))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_drop_method_strict_false_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.drop("missing", strict=False))  # E: revealed type: DataFrame
reveal_type(df)  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_drop_method_empty_call_unchanged,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.drop())  # E: revealed type: DataFrame[a: Int64, b: String, c: Float64]
"#,
);

testcase!(
    test_drop_method_literal_list_and_tuple_arguments,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.drop(["a", "b"]))  # E: revealed type: DataFrame[c: Float64]
reveal_type(df.drop(("a", "c")))  # E: revealed type: DataFrame[b: String]
"#,
);

testcase!(
    test_drop_method_across_import,
    env_cross_file(),
    r#"
from defs import df
from typing import reveal_type
reveal_type(df.drop("a"))  # E: revealed type: DataFrame[b: String]
"#,
);

testcase!(
    test_rename_maps_keys_preserving_types_and_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.rename({"b": "z"}))  # E: revealed type: DataFrame[a: Int64, z: String, c: Float64]
"#,
);

testcase!(
    test_rename_swaps_two_columns_in_single_pass,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({"a": "b", "b": "a"}))  # E: revealed type: DataFrame[b: Int64, a: String]
"#,
);

testcase!(
    test_rename_empty_mapping_unchanged,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({}))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_rename_column_to_itself_is_a_noop,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({"a": "a"}))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_rename_leaves_original_schema_unchanged,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
df.rename({"a": "z"})
reveal_type(df)  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_rename_unknown_source_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.rename({"missing": "z"}))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_rename_two_sources_same_target_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({"a": "c", "b": "c"}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_rename_target_collides_with_unrenamed_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({"a": "b"}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_rename_duplicate_source_key_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({"a": "y", "a": "z"}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_rename_keyword_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.rename({"a": "z"}, strict=False))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_rename_non_string_literal_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.rename({1: "z"}))  # E: revealed type: DataFrame
reveal_type(df.rename({"a": 2}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_with_columns_bare_string_is_column_reference,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# A bare string keyword value names a column to copy, so `b` takes `a`'s dtype.
reveal_type(df.with_columns(b="a"))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_with_columns_overwrites_existing_in_place,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.with_columns(a=pl.col("b")))  # E: revealed type: DataFrame[a: String, b: String]
"#,
);

testcase!(
    test_with_columns_append_and_overwrite_pins_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.with_columns(a=pl.lit(2.0), c=pl.lit(3)))  # E: revealed type: DataFrame[a: Float64, b: String, c: Int32]
"#,
);

testcase!(
    test_with_columns_col_copy,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(b=pl.col("a")))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_with_columns_lit_scalar_kinds,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(i=pl.lit(5)))  # E: revealed type: DataFrame[a: Int64, i: Int32]
reveal_type(df.with_columns(f=pl.lit(5.0)))  # E: revealed type: DataFrame[a: Int64, f: Float64]
reveal_type(df.with_columns(s=pl.lit("x")))  # E: revealed type: DataFrame[a: Int64, s: String]
reveal_type(df.with_columns(bo=pl.lit(True)))  # E: revealed type: DataFrame[a: Int64, bo: Boolean]
reveal_type(df.with_columns(n=pl.lit(None)))  # E: revealed type: DataFrame[a: Int64, n: Null]
reveal_type(df.with_columns(by=pl.lit(b"x")))  # E: revealed type: DataFrame[a: Int64, by: Binary]
"#,
);

testcase!(
    test_with_columns_lit_int_magnitude,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# An int literal widens by magnitude, so within i32 is Int32, past i32 is Int64, and past i64 is Int128.
reveal_type(df.with_columns(b=pl.lit(1099511627776)))  # E: revealed type: DataFrame[a: Int64, b: Int64]
reveal_type(df.with_columns(c=pl.lit(1180591620717411303424)))  # E: revealed type: DataFrame[a: Int64, c: Int128]
"#,
);

testcase!(
    test_with_columns_lit_dtype_keyword,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(b=pl.lit(5, dtype=pl.Int64)))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_with_columns_arithmetic_supertype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "c": [1.0]})
reveal_type(df.with_columns(x=pl.col("a") + pl.col("c")))  # E: revealed type: DataFrame[a: Int64, c: Float64, x: Float64]
reveal_type(df.with_columns(x=pl.col("a") + 1))  # E: revealed type: DataFrame[a: Int64, c: Float64, x: Int64]
reveal_type(df.with_columns(x=pl.col("a") / pl.col("a")))  # E: revealed type: DataFrame[a: Int64, c: Float64, x: Float64]
"#,
);

testcase!(
    test_with_columns_float32_division_keeps_width,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame(schema={"a": pl.Float32, "b": pl.Int32})
# True division promotes to Float64 only when the numeric supertype is an integer, so a Float32 supertype stays Float32.
reveal_type(df.with_columns(x=pl.col("a") / pl.col("a")))  # E: revealed type: DataFrame[a: Float32, b: Int32, x: Float32]
reveal_type(df.with_columns(x=pl.col("a") / 2.0))  # E: revealed type: DataFrame[a: Float32, b: Int32, x: Float32]
reveal_type(df.with_columns(x=pl.col("a") / 2))  # E: revealed type: DataFrame[a: Float32, b: Int32, x: Float32]
# Int32 does not fit in Float32, so the supertype is Float64.
reveal_type(df.with_columns(x=pl.col("a") / pl.col("b")))  # E: revealed type: DataFrame[a: Float32, b: Int32, x: Float64]
reveal_type(df.with_columns(x=pl.col("b") / pl.col("b")))  # E: revealed type: DataFrame[a: Float32, b: Int32, x: Float64]
"#,
);

testcase!(
    test_with_columns_comparison_is_boolean,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(b=pl.col("a") > pl.col("a")))  # E: revealed type: DataFrame[a: Int64, b: Boolean]
reveal_type(df.with_columns(b=pl.col("a") == pl.col("a")))  # E: revealed type: DataFrame[a: Int64, b: Boolean]
reveal_type(df.with_columns(b=pl.col("a") > 1))  # E: revealed type: DataFrame[a: Int64, b: Boolean]
"#,
);

testcase!(
    test_with_columns_bitwise_and_invert_boolean,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"d": [True]})
reveal_type(df.with_columns(x=pl.col("d") & pl.col("d")))  # E: revealed type: DataFrame[d: Boolean, x: Boolean]
reveal_type(df.with_columns(x=~pl.col("d")))  # E: revealed type: DataFrame[d: Boolean, x: Boolean]
"#,
);

testcase!(
    test_with_columns_cast,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(b=pl.col("a").cast(pl.Float64)))  # E: revealed type: DataFrame[a: Int64, b: Float64]
"#,
);

testcase!(
    test_with_columns_alias_passes_value_through,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# The keyword name is the output name, so an inner `.alias` only forwards the dtype.
reveal_type(df.with_columns(b=(pl.col("a") + 1).alias("z")))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_with_columns_unknown_column_errors_and_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(b=pl.col("nope")))  # E: revealed type: DataFrame[a: Int64, b: Unknown] # E: Column `nope` is not in the DataFrame schema
reveal_type(df.with_columns(b="nope"))  # E: revealed type: DataFrame[a: Int64, b: Unknown] # E: Column `nope` is not in the DataFrame schema
"#,
);

testcase!(
    test_with_columns_parallel_evaluation,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# `z` reads the pre-call `a` (Int64), not the `Float64` a sibling assigns in the same call.
reveal_type(df.with_columns(a=pl.col("a").cast(pl.Float64), z=pl.col("a") + 1))  # E: revealed type: DataFrame[a: Float64, z: Int64]
# A sibling's new column is not visible, so `c=col("b")` cannot see the just-added `b`.
reveal_type(df.with_columns(b=pl.col("a").cast(pl.Float64), c=pl.col("b")))  # E: revealed type: DataFrame[a: Int64, b: Float64, c: Unknown] # E: Column `b` is not in the DataFrame schema
"#,
);

testcase!(
    test_with_columns_narrow_overflow_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame(schema={"a": pl.Int8})
# Int8 + 1000 widens to Int16 by a data-dependent rule, so the column degrades rather than guess.
reveal_type(df.with_columns(b=pl.col("a") + 1000))  # E: revealed type: DataFrame[a: Int8, b: Unknown]
reveal_type(df.with_columns(b=pl.col("a") + 1))  # E: revealed type: DataFrame[a: Int8, b: Int8]
"#,
);

testcase!(
    test_with_columns_unsigned_negation_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame(schema={"a": pl.UInt8})
# Negating an unsigned column raises at runtime, so the column degrades rather than guess.
reveal_type(df.with_columns(b=-pl.col("a")))  # E: revealed type: DataFrame[a: UInt8, b: Unknown]
"#,
);

testcase!(
    test_with_columns_selector_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.with_columns(c=pl.col("a", "b")))  # E: revealed type: DataFrame[a: Int64, b: String, c: Unknown]
"#,
);

testcase!(
    test_with_columns_bare_string_selector_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# A regex or wildcard bare-string value selects a data-dependent set of columns, so the added
# column falls back to Unknown rather than emit a false unknown-column error.
reveal_type(df.with_columns(z="^a$"))  # E: revealed type: DataFrame[a: Int64, z: Unknown]
reveal_type(df.with_columns(z="*"))  # E: revealed type: DataFrame[a: Int64, z: Unknown]
"#,
);

testcase!(
    test_with_columns_keyword_unresolved_value_is_unknown,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
s = pl.col("a")
reveal_type(df.with_columns(b=s))  # E: revealed type: DataFrame[a: Int64, b: Unknown]
"#,
);

testcase!(
    test_with_columns_keyword_value_type_error_is_reported,
    env_with_polars_stubs(),
    r#"
import polars as pl
def f(x: int) -> int:
    return x
df = pl.DataFrame({"a": [1]})
df.with_columns(b=f("s"))  # E: Argument `Literal['s']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_with_columns_positional_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(pl.Series()))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_with_columns_spread_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(**{"b": "x"}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_with_columns_keyword_and_spread_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(a="y", **{"c": "z"}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_filter_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.filter(df["a"]))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_sort_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.sort("a"))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_fill_null_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.fill_null(0))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_head_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.head())  # E: revealed type: DataFrame[a: Int64, b: String]
reveal_type(df.head(2))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_slice_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.slice(1, 2))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_unique_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.unique(subset="a"))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_drop_nulls_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.drop_nulls())  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_head_preserves_complete_schema_for_reads,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.head()["missing"])  # E: revealed type: Series # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_row_transform_preserves_complete_schema_for_reads,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.sort("a")["missing"])  # E: revealed type: Series # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_row_transform_reports_error_in_argument,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.filter(undefined_name)  # E: Could not find name `undefined_name`
"#,
);

testcase!(
    test_cast_single_dtype_casts_all_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": [1.0]})
reveal_type(df.cast(pl.Float64))  # E: revealed type: DataFrame[a: Float64, b: Float64]
"#,
);

testcase!(
    test_cast_mapping_casts_named_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.cast({"a": pl.String}))  # E: revealed type: DataFrame[a: String, b: String]
"#,
);

testcase!(
    test_cast_unknown_column_is_reported,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.cast({"z": pl.Int32}))  # E: revealed type: DataFrame[a: Int64] # E: Column `z` is not in the DataFrame schema
"#,
);

testcase!(
    test_cast_unrecognized_dtype_falls_back_without_column_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
# An unrecognized dtype makes the whole cast fall back, so the absent column must not be reported.
reveal_type(df.cast({"z": pl.Int32, "a": 5}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_schema_form_shown_in_error_messages,
    env_with_polars_stubs(),
    r#"
import polars as pl
def want_int(x: int) -> None: ...
df = pl.DataFrame({"a": [1], "b": ["x"]})
want_int(df)  # E: Argument `DataFrame[a: Int64, b: String]` is not assignable to parameter `x` with type `int` in function `want_int`
"#,
);

testcase!(
    test_records_basic_single_key,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1}, {"a": 2}]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_records_two_keys,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}]))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_records_fold_int_then_float,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Records fold the supertype and never error, unlike the dict path which errors on this mix.
reveal_type(pl.DataFrame([{"a": 1}, {"a": 2.0}]))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_records_fold_float_then_int,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 2.0}, {"a": 1}]))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_records_fold_bool_then_int,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": True}, {"a": 2}]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_records_fold_bool_then_float,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": True}, {"a": 1.5}]))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_records_none_then_int,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": None}, {"a": 2}]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_records_int_then_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1}, {"a": None}]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_records_all_none,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": None}, {"a": None}]))  # E: revealed type: DataFrame[a: Null]
"#,
);

testcase!(
    test_records_second_row_adds_key,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1}, {"a": 2, "b": 3}]))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_records_first_row_extra_key,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1, "b": 2}, {"a": 3}]))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_records_disjoint_keys,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1}, {"b": 2}]))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_records_missing_key_takes_present_dtype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A key present in only one row is null-filled elsewhere but takes its present value's dtype.
reveal_type(pl.DataFrame([{"a": 1}, {"a": 2, "b": 3.0}]))  # E: revealed type: DataFrame[a: Int64, b: Float64]
"#,
);

testcase!(
    test_records_first_appearance_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# The column order follows first appearance across rows, not the last row.
reveal_type(pl.DataFrame([{"b": 1, "a": 2}, {"a": 3, "b": 4}]))  # E: revealed type: DataFrame[b: Int64, a: Int64]
"#,
);

testcase!(
    test_records_no_supertype_int_str_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Polars widens to String at runtime, but we model no such supertype, so the column degrades and
# no error is emitted on the record path.
reveal_type(pl.DataFrame([{"a": 1}, {"a": "x"}]))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_records_no_supertype_str_bytes_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": "x"}, {"a": b"y"}]))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_records_no_supertype_int_bytes_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1}, {"a": b"x"}]))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_records_non_literal_degrades_only_its_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
def g() -> int: ...
reveal_type(pl.DataFrame([{"a": 1, "b": g()}, {"a": 2, "b": 3}]))  # E: revealed type: DataFrame[a: Int64, b: Unknown]
"#,
);

testcase!(
    test_records_datetime_value_resolves,
    env_with_polars_stubs(),
    r#"
import polars as pl
from datetime import date
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": date(2020, 1, 1)}]))  # E: revealed type: DataFrame[a: Date]
"#,
);

testcase!(
    test_records_i64_max_is_int64,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 9223372036854775807}]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_records_int_above_i64_degrades,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Records give a past-i64 integer the Int128 dtype at runtime, which our i64-bounded model does not
# carry, so the column degrades rather than claiming Int64.
reveal_type(pl.DataFrame([{"a": 9223372036854775808}]))  # E: revealed type: DataFrame[a: Unknown]
"#,
);

testcase!(
    test_records_schema_overrides_wins,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1}, {"a": 2.0}], schema_overrides={"a": pl.Int32}))  # E: revealed type: DataFrame[a: Int32]
"#,
);

testcase!(
    test_records_empty_list_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_records_empty_dicts_fall_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{}]))  # E: revealed type: DataFrame
reveal_type(pl.DataFrame([{}, {}]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_records_non_dict_element_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([(1, 2), (3, 4)]))  # E: revealed type: DataFrame
reveal_type(pl.DataFrame([[1, 2], [3, 4]]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_records_mixed_dict_and_non_dict_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1}, (2,)]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_records_duplicate_key_in_row_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1, "a": 2}]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_records_with_schema_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Records combined with an explicit `schema=` are not yet modeled, so we fall back rather than
# apply the dict-path exact-match rules that records do not obey.
reveal_type(pl.DataFrame([{"a": 1}], schema={"a": pl.Int64}))  # E: revealed type: DataFrame
reveal_type(pl.DataFrame([{"x": 1}], schema={"a": pl.Int64}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_records_read_known_and_unknown_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame([{"a": 1}, {"b": 2}])
df["a"]
df["b"]
df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_records_exactly_100_rows_modeled,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame([{"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_records_over_100_rows_reads_first_100,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# We read only the first 100 rows like Polars, so the row-101 key `b` is dropped and the column
# set matches the runtime schema.
reveal_type(pl.DataFrame([{"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"b": 2}]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_concat_vertical_relaxed_supertype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame({"a": [1]}, schema={"a": pl.Int64})
d2 = pl.DataFrame({"a": [1.0]}, schema={"a": pl.Float64})
reveal_type(pl.concat([d1, d2], how="vertical_relaxed"))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_concat_vertical_relaxed_int128_absorbs_wide_unsigned,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Int128 is our widest signed dtype, so Polars keeps a UInt64 or UInt128 partner as Int128
# rather than promoting the pair to Float64.
d1 = pl.DataFrame(schema={"a": pl.Int128})
d2 = pl.DataFrame(schema={"a": pl.UInt64})
reveal_type(pl.concat([d1, d2], how="vertical_relaxed"))  # E: revealed type: DataFrame[a: Int128]
"#,
);

testcase!(
    test_concat_vertical_relaxed_uint128_widens_to_int128,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Polars caps a UInt128 against any signed at Int128, unlike a UInt64 which promotes to Float64.
d1 = pl.DataFrame(schema={"a": pl.Int8})
d2 = pl.DataFrame(schema={"a": pl.UInt128})
reveal_type(pl.concat([d1, d2], how="vertical_relaxed"))  # E: revealed type: DataFrame[a: Int128]
"#,
);

testcase!(
    test_concat_vertical_relaxed_multi_column_fold,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int32, "b": pl.Int64})
d2 = pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int8})
reveal_type(pl.concat([d1, d2], how="vertical_relaxed"))  # E: revealed type: DataFrame[a: Int64, b: Int64]
"#,
);

testcase!(
    test_concat_vertical_relaxed_three_frame_fold,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
i = pl.DataFrame(schema={"a": pl.Int64})
f = pl.DataFrame(schema={"a": pl.Float64})
reveal_type(pl.concat([i, i, f], how="vertical_relaxed"))  # E: revealed type: DataFrame[a: Float64]
"#,
);

testcase!(
    test_concat_vertical_relaxed_unmodeled_supertype_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# The runtime supertype of int and string is String, but our `supertype()` models only the numeric
# tower and returns None here, so we fall back rather than risk a wrong column dtype.
i = pl.DataFrame(schema={"a": pl.Int64})
s = pl.DataFrame(schema={"a": pl.String})
reveal_type(pl.concat([i, s], how="vertical_relaxed"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_concat_vertical_relaxed_name_mismatch_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64})
d2 = pl.DataFrame(schema={"b": pl.Int64})
reveal_type(pl.concat([d1, d2], how="vertical_relaxed"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_concat_vertical_relaxed_order_mismatch_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64})
d2 = pl.DataFrame(schema={"b": pl.Int64, "a": pl.Int64})
reveal_type(pl.concat([d1, d2], how="vertical_relaxed"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_concat_vertical_identical_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
d2 = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
reveal_type(pl.concat([d1, d2], how="vertical"))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_concat_default_how_is_vertical,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64})
reveal_type(pl.concat([d1, d1]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_concat_vertical_dtype_mismatch_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# `vertical` requires identical schemas; a differing dtype could be a spurious inferred difference,
# so we fall back rather than emit an error that risks a false positive.
d1 = pl.DataFrame(schema={"a": pl.Int64})
d2 = pl.DataFrame(schema={"a": pl.Float64})
reveal_type(pl.concat([d1, d2], how="vertical"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_concat_single_frame,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64})
reveal_type(pl.concat([d1], how="vertical"))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_concat_tuple_items,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64})
reveal_type(pl.concat((d1, d1), how="vertical"))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_concat_non_literal_items_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64})
frames = [d1, d1]
reveal_type(pl.concat(frames, how="vertical"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_concat_how_literal_variable,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64})
how = "vertical"
reveal_type(pl.concat([d1, d1], how=how))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_concat_non_literal_how_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
def f(how: str) -> None:
    d1 = pl.DataFrame(schema={"a": pl.Int64})
    reveal_type(pl.concat([d1, d1], how=how))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_concat_unmodeled_how_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# `diagonal` unions the columns; it is a deliberate non-goal, so we fall back.
d1 = pl.DataFrame(schema={"a": pl.Int64})
reveal_type(pl.concat([d1, d1], how="diagonal"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_concat_element_without_schema_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"a": pl.Int64})
opaque = pl.DataFrame(schema={1: pl.Int64})
reveal_type(pl.concat([d1, opaque], how="vertical"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_concat_cross_file_schema,
    env_cross_file(),
    r#"
import polars as pl
from defs import df
from typing import reveal_type
reveal_type(pl.concat([df, df], how="vertical"))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

fn env_join() -> TestEnv {
    let mut env = env_with_polars_stubs();
    env.add(
        "frames",
        r#"
import polars as pl
left = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Float64, "b": pl.String})
right = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64, "c": pl.Boolean})
"#,
    );
    env
}

testcase!(
    test_join_inner_coalesces_left_primary,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
reveal_type(left.join(right, on="k", how="inner"))  # E: revealed type: DataFrame[k: Int64, a: Float64, b: String, a_right: Int64, c: Boolean]
"#,
);

testcase!(
    test_join_left_matches_inner_shape,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
reveal_type(left.join(right, on="k", how="left"))  # E: revealed type: DataFrame[k: Int64, a: Float64, b: String, a_right: Int64, c: Boolean]
"#,
);

testcase!(
    test_join_right_coalesces_right_primary,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
reveal_type(left.join(right, on="k", how="right"))  # E: revealed type: DataFrame[a: Float64, b: String, k: Int64, a_right: Int64, c: Boolean]
"#,
);

testcase!(
    test_join_full_keeps_both_keys_suffixed,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
reveal_type(left.join(right, on="k", how="full"))  # E: revealed type: DataFrame[k: Int64, a: Float64, b: String, k_right: Int64, a_right: Int64, c: Boolean]
"#,
);

testcase!(
    test_join_semi_keeps_left_only,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
reveal_type(left.join(right, on="k", how="semi"))  # E: revealed type: DataFrame[k: Int64, a: Float64, b: String]
"#,
);

testcase!(
    test_join_anti_keeps_left_only,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
reveal_type(left.join(right, on="k", how="anti"))  # E: revealed type: DataFrame[k: Int64, a: Float64, b: String]
"#,
);

testcase!(
    test_join_cross_no_keys_keeps_both_suffixed,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
reveal_type(left.join(right, how="cross"))  # E: revealed type: DataFrame[k: Int64, a: Float64, b: String, k_right: Int64, a_right: Int64, c: Boolean]
"#,
);

testcase!(
    test_join_default_how_is_inner,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64, "b": pl.Int64})
reveal_type(d1.join(d2, on="k"))  # E: revealed type: DataFrame[k: Int64, a: Int64, b: Int64]
"#,
);

testcase!(
    test_join_multi_key_inner,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k1": pl.Int64, "k2": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k1": pl.Int64, "k2": pl.Int64, "b": pl.Int64})
reveal_type(d1.join(d2, on=["k1", "k2"], how="inner"))  # E: revealed type: DataFrame[k1: Int64, k2: Int64, a: Int64, b: Int64]
"#,
);

testcase!(
    test_join_multi_key_full_suffixes_both_keys,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k1": pl.Int64, "k2": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k1": pl.Int64, "k2": pl.Int64, "b": pl.Int64})
reveal_type(d1.join(d2, on=["k1", "k2"], how="full"))  # E: revealed type: DataFrame[k1: Int64, k2: Int64, a: Int64, k1_right: Int64, k2_right: Int64, b: Int64]
"#,
);

testcase!(
    test_join_tuple_key,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64, "b": pl.Int64})
reveal_type(d1.join(d2, on=("k",), how="inner"))  # E: revealed type: DataFrame[k: Int64, a: Int64, b: Int64]
"#,
);

testcase!(
    test_join_no_overlap_no_suffix,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64, "b": pl.String})
reveal_type(d1.join(d2, on="k", how="inner"))  # E: revealed type: DataFrame[k: Int64, a: Int64, b: String]
"#,
);

testcase!(
    test_join_result_schema_reads_columns,
    env_join(),
    r#"
from frames import left, right
joined = left.join(right, on="k", how="inner")
joined["a_right"]
joined["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_join_cross_file_schema,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
reveal_type(left.join(right, on="k", how="left"))  # E: revealed type: DataFrame[k: Int64, a: Float64, b: String, a_right: Int64, c: Boolean]
"#,
);

testcase!(
    test_join_leaves_receiver_schema_unchanged,
    env_join(),
    r#"
from frames import left, right
from typing import reveal_type
left.join(right, on="k", how="inner")
reveal_type(left)  # E: revealed type: DataFrame[k: Int64, a: Float64, b: String]
"#,
);

testcase!(
    test_join_unknown_key_errors_and_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64})
reveal_type(d1.join(d2, on="missing", how="inner"))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_key_missing_from_right_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64})
d2 = pl.DataFrame(schema={"j": pl.Int64})
reveal_type(d1.join(d2, on="k", how="inner"))  # E: Column `k` is not in the DataFrame schema # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_coalesced_key_dtype_mismatch_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A coalesced key with differing dtypes is cast or rejected at runtime, so we fall back rather
# than pick one side's dtype.
d1 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Float64, "b": pl.Int64})
reveal_type(d1.join(d2, on="k", how="inner"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_full_dtype_mismatch_kept_separately,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A full join keeps both keys, so differing key dtypes never coalesce and the schema stands.
d1 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Float64, "b": pl.Int64})
reveal_type(d1.join(d2, on="k", how="full"))  # E: revealed type: DataFrame[k: Int64, a: Int64, k_right: Float64, b: Int64]
"#,
);

testcase!(
    test_join_suffix_collision_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# The right `a` would become `a_right`, which already exists on the left, a runtime DuplicateError,
# so we fall back rather than emit a schema with a duplicate column.
d1 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64, "a_right": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
reveal_type(d1.join(d2, on="k", how="inner"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_cross_with_keys_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A cross join with join keys raises at runtime, so we fall back and let call-checking report it.
d1 = pl.DataFrame(schema={"k": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64})
reveal_type(d1.join(d2, on="k", how="cross"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_non_cross_without_keys_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64})
reveal_type(d1.join(d2, how="inner"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_non_literal_how_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64})
how = "inner"
reveal_type(d1.join(d2, on="k", how=how))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_unmodeled_how_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64})
reveal_type(d1.join(d2, on="k", how="outer"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_non_literal_key_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64})
k = "k"
reveal_type(d1.join(d2, on=k, how="inner"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_left_on_right_on_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# Differing key names via left_on/right_on are not yet modeled, so we fall back.
d1 = pl.DataFrame(schema={"kl": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"kr": pl.Int64, "a": pl.Int64})
reveal_type(d1.join(d2, left_on="kl", right_on="kr", how="inner"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_explicit_coalesce_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# An explicit coalesce= is not yet modeled, so we fall back.
d1 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64, "b": pl.Int64})
reveal_type(d1.join(d2, on="k", how="inner", coalesce=False))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_custom_suffix_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A custom suffix= is not yet modeled, so we fall back.
d1 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64, "a": pl.Int64})
reveal_type(d1.join(d2, on="k", how="inner", suffix="_r"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_other_without_schema_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64})
opaque = pl.DataFrame(schema={1: pl.Int64})
reveal_type(d1.join(opaque, on="k", how="inner"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_join_error_in_other_reported_once,
    env_with_polars_stubs(),
    r#"
import polars as pl
d1 = pl.DataFrame(schema={"k": pl.Int64})
d1.join(pl.DataFrame({"k": [undefined_name]}), on="k", how="inner")  # E: Could not find name `undefined_name`
"#,
);

testcase!(
    test_join_spread_keyword_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
d1 = pl.DataFrame(schema={"k": pl.Int64})
d2 = pl.DataFrame(schema={"k": pl.Int64})
reveal_type(d1.join(d2, on="k", **{"how": "inner"}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_vstack_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
other = pl.DataFrame({"a": [2], "b": ["y"]})
reveal_type(df.vstack(other))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_extend_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
other = pl.DataFrame({"a": [2], "b": ["y"]})
reveal_type(df.extend(other))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_vstack_opaque_other_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# vstack requires an identical schema at runtime, so the receiver schema is returned without
# inspecting `other`, even when `other` carries no schema of its own.
df = pl.DataFrame({"a": [1], "b": ["x"]})
opaque = pl.DataFrame(schema={1: pl.Int64})
reveal_type(df.vstack(opaque))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_vstack_reports_error_in_other,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.vstack(undefined_name)  # E: Could not find name `undefined_name`
"#,
);

testcase!(
    test_vstack_cross_file_schema,
    env_cross_file(),
    r#"
from defs import df
from typing import reveal_type
reveal_type(df.vstack(df))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_hstack_appends_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
other = pl.DataFrame({"c": [1.0], "d": [True]})
reveal_type(df.hstack(other))  # E: revealed type: DataFrame[a: Int64, b: String, c: Float64, d: Boolean]
"#,
);

testcase!(
    test_hstack_three_frame_chain,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
a = pl.DataFrame({"a": [1]})
b = pl.DataFrame({"b": [1.0]})
c = pl.DataFrame({"c": [True]})
reveal_type(a.hstack(b).hstack(c))  # E: revealed type: DataFrame[a: Int64, b: Float64, c: Boolean]
"#,
);

testcase!(
    test_hstack_overlapping_name_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# An overlapping column raises DuplicateError at runtime, so fall back rather than emit a duplicate.
df = pl.DataFrame({"a": [1]})
other = pl.DataFrame({"a": [2.0]})
reveal_type(df.hstack(other))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_hstack_series_list_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A list of Series carries only runtime column names, so fall back rather than guess.
df = pl.DataFrame({"a": [1]})
reveal_type(df.hstack([df["a"]]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_hstack_opaque_other_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
opaque = pl.DataFrame(schema={1: pl.Int64})
reveal_type(df.hstack(opaque))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_hstack_in_place_keyword_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
other = pl.DataFrame({"b": [2.0]})
reveal_type(df.hstack(other, in_place=True))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_hstack_opaque_receiver_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
opaque = pl.DataFrame(schema={1: pl.Int64})
other = pl.DataFrame({"a": [1]})
reveal_type(opaque.hstack(other))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_hstack_reports_error_in_other,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.hstack(pl.DataFrame({"b": [undefined_name]}))  # E: Could not find name `undefined_name`
"#,
);

testcase!(
    test_hstack_cross_file_schema,
    env_cross_file(),
    r#"
import polars as pl
from defs import df
from typing import reveal_type
other = pl.DataFrame({"c": [1.0]})
reveal_type(df.hstack(other))  # E: revealed type: DataFrame[a: Int64, b: String, c: Float64]
"#,
);

testcase!(
    test_vstack_non_frame_arg_reports_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
# A non-frame argument raises TypeError at runtime, so fall back and let the arg-type check fire.
df = pl.DataFrame({"a": [1]})
df.vstack(5)  # E: Argument `Literal[5]` is not assignable to parameter `other` with type `DataFrame`
"#,
);

testcase!(
    test_extend_non_frame_arg_reports_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
# A non-frame argument raises TypeError at runtime, so fall back and let the arg-type check fire.
df = pl.DataFrame({"a": [1]})
df.extend("foo")  # E: Argument `Literal['foo']` is not assignable to parameter `other` with type `DataFrame`
"#,
);

testcase!(
    test_vstack_pandas_arg_reports_error,
    env_with_polars_and_pandas_stubs(),
    r#"
import polars as pl
import pandas as pd
# A pandas frame is not a Polars frame, so fall back and let the arg-type check fire instead of
# swallowing the runtime TypeError.
df = pl.DataFrame({"a": [1]})
other = pd.DataFrame({"a": [1]})
df.vstack(other)  # E: is not assignable to parameter `other` with type `polars.dataframe.frame.DataFrame`
"#,
);

testcase!(
    test_hstack_pandas_arg_falls_back,
    env_with_polars_and_pandas_stubs(),
    r#"
import polars as pl
import pandas as pd
from typing import reveal_type
# A pandas frame raises AttributeError at runtime, so hstack must not fabricate a merged schema.
df = pl.DataFrame({"a": [1]})
other = pd.DataFrame({"c": [1.0]})
reveal_type(df.hstack(other))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_pandas_column_read_falls_back,
    env_with_polars_and_pandas_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
# A pandas frame is Partial and its column dtypes are unmodeled, so a column read stays opaque.
pdf = pd.DataFrame({"a": [1]})
reveal_type(pdf["a"])  # E: revealed type: Series
"#,
);

testcase!(
    test_insert_column_literal_keeps_known_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A literal index and `pl.Series` name are statically known, so the exact column is inserted in place
# and the schema stays Complete. Its dtype is Unknown until Series construction inference lands.
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
reveal_type(df)  # E: revealed type: DataFrame[a: Int64, b: Unknown]
df["b"]
df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_insert_column_non_literal_degrades_to_partial,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A non-literal index is not statically known, so the frame degrades to Partial and any read is allowed.
df = pl.DataFrame({"a": [1]})
i = 1
df.insert_column(i, pl.Series("b", [2]))
reveal_type(df)  # E: revealed type: DataFrame[a: Int64, ...]
df["anything"]
"#,
);

testcase!(
    test_insert_column_non_series_call_degrades_to_partial,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
def make_column(name: str, values: object) -> object: ...
df = pl.DataFrame({"a": [1]})
df.insert_column(1, make_column("b", [2]))
reveal_type(df)  # E: revealed type: DataFrame[a: Int64, ...]
df["anything"]
"#,
);

testcase!(
    test_hstack_in_place_degrades_receiver,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
other = pl.DataFrame({"b": [2.0]})
df.hstack(other, in_place=True)
reveal_type(df)  # E: revealed type: DataFrame[a: Int64, ...]
df["b"]
"#,
);

testcase!(
    test_insert_column_existing_column_still_reads,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
reveal_type(df["a"])  # E: revealed type: Series
"#,
);

testcase!(
    test_hstack_in_place_false_keeps_complete,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
other = pl.DataFrame({"b": [2.0]})
# A literal in_place=False returns a new frame and leaves the receiver's complete schema intact.
df.hstack(other, in_place=False)
df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_hstack_in_place_non_literal_degrades_receiver,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
other = pl.DataFrame({"b": [2.0]})
flag = True
# A non-literal in_place may be True at runtime, so degrade conservatively.
df.hstack(other, in_place=flag)
reveal_type(df)  # E: revealed type: DataFrame[a: Int64, ...]
df["b"]
"#,
);

testcase!(
    test_insert_column_return_value_keeps_known_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
df2 = df.insert_column(1, pl.Series("b", [2]))
reveal_type(df2)  # E: revealed type: DataFrame[a: Int64, b: Unknown]
df2["b"]
"#,
);

testcase!(
    test_replace_column_degrades_to_opaque,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# replace_column overwrites a column at an index we cannot map to a name, so the frame falls back to opaque.
df = pl.DataFrame({"a": [1]})
df.replace_column(0, pl.Series("z", [9.0]))
reveal_type(df)  # E: revealed type: DataFrame
df["z"]
"#,
);

testcase!(
    test_replace_column_removed_column_no_false_positive,
    env_with_polars_stubs(),
    r#"
import polars as pl
# The overwritten column may be gone at runtime, so reading it must not error on the opaque frame.
df = pl.DataFrame({"a": [1]})
df.replace_column(0, pl.Series("z", [9.0]))
df["a"]
"#,
);

testcase!(
    test_replace_column_return_value_is_opaque,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
df2 = df.replace_column(0, pl.Series("z", [9.0]))
reveal_type(df2)  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_replace_column_non_name_receiver_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}).replace_column(0, pl.Series("z", [9.0])))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_insert_column_non_name_receiver_keeps_known_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A non-name receiver has no flow binding to rebind, but the return value still carries the inserted column.
reveal_type(pl.DataFrame({"a": [1]}).insert_column(1, pl.Series("b", [2])))  # E: revealed type: DataFrame[a: Int64, b: Unknown]
"#,
);

testcase!(
    test_degraded_frame_select_no_unknown_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
df.select("b")
"#,
);

testcase!(
    test_degraded_frame_drop_no_unknown_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
df.drop("b")
"#,
);

testcase!(
    test_degraded_frame_rename_no_unknown_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
df.rename({"b": "c"})
"#,
);

testcase!(
    test_degraded_frame_cast_no_unknown_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
df.cast({"b": pl.Int64})
"#,
);

testcase!(
    test_degraded_frame_join_no_unknown_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
other = pl.DataFrame({"b": [2], "c": [3]})
df.join(other, on="b")
"#,
);

testcase!(
    test_vstack_in_place_does_not_degrade,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
other = pl.DataFrame({"a": [2]})
# vstack appends rows without changing the column set, so the complete schema stays valid.
df.vstack(other, in_place=True)
df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_extend_does_not_degrade,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
other = pl.DataFrame({"a": [2]})
df.extend(other)
df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_select_expr_reducer_sum_keeps_dtype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"x": [1, 2]})
reveal_type(df.select(pl.col("x").sum()))  # E: revealed type: DataFrame[x: Int64]
"#,
);

testcase!(
    test_with_columns_reducer_mean_promotes,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"x": [1, 2]})
reveal_type(df.with_columns(m=pl.col("x").mean()))  # E: revealed type: DataFrame[x: Int64, m: Float64]
"#,
);

testcase!(
    test_select_expr_len_is_uint32,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"x": [1, 2]})
reveal_type(df.select(pl.len()))  # E: revealed type: DataFrame[len: UInt32]
"#,
);

testcase!(
    test_group_by_agg_sum_keeps_dtype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x").sum()))  # E: revealed type: DataFrame[g: String, x: Int64]
"#,
);

testcase!(
    test_group_by_agg_mean_promotes_to_float,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x").mean()))  # E: revealed type: DataFrame[g: String, x: Float64]
"#,
);

testcase!(
    test_group_by_agg_count_is_uint32,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x").count()))  # E: revealed type: DataFrame[g: String, x: UInt32]
"#,
);

testcase!(
    test_group_by_agg_len_is_uint32,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.len()))  # E: revealed type: DataFrame[g: String, len: UInt32]
"#,
);

testcase!(
    test_group_by_agg_sum_narrow_int_widens_to_int64,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]}, schema={"g": pl.String, "x": pl.Int8})
reveal_type(df.group_by("g").agg(pl.col("x").sum()))  # E: revealed type: DataFrame[g: String, x: Int64]
"#,
);

testcase!(
    test_group_by_agg_sum_int32_stays_int32,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]}, schema={"g": pl.String, "x": pl.Int32})
reveal_type(df.group_by("g").agg(pl.col("x").sum()))  # E: revealed type: DataFrame[g: String, x: Int32]
"#,
);

testcase!(
    test_group_by_agg_multiple_keys_positional,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "h": [1], "x": [1.0]})
reveal_type(df.group_by("g", "h").agg(pl.col("x").sum()))  # E: revealed type: DataFrame[g: String, h: Int64, x: Float64]
"#,
);

testcase!(
    test_group_by_agg_keys_as_list,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "h": [1], "x": [1]})
reveal_type(df.group_by(["g", "h"]).agg(pl.col("x").sum()))  # E: revealed type: DataFrame[g: String, h: Int64, x: Int64]
"#,
);

testcase!(
    test_group_by_agg_expression_key_with_alias,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"x": [1], "y": [1.0]})
reveal_type(df.group_by((pl.col("x") > 1).alias("big")).agg(pl.col("y").sum()))  # E: revealed type: DataFrame[big: Boolean, y: Float64]
"#,
);

testcase!(
    test_group_by_agg_col_key_resolves_dtype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by(pl.col("g")).agg(pl.col("x").sum()))  # E: revealed type: DataFrame[g: String, x: Int64]
"#,
);

testcase!(
    test_group_by_agg_multiple_aggs,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1], "y": [1.0]})
reveal_type(df.group_by("g").agg(pl.col("x").sum(), pl.col("y").mean()))  # E: revealed type: DataFrame[g: String, x: Int64, y: Float64]
"#,
);

testcase!(
    test_group_by_agg_aggs_as_list,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1], "y": [1.0]})
reveal_type(df.group_by("g").agg([pl.col("x").sum(), pl.col("y").mean()]))  # E: revealed type: DataFrame[g: String, x: Int64, y: Float64]
"#,
);

testcase!(
    test_group_by_agg_alias_names_output,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x").sum().alias("total")))  # E: revealed type: DataFrame[g: String, total: Int64]
"#,
);

testcase!(
    test_group_by_agg_named_keyword_names_output,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(total=pl.col("x").sum()))  # E: revealed type: DataFrame[g: String, total: Int64]
"#,
);

testcase!(
    test_group_by_agg_bare_col_is_unknown_dtype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x")))  # E: revealed type: DataFrame[g: String, x: Unknown]
"#,
);

testcase!(
    test_group_by_agg_bare_string_is_unknown_dtype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg("x"))  # E: revealed type: DataFrame[g: String, x: Unknown]
"#,
);

testcase!(
    test_group_by_agg_empty_keeps_only_keys,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg())  # E: revealed type: DataFrame[g: String]
"#,
);

testcase!(
    test_group_by_agg_selector_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x", "y").sum()))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_group_by_agg_variable_receiver_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
gb = df.group_by("g")
reveal_type(gb.agg(pl.col("x").sum()))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_group_by_agg_unknown_key_reports,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("missing").agg(pl.col("x").sum()))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame
"#,
);

testcase!(
    test_group_by_agg_key_agg_collision_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x").sum().alias("g")))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_group_by_agg_agg_agg_collision_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x").sum(), pl.col("x").mean()))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_group_by_agg_collision_reports_argument_error_once,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("x").sum(1).alias("g")))  # E: Expected 0 positional arguments, got 1 # E: revealed type: DataFrame
"#,
);

testcase!(
    test_group_by_agg_collision_does_not_emit_schema_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"g": ["a"], "x": [1]})
reveal_type(df.group_by("g").agg(pl.col("missing").sum().alias("g")))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_group_by_agg_cross_file_schema,
    env_cross_file(),
    r#"
import defs
import polars as pl
from typing import reveal_type
reveal_type(defs.df.group_by("b").agg(pl.col("a").sum()))  # E: revealed type: DataFrame[b: String, a: Int64]
"#,
);

// Section F: `pl.Series(...)` construction, which reuses the DataFrame column fold for its element dtype.

testcase!(
    test_series_construct_int,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series("a", [1]))  # E: revealed type: Series[Int64]
"#,
);

testcase!(
    test_series_construct_scalar_dtypes,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series("a", [1, 2, 3]))  # E: revealed type: Series[Int64]
reveal_type(pl.Series("a", [1.0]))  # E: revealed type: Series[Float64]
reveal_type(pl.Series("a", ["x"]))  # E: revealed type: Series[String]
reveal_type(pl.Series("a", [True]))  # E: revealed type: Series[Boolean]
reveal_type(pl.Series("a", [b"x"]))  # E: revealed type: Series[Binary]
reveal_type(pl.Series("a", [None]))  # E: revealed type: Series[Null]
"#,
);

testcase!(
    test_series_construct_values_as_first_arg,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series([1, 2, 3]))  # E: revealed type: Series[Int64]
reveal_type(pl.Series((1, 2)))  # E: revealed type: Series[Int64]
"#,
);

testcase!(
    test_series_construct_keyword_values,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series(name="a", values=[1]))  # E: revealed type: Series[Int64]
reveal_type(pl.Series("a", values=[1]))  # E: revealed type: Series[Int64]
"#,
);

testcase!(
    test_series_construct_none_anchoring,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series("a", [1, None, 2]))  # E: revealed type: Series[Int64]
reveal_type(pl.Series("a", [None, 1]))  # E: revealed type: Series[Int64]
reveal_type(pl.Series("a", [1, True]))  # E: revealed type: Series[Int64]
"#,
);

testcase!(
    test_series_construct_dtype_override,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series("a", [1], dtype=pl.Int8))  # E: revealed type: Series[Int8]
reveal_type(pl.Series("a", [1, 2], dtype=pl.Int8))  # E: revealed type: Series[Int8]
reveal_type(pl.Series("a", [1], pl.Float32))  # E: revealed type: Series[Float32]
reveal_type(pl.Series("a", dtype=pl.Int8))  # E: revealed type: Series[Int8]
"#,
);

testcase!(
    test_series_construct_strict_false_supertype,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series("a", [1, 2.0], strict=False))  # E: revealed type: Series[Float64]
"#,
);

testcase!(
    test_series_construct_no_values_is_null,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series("a"))  # E: revealed type: Series[Null]
reveal_type(pl.Series())  # E: revealed type: Series[Null]
"#,
);

// Reusing the column fold reports an empty list as `Unknown` like the DataFrame path, though the runtime dtype is `Null`.
testcase!(
    bug = "empty-values Series is Series[Unknown], runtime is Null",
    test_series_construct_empty_values,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series("a", []))  # E: revealed type: Series[Unknown]
"#,
);

testcase!(
    test_series_construct_mismatch_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.Series("a", [1, 2.0]))  # E: revealed type: Series
reveal_type(pl.Series("a", [1, "x"]))  # E: revealed type: Series
"#,
);

testcase!(
    test_series_construct_unmodeled_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
xs = [1, 2, 3]
reveal_type(pl.Series("a", xs))  # E: revealed type: Series
reveal_type(pl.Series("a", range(3)))  # E: revealed type: Series
reveal_type(pl.Series("a", [10**19]))  # E: revealed type: Series
reveal_type(pl.Series("a", [[1, 2], [3]]))  # E: revealed type: Series
"#,
);

testcase!(
    test_series_construct_cross_file,
    env_cross_file(),
    r#"
import defs
from typing import reveal_type
reveal_type(defs.s)  # E: revealed type: Series[Int64]
"#,
);

// Section G: `df.get_column(name)` and `df.to_series(index)`, which read one column and return a typed Series.

testcase!(
    test_get_column_typed,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.get_column("a"))  # E: revealed type: Series[Int64]
reveal_type(df.get_column("b"))  # E: revealed type: Series[String]
reveal_type(df.get_column(name="c"))  # E: revealed type: Series[Float64]
"#,
);

testcase!(
    test_get_column_unknown_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.get_column("zzz"))  # E: Column `zzz` is not in the DataFrame schema # E: revealed type: Series
"#,
);

testcase!(
    test_get_column_scalar_is_unknown,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": 1})
reveal_type(df.get_column("a"))  # E: revealed type: Series[Unknown]
"#,
);

testcase!(
    test_get_column_default_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
# A `default=` makes the return `Series | Any` and suppresses the raise, so we do not model it.
df = pl.DataFrame({"a": [1]})
reveal_type(df.get_column("zzz", default=None))  # E: revealed type: Series
"#,
);

testcase!(
    test_get_column_non_literal_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
n: str = "a"
reveal_type(df.get_column(n))  # E: revealed type: Series
"#,
);

testcase!(
    test_get_column_partial_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
reveal_type(df.get_column("a"))  # E: revealed type: Series
"#,
);

testcase!(
    test_get_column_cross_file,
    env_cross_file(),
    r#"
from defs import df
from typing import reveal_type
reveal_type(df.get_column("a"))  # E: revealed type: Series[Int64]
reveal_type(df.get_column("b"))  # E: revealed type: Series[String]
"#,
);

testcase!(
    test_to_series_index,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.to_series())  # E: revealed type: Series[Int64]
reveal_type(df.to_series(0))  # E: revealed type: Series[Int64]
reveal_type(df.to_series(1))  # E: revealed type: Series[String]
reveal_type(df.to_series(index=2))  # E: revealed type: Series[Float64]
"#,
);

testcase!(
    test_to_series_negative_index,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.to_series(-1))  # E: revealed type: Series[Float64]
reveal_type(df.to_series(-3))  # E: revealed type: Series[Int64]
"#,
);

testcase!(
    test_to_series_out_of_range_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.to_series(5))  # E: Index 5 is out of bounds for a DataFrame with 3 columns # E: revealed type: Series
reveal_type(df.to_series(-5))  # E: Index -5 is out of bounds for a DataFrame with 3 columns # E: revealed type: Series
"#,
);

testcase!(
    test_to_series_non_literal_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
i: int = 0
reveal_type(df.to_series(i))  # E: revealed type: Series
"#,
);

testcase!(
    test_to_series_scalar_is_unknown,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": 1})
reveal_type(df.to_series())  # E: revealed type: Series[Unknown]
"#,
);

testcase!(
    test_to_series_partial_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
df.insert_column(1, pl.Series("b", [2]))
reveal_type(df.to_series())  # E: revealed type: Series
"#,
);

testcase!(
    test_fp_rename_then_read_new_name,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1], "b": ["x"]})
renamed = df.rename({"a": "z"})
renamed["z"]
renamed["b"]
renamed["a"]  # E: Column `a` is not in the DataFrame schema
"#,
);

testcase!(
    test_fp_with_columns_added_then_read,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1]})
wc = df.with_columns(c=pl.col("a") + 1)
wc["c"]
wc["a"]
wc["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_fp_drop_then_read_remaining,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1], "b": ["x"]})
dropped = df.drop("a")
dropped["b"]
dropped["a"]  # E: Column `a` is not in the DataFrame schema
"#,
);

testcase!(
    test_fp_select_then_read_kept,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1], "b": ["x"]})
narrowed = df.select("a")
narrowed["a"]
narrowed["b"]  # E: Column `b` is not in the DataFrame schema
"#,
);

testcase!(
    test_lazy_preserves_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.lazy())  # E: revealed type: LazyFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_lazy_collect_round_trips_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.lazy().collect())  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_lazy_transform_narrows_schema,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.lazy().select("a"))  # E: revealed type: LazyFrame[a: Int64]
"#,
);

testcase!(
    test_lazy_unknown_column_read_errors_after_collect,
    env_with_polars_stubs(),
    r#"
import polars as pl
df = pl.DataFrame({"a": [1], "b": ["x"]})
collected = df.lazy().select("a").collect()
collected["a"]
collected["b"]  # E: Column `b` is not in the DataFrame schema
"#,
);

testcase!(
    test_lazy_on_opaque_frame_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
def f(df: pl.DataFrame) -> None:
    reveal_type(df.lazy())  # E: revealed type: LazyFrame
"#,
);

// `collect` keeps the schema while the stub enforces the `engine` literal, which polars also
// rejects at runtime for an unknown value.
testcase!(
    test_lazy_collect_engine_literal_enforced,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.lazy().collect(engine="streaming"))  # E: revealed type: DataFrame[a: Int64, b: String]
reveal_type(df.lazy().collect(engine="bad"))  # E: revealed type: DataFrame[a: Int64, b: String] # E: not assignable to parameter `engine`
"#,
);

testcase!(
    test_schema_class_construction,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
class MySchema:
    price: pl.Float64
    asset: pl.String
reveal_type(pl.DataFrame(schema=MySchema))  # E: revealed type: DataFrame[price: Float64, asset: String]
"#,
);

testcase!(
    test_schema_class_ignores_non_dtype_fields,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
class MySchema:
    price: int
reveal_type(pl.DataFrame(schema=MySchema))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_dataframe_schema_annotation,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
class MySchema:
    price: pl.Float64
    asset: pl.String
def f(df: pl.DataFrame[MySchema]) -> None:
    reveal_type(df)  # E: revealed type: DataFrame[price: Float64, asset: String]
"#,
);

testcase!(
    test_dataframe_schema_annotation_reads_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
class MySchema:
    price: pl.Float64
def f(df: pl.DataFrame[MySchema]) -> None:
    reveal_type(df["price"])  # E: revealed type: Series
    df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);
