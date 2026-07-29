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
        "polars.dataframe.frame",
        "polars/dataframe/frame.pyi",
        r#"
from typing import Iterator, overload
class Series: ...
class DataFrame:
    columns: list[str]
    def __init__(self, data: object = None, schema: object = None, schema_overrides: object = None, strict: bool = True) -> None: ...
    @overload
    def __getitem__(self, key: str) -> Series: ...
    @overload
    def __getitem__(self, key: list[str] | list[int]) -> "DataFrame": ...
    def __iter__(self) -> Iterator[Series]: ...
    def __contains__(self, key: str) -> bool: ...
    def head(self, n: int = 5) -> "DataFrame": ...
    def select(self, *exprs: object, **named_exprs: object) -> "DataFrame": ...
    def drop(self, *columns: object, strict: bool = True) -> "DataFrame": ...
    def rename(self, mapping: object, *, strict: bool = True) -> "DataFrame": ...
    def with_columns(self, *exprs: object, **named_exprs: object) -> "DataFrame": ...
    def filter(self, *predicates: object, **constraints: object) -> "DataFrame": ...
    def sort(self, by: object, *more: object, descending: bool = False) -> "DataFrame": ...
    def fill_null(self, value: object = None) -> "DataFrame": ...
    def cast(self, dtypes: object, *, strict: bool = True) -> "DataFrame": ...
"#,
    );
    env.add(
        "polars",
        r#"
from polars.dataframe.frame import DataFrame as DataFrame, Series as Series
class Int8: ...
class Int32: ...
class Float64: ...
class String: ...
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
    test_fallback_keyword_argument,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"a": [1]}))  # E: revealed type: DataFrame
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
    test_schema_keyword_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, schema={"a": pl.Int8}))  # E: revealed type: DataFrame
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
    test_pandas_columns_keyword_falls_back,
    env_with_pandas_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
reveal_type(pd.DataFrame({"a": [1]}, columns=["a"]))  # E: revealed type: DataFrame
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
reveal_type(df.head())  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_schema_dataframe_subscript,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df["a"])  # E: revealed type: Series
"#,
);

testcase!(
    test_known_column_read_no_error,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df["a"])  # E: revealed type: Series
reveal_type(df["b"])  # E: revealed type: Series
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
df = pl.DataFrame(data={"a": [1]})
reveal_type(df["missing"])  # E: revealed type: Series
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
    test_drop_method_list_argument_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
# An iterable argument is not a bare string literal, so we under-report rather than guess.
reveal_type(df.drop(["a", "b"]))  # E: revealed type: DataFrame
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
    test_with_columns_appends_new_keyword_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.with_columns(b="x"))  # E: revealed type: DataFrame[a: Int64, b: Unknown]
"#,
);

testcase!(
    test_with_columns_overwrites_existing_in_place,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.with_columns(a="y"))  # E: revealed type: DataFrame[a: Unknown, b: String]
"#,
);

testcase!(
    test_with_columns_append_and_overwrite_pins_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.with_columns(a="y", c="z"))  # E: revealed type: DataFrame[a: Unknown, b: String, c: Unknown]
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
