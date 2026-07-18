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
    def __init__(self, data: object = None, schema: object = None) -> None: ...
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
"#,
    );
    env.add(
        "polars",
        "from polars.dataframe.frame import DataFrame as DataFrame, Series as Series",
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

testcase!(
    test_construct_int_and_str_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))  # E: revealed type: DataFrame[a: int, b: str]
"#,
);

testcase!(
    test_columns_in_source_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"b": ["x"], "a": [1]}))  # E: revealed type: DataFrame[b: str, a: int]
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
    test_fallback_value_not_list,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": 1}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_non_literal_element,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
x: int = 1
reveal_type(pl.DataFrame({"a": [x]}))  # E: revealed type: DataFrame
def g() -> int: ...
reveal_type(pl.DataFrame({"b": [g()]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_construct_incompatible_mix_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, "s"]}))  # E: revealed type: DataFrame # E: Polars builds column `a` with type `int`
"#,
);

testcase!(
    test_construct_int_then_float_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2.0]}))  # E: revealed type: DataFrame # E: Polars builds column `a` with type `int`
"#,
);

testcase!(
    test_construct_float_then_int_widens_to_float,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [2.0, 1]}))  # E: revealed type: DataFrame[a: float]
"#,
);

testcase!(
    test_construct_float_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1.0, 2.0]}))  # E: revealed type: DataFrame[a: float]
"#,
);

testcase!(
    test_construct_bool_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [True, False]}))  # E: revealed type: DataFrame[a: bool]
"#,
);

testcase!(
    test_construct_bytes_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [b"x", b"y"]}))  # E: revealed type: DataFrame[a: bytes]
"#,
);

testcase!(
    test_fallback_complex_not_modeled,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1j]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_construct_int_then_bool_is_int,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, True]}))  # E: revealed type: DataFrame[a: int]
"#,
);

testcase!(
    test_construct_bool_then_int_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [True, 1]}))  # E: revealed type: DataFrame # E: Polars builds column `a` with type `bool`
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
reveal_type(pl.DataFrame({"a": [1], "b": [], "c": [2.0, 1]}))  # E: revealed type: DataFrame[a: int, b: Unknown, c: float]
"#,
);

testcase!(
    test_fallback_mixed_literal_and_non_literal,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
x: int = 1
reveal_type(pl.DataFrame({"a": [1, x]}))  # E: revealed type: DataFrame
def g() -> int: ...
reveal_type(pl.DataFrame({"b": [2, g()]}))  # E: revealed type: DataFrame
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
    test_fallback_one_bad_column_pins_whole_dict,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": 2}))  # E: revealed type: DataFrame
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
df = pl.DataFrame({"a": 1})
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
reveal_type(df[["c", "a"]])  # E: revealed type: DataFrame[c: float, a: int]
"#,
);

testcase!(
    test_select_list_unknown_column_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df[["a", "missing"]])  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[a: int]
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
reveal_type(df.select("c", "a"))  # E: revealed type: DataFrame[c: float, a: int]
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
reveal_type(df)  # E: revealed type: DataFrame[a: int, b: str]
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
reveal_type(df.select("a", "missing"))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[a: int]
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
reveal_type(df.drop("b"))  # E: revealed type: DataFrame[a: int, c: float]
"#,
);

testcase!(
    test_drop_method_multi_column_removes_both,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.drop("a", "c"))  # E: revealed type: DataFrame[b: str]
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
reveal_type(df.drop("a", "a"))  # E: revealed type: DataFrame[b: str]
"#,
);

testcase!(
    test_drop_method_unknown_column_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.drop("missing"))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[a: int]
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
reveal_type(df)  # E: revealed type: DataFrame[a: int]
"#,
);

testcase!(
    test_rename_maps_keys_preserving_types_and_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
reveal_type(df.rename({"b": "z"}))  # E: revealed type: DataFrame[a: int, z: str, c: float]
"#,
);

testcase!(
    test_rename_swaps_two_columns_in_single_pass,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({"a": "b", "b": "a"}))  # E: revealed type: DataFrame[b: int, a: str]
"#,
);

testcase!(
    test_rename_empty_mapping_unchanged,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({}))  # E: revealed type: DataFrame[a: int, b: str]
"#,
);

testcase!(
    test_rename_column_to_itself_is_a_noop,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.rename({"a": "a"}))  # E: revealed type: DataFrame[a: int, b: str]
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
reveal_type(df)  # E: revealed type: DataFrame[a: int, b: str]
"#,
);

testcase!(
    test_rename_unknown_source_errors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.rename({"missing": "z"}))  # E: Column `missing` is not in the DataFrame schema # E: revealed type: DataFrame[a: int]
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
reveal_type(df.with_columns(b="x"))  # E: revealed type: DataFrame[a: int, b: Unknown]
"#,
);

testcase!(
    test_with_columns_overwrites_existing_in_place,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.with_columns(a="y"))  # E: revealed type: DataFrame[a: Unknown, b: str]
"#,
);

testcase!(
    test_with_columns_append_and_overwrite_pins_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1], "b": ["x"]})
reveal_type(df.with_columns(a="y", c="z"))  # E: revealed type: DataFrame[a: Unknown, b: str, c: Unknown]
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
