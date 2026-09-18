/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::narwhals_testcase;
use crate::test::narwhals::util::env_with_narwhals_and_polars_stubs;
use crate::testcase;

testcase!(
    test_from_native_carries_backend_schema,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
reveal_type(nw.from_native(pl.DataFrame({"a": [1], "b": ["x"]})))  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_from_native_on_untracked_frame_falls_back,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
def f(df: pl.DataFrame) -> None:
    reveal_type(nw.from_native(df))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_select_narrows_columns,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"a": [1], "b": ["x"]}))
reveal_type(df.select("a"))  # E: revealed type: DataFrame[a: Int64]
df.select("missing")  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_drop_and_rename,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"a": [1], "b": ["x"]}))
reveal_type(df.drop("b"))  # E: revealed type: DataFrame[a: Int64]
reveal_type(df.rename({"a": "c"}))  # E: revealed type: DataFrame[c: Int64, b: String]
"#,
);

testcase!(
    test_with_columns_adds_column,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"a": [1]}))
reveal_type(df.with_columns(b=nw.col("a") * nw.lit(2)))  # E: revealed type: DataFrame[a: Int64, b: Int64]
df.with_columns(c=nw.col("missing"))  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_row_transforms_preserve_schema,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"a": [1], "b": ["x"]}))
reveal_type(df.filter(nw.col("a") > 0))  # E: revealed type: DataFrame[a: Int64, b: String]
reveal_type(df.sort("a"))  # E: revealed type: DataFrame[a: Int64, b: String]
reveal_type(df.head(3))  # E: revealed type: DataFrame[a: Int64, b: String]
reveal_type(df.drop_nulls())  # E: revealed type: DataFrame[a: Int64, b: String]
reveal_type(df.unique())  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_lazy_collect_round_trips_schema,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"a": [1], "b": ["x"]}))
reveal_type(df.lazy())  # E: revealed type: LazyFrame[a: Int64, b: String]
reveal_type(df.lazy().select("a"))  # E: revealed type: LazyFrame[a: Int64]
reveal_type(df.lazy().collect())  # E: revealed type: DataFrame[a: Int64, b: String]
"#,
);

testcase!(
    test_column_access_and_unknown_column,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"a": [1], "b": ["x"]}))
reveal_type(df["a"])  # E: revealed type: Series[Int64]
reveal_type(df.get_column("b"))  # E: revealed type: Series[String]
df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_join_merges_schemas,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
left = nw.from_native(pl.DataFrame({"key": [1], "a": ["x"]}))
right = nw.from_native(pl.DataFrame({"key": [1], "b": [1.0]}))
reveal_type(left.join(right, on="key"))  # E: revealed type: DataFrame[key: Int64, a: String, b: Float64]
"#,
);

testcase!(
    test_group_by_agg,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"g": ["x"], "v": [1]}))
reveal_type(df.group_by("g").agg(nw.col("v").sum()))  # E: revealed type: DataFrame[g: String, v: Int64]
"#,
);

// `narwhals.len` is `narwhals.functions.len_` imported under its public name, since `len_`
// avoids shadowing the builtin; `narwhals.stable.v1.len` is instead its own function.
testcase!(
    test_len_resolves_from_top_level_and_stable_namespaces,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import narwhals.stable.v1 as nw_v1
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"g": ["x"]}))
reveal_type(df.group_by("g").agg(nw.len()))  # E: revealed type: DataFrame[g: String, len: UInt32]

df_v1 = nw_v1.from_native(pl.DataFrame({"g": ["x"]}))
reveal_type(df_v1.group_by("g").agg(nw_v1.len()))  # E: revealed type: DataFrame[g: String, len: UInt32]
"#,
);

testcase!(
    test_lazy_group_by_agg,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"g": ["x"], "v": [1]})).lazy()
reveal_type(df.group_by("g").agg(nw.col("v").sum()))  # E: revealed type: LazyFrame[g: String, v: Int64]
df.group_by("missing").agg(nw.col("v").sum())  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_duplicate_output_column_is_rejected,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
df = nw.from_native(pl.DataFrame({"a": [1], "b": ["x"]}))
df.select("a", "a")  # E: Operation produces duplicate column `a`
"#,
);

testcase!(
    test_concat_preserves_schema,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import reveal_type
a = nw.from_native(pl.DataFrame({"a": [1]}))
b = nw.from_native(pl.DataFrame({"a": [2]}))
reveal_type(nw.concat([a, b]))  # E: revealed type: DataFrame[a: Int64]
"#,
);

testcase!(
    test_stable_v1_namespace_tracks_schema,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals.stable.v1 as nw
import polars as pl
from typing import reveal_type
df = nw.from_native(pl.DataFrame({"a": [1], "b": ["x"]}))
reveal_type(df)  # E: revealed type: DataFrame[a: Int64, b: String]
reveal_type(df.select("a"))  # E: revealed type: DataFrame[a: Int64]
df.select("missing")  # E: Column `missing` is not in the DataFrame schema
"#,
);

narwhals_testcase!(
    test_dataframe_schema_annotation,
    r#"
import narwhals as nw
from typing import Annotated, reveal_type
class MySchema:
    price: nw.Float64
    asset: nw.String
def f(df: Annotated[nw.DataFrame, MySchema]) -> None:
    reveal_type(df)  # E: revealed type: DataFrame[price: Float64, asset: String]
    reveal_type(df["price"])  # E: revealed type: Series[Float64]
    df["missing"]  # E: Column `missing` is not in the DataFrame schema
"#,
);

testcase!(
    test_schema_contracts_do_not_cross_libraries,
    env_with_narwhals_and_polars_stubs(),
    r#"
import narwhals as nw
import polars as pl
from typing import Annotated
class MySchema:
    a: nw.Int64
def takes_narwhals(df: Annotated[nw.DataFrame, MySchema]) -> None: ...
takes_narwhals(nw.from_native(pl.DataFrame({"a": [1]})))
takes_narwhals(pl.DataFrame({"a": [1]}))  # E: `polars.dataframe.frame.DataFrame[a: Int64]` is not assignable to parameter `df` with type `narwhals.dataframe.DataFrame[a: Int64]`
"#,
);

narwhals_testcase!(
    test_opaque_frame_falls_back,
    r#"
import narwhals as nw
from typing import reveal_type
def f(df: nw.DataFrame) -> None:
    reveal_type(df.select("anything"))  # E: revealed type: DataFrame
"#,
);
