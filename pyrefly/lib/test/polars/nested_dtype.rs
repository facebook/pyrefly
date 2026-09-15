/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use super::dataframe::env_with_polars_stubs;
use crate::testcase;

testcase!(
    test_nested_dtype_constructors,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type

reveal_type(  # E: revealed type: DataFrame[items: List(String), array: Array(Int64, shape=(2,)), matrix: Array(Float32, shape=(2, 3)), nested_array: Array(Int32, shape=(2, 4)), array_lists: Array(List(String), shape=(3,)), record: Struct({'id': Int64, 'tags': List(String)})]
pl.DataFrame(schema={
    "items": pl.List(pl.String),
    "array": pl.Array(pl.Int64, 2),
    "matrix": pl.Array(pl.Float32, shape=(2, 3)),
    "nested_array": pl.Array(pl.Array(pl.Int32, 4), 2),
    "array_lists": pl.Array(pl.List(pl.String), shape=3),
    "record": pl.Struct({"id": pl.Int64, "tags": pl.List(pl.String)}),
}))
"#,
);

testcase!(
    test_nested_dtype_keyword_arguments,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type

reveal_type(  # E: revealed type: DataFrame[items: List(Int16), array: Array(UInt8, shape=(3,)), empty_array: Array(Int64, shape=(0,)), record: Struct({'value': Float64})]
pl.DataFrame(schema={
    "items": pl.List(inner=pl.Int16),
    "array": pl.Array(inner=pl.UInt8, shape=(3,)),
    "empty_array": pl.Array(inner=pl.Int64, shape=0),
    "record": pl.Struct(fields={"value": pl.Float64}),
}))
"#,
);

testcase!(
    test_dynamic_and_unsupported_nested_dtype_constructors_fall_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type

def dynamic_shape() -> int: ...

reveal_type(pl.DataFrame(schema={"array": pl.Array(pl.Int64, dynamic_shape())}))  # E: revealed type: DataFrame
reveal_type(pl.DataFrame(schema={"array": pl.Array(pl.Int64, -1)}))  # E: revealed type: DataFrame
reveal_type(pl.DataFrame(schema={"array": pl.Array(pl.Int64, ())}))  # E: revealed type: DataFrame
reveal_type(pl.DataFrame(schema={"record": pl.Struct({"field": object})}))  # E: revealed type: DataFrame

class List:
    def __init__(self, inner: object) -> None: ...

reveal_type(pl.DataFrame(schema={"items": List(pl.Int64)}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_scalar_dtype_calls_still_use_inferred_type,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type

def dtype_factory() -> pl.Int16:
    return pl.Int16()

reveal_type(pl.DataFrame(schema={"direct": pl.Int64(), "factory": dtype_factory()}))  # E: revealed type: DataFrame[direct: Int64, factory: Int16]
"#,
);

testcase!(
    test_nested_dtype_arguments_are_checked,
    env_with_polars_stubs(),
    r#"
import polars as pl

pl.DataFrame(schema={"items": pl.List(pl.Int64(1))})  # E: Expected 0 positional arguments, got 1
"#,
);
