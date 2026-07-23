/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

/// Creates a test environment with corrected pandas stubs.
/// The `index` method has position-only markers (`/`) matching `list.index`.
fn env_with_fixed_pandas_stubs() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "pandas._typing",
        r#"
from typing import Any, Iterator, Protocol, Sequence, TypeVar, overload
from typing_extensions import SupportsIndex
_T_co = TypeVar("_T_co", covariant=True)

class SequenceNotStr(Protocol[_T_co]):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T_co]: ...
    def __contains__(self, value: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    # FIXED: All parameters position-only to match list.index
    def index(self, value: Any, start: int = ..., stop: int = ..., /) -> int: ...
    def count(self, value: Any, /) -> int: ...
    def __reversed__(self) -> Iterator[_T_co]: ...
"#,
    );
    add_pandas_core_frame(&mut env);
    add_pandas_init(&mut env);
    env
}

/// Creates a test environment with broken pandas 2.x stubs.
/// The `index` method is missing position-only markers, which doesn't match `list.index`.
/// This tests that the SequenceNotStr-specific hack in `is_subset_protocol` works.
fn env_with_broken_pandas_stubs() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "pandas._typing",
        r#"
from typing import Any, Iterator, Protocol, Sequence, TypeVar, overload
from typing_extensions import SupportsIndex
_T_co = TypeVar("_T_co", covariant=True)

class SequenceNotStr(Protocol[_T_co]):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T_co]: ...
    def __contains__(self, value: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    # BROKEN: Missing position-only markers (actual pandas 2.x stubs)
    def index(self, value: Any, start: int = ..., stop: int = ...) -> int: ...
    def count(self, value: Any, /) -> int: ...
    def __reversed__(self) -> Iterator[_T_co]: ...
"#,
    );
    add_pandas_core_frame(&mut env);
    add_pandas_init(&mut env);
    env
}

fn add_pandas_core_frame(env: &mut TestEnv) {
    env.add(
        "pandas.core.frame",
        r#"
from typing import Any
from pandas._typing import SequenceNotStr
Axes = SequenceNotStr[Any] | range

class DataFrame:
    def __init__(
        self,
        data: Any = None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Any = None,
        copy: bool | None = None,
    ) -> None: ...
"#,
    );
}

fn add_pandas_init(env: &mut TestEnv) {
    env.add(
        "pandas",
        "from pandas.core.frame import DataFrame as DataFrame",
    );
}

/// A pandas stub whose `DataFrame` lives at the real qname `pandas.core.frame` and
/// whose column-access methods return opaque types, so col-infer can pin that a pandas
/// frame carries a Partial schema yet falls back to these stubs for every transform.
fn env_with_pandas_frame_stubs() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "pandas.core.frame",
        r#"
class Series: ...
class DataFrame:
    columns: list[str]
    def __init__(self, data: object = None) -> None: ...
    def __getitem__(self, key: object) -> Series: ...
    def drop(self, labels: object = None, *, axis: int = 0) -> "DataFrame": ...
    def rename(self, mapping: object = None, *, axis: int = 0) -> "DataFrame": ...
    def filter(self, items: object = None, *, axis: int = 0) -> "DataFrame": ...
"#,
    );
    env.add(
        "pandas",
        "from pandas.core.frame import DataFrame as DataFrame, Series as Series",
    );
    env
}

testcase!(
    test_pandas_construct_partial_schema_with_trailing_marker,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
reveal_type(pd.DataFrame({"a": [1], "b": ["x"]}))  # E: revealed type: DataFrame[a: int, b: str, ...]
"#,
);

testcase!(
    test_pandas_widening_column,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
reveal_type(pd.DataFrame({"a": [2.0, 1]}))  # E: revealed type: DataFrame[a: float, ...]
"#,
);

testcase!(
    test_pandas_mixed_column_falls_back_without_error,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
# Pandas coerces a mixed column instead of raising, so we drop the schema without the Polars error.
reveal_type(pd.DataFrame({"a": [1, 2.0]}))  # E: revealed type: DataFrame
reveal_type(pd.DataFrame({"a": [1, "s"]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_pandas_unknown_column_read_delegates_without_error,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
df = pd.DataFrame({"a": [1]})
reveal_type(df["missing"])  # E: revealed type: Series
"#,
);

testcase!(
    test_pandas_known_column_read_delegates,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
df = pd.DataFrame({"a": [1]})
reveal_type(df["a"])  # E: revealed type: Series
"#,
);

testcase!(
    test_pandas_dynamic_key_unaffected,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
df = pd.DataFrame({"a": [1]})
k = "missing"
reveal_type(df[k])  # E: revealed type: Series
"#,
);

testcase!(
    test_pandas_list_subscript_falls_back,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
df = pd.DataFrame({"a": [1]})
reveal_type(df[["a", "missing"]])  # E: revealed type: Series
"#,
);

testcase!(
    test_pandas_filter_falls_back,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
# Pandas `filter` selects columns, unlike Polars, so the row-transform must not fire here.
df = pd.DataFrame({"a": [1]})
reveal_type(df.filter(items=["a"]))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_pandas_drop_falls_back,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
df = pd.DataFrame({"a": [1]})
reveal_type(df.drop("missing"))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_pandas_rename_falls_back,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
df = pd.DataFrame({"a": [1]})
reveal_type(df.rename({"missing": "z"}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_pandas_subclass_falls_back,
    env_with_pandas_frame_stubs(),
    r#"
import pandas as pd
from typing import reveal_type
class MyFrame(pd.DataFrame): ...
reveal_type(MyFrame({"a": [1]}))  # E: revealed type: MyFrame
"#,
);

testcase!(
    test_dataframe_list_str_columns,
    env_with_fixed_pandas_stubs(),
    r#"
import pandas as pd

# This should work: passing list[str] for columns
df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
"#,
);

testcase!(
    test_dataframe_list_str_both,
    env_with_fixed_pandas_stubs(),
    r#"
import pandas as pd

# Test list[str] for both columns and index
df = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6]],
    columns=["A", "B", "C"],
    index=["row1", "row2"]
)
"#,
);

// Test with BROKEN pandas 2.x stubs (without position-only markers)
// This demonstrates the SequenceNotStr-specific hack in is_subset_protocol works
testcase!(
    test_dataframe_with_broken_stubs,
    env_with_broken_pandas_stubs(),
    r#"
import pandas as pd

# This should work even with broken stubs: list[str] satisfies SequenceNotStr[Any]
# because we have a specific hack in is_subset_protocol for pandas SequenceNotStr
df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
"#,
);

// https://github.com/facebook/pyrefly/issues/3891
testcase!(
    test_sequence_not_str_element_type_overload_old_stubs,
    env_with_broken_pandas_stubs(),
    r#"
from typing import Any, Generic, Hashable, TypeVar, assert_type, overload
from pandas._typing import SequenceNotStr

CategoricalValueT = TypeVar("CategoricalValueT", str, int, float, object, default=object)

class Categorical(Generic[CategoricalValueT]):
    @overload
    def __new__(cls, values: SequenceNotStr[str]) -> "Categorical[str]": ...
    @overload
    def __new__(cls, values: SequenceNotStr[Hashable]) -> "Categorical": ...
    def __new__(cls, values: Any) -> Any: ...

assert_type(Categorical(["a", "b"]), Categorical[str])
assert_type(Categorical(["a", 1, "b"]), Categorical)
"#,
);

// https://github.com/facebook/pyrefly/issues/3891
testcase!(
    test_sequence_not_str_element_type_overload_new_stubs,
    env_with_fixed_pandas_stubs(),
    r#"
from typing import Any, Generic, Hashable, TypeVar, assert_type, overload
from pandas._typing import SequenceNotStr

CategoricalValueT = TypeVar("CategoricalValueT", str, int, float, object, default=object)

class Categorical(Generic[CategoricalValueT]):
    @overload
    def __new__(cls, values: SequenceNotStr[str]) -> "Categorical[str]": ...
    @overload
    def __new__(cls, values: SequenceNotStr[Hashable]) -> "Categorical": ...
    def __new__(cls, values: Any) -> Any: ...

assert_type(Categorical(["a", "b"]), Categorical[str])
assert_type(Categorical(["a", 1, "b"]), Categorical)
"#,
);
