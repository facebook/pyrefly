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
