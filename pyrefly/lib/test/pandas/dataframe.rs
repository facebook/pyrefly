/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_dataframe_list_str_columns,
    {
        let mut env = TestEnv::new();
        // Add corrected pandas stubs inline
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
        env.add(
            "pandas",
            r#"
from pandas.core.frame import DataFrame as DataFrame
"#,
        );
        env
    },
    r#"
import pandas as pd

# This should work: passing list[str] for columns
df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
"#,
);

testcase!(
    test_dataframe_list_str_both,
    {
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
        env.add(
            "pandas",
            "from pandas.core.frame import DataFrame as DataFrame",
        );
        env
    },
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
