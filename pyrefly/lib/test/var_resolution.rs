/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_await_any,
    r#"
from typing import Any, assert_type
async def async_return_any() -> Any: ...
def return_any() -> Any: ...
async def test() -> None:
    z = await async_return_any()
    assert_type(z, Any)
    z = await return_any()
"#,
);

testcase!(
    test_await_bottom,
    r#"
from typing import Never, NoReturn, assert_type
def returns_never() -> Never: ...
def returns_noreturn() -> NoReturn: ...
async def test() -> None:
    z = await returns_never()
    assert_type(z, Never)
    z = await returns_noreturn()
    assert_type(z, Never)
"#,
);

testcase!(
    test_await_union,
    r#"
from typing import Never, Any, Awaitable, reveal_type
def union_any0() -> Any | Awaitable[int]: ...
def union_any1() -> Awaitable[int] | Any: ...
def union_int_str() -> Awaitable[int] | Awaitable[str]: ...
async def test() -> None:
    z = await union_any0()
    reveal_type(z)  # E: revealed type: int
    z = await union_any1()
    reveal_type(z)  # E: revealed type: int
    # (This one is a backtracking bug: we pin the var to int, then fail the
    #  str check because the upper bound has mutated and is no longer a var).
    z = await union_int_str()
"#,
);
