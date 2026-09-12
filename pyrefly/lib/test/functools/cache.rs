/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `functools.cache` and `functools.lru_cache` preserve the wrapped function's call signature.

use crate::functools_testcase;
use crate::test::util::TestEnv;
use crate::testcase;

fn cached_overload_env() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "pkg",
        "pkg/__init__.pyi",
        r#"
from functools import lru_cache
from typing import Literal, Self, overload

class HasProps:
    @overload
    @classmethod
    @lru_cache
    def properties(cls: type[Self], *, _with_props: Literal[False] = False) -> set[str]: ...
    @overload
    @classmethod
    @lru_cache
    def properties(cls: type[Self], *, _with_props: Literal[True] = True) -> dict[str, object]: ...
"#,
    );
    env
}

functools_testcase!(
    test_cache_preserves_function_signature,
    r#"
from functools import cache
from typing import assert_type

@cache
def f() -> int:
    return 0

assert_type(f(), int)
f(1, 2, 3)  # E: Expected 0 positional arguments, got 3
f.cache_clear()
assert_type(f.cache_info().hits, int)
"#,
);

functools_testcase!(
    test_lru_cache_preserves_function_signature,
    r#"
from functools import lru_cache
from typing import assert_type

@lru_cache
def f(x: int, y: str = "") -> bool:
    return True

assert_type(f(1), bool)
assert_type(f(1, "x"), bool)
f("x")  # E: Argument `Literal['x']` is not assignable to parameter `x` with type `int`
f(1, y=2)  # E: Argument `Literal[2]` is not assignable to parameter `y` with type `str`
f.cache_info()
"#,
);

functools_testcase!(
    test_lru_cache_call_preserves_function_signature,
    r#"
from functools import lru_cache
from typing import assert_type

@lru_cache(maxsize=None)
def f(x: int) -> str:
    return ""

assert_type(f(1), str)
f("x")  # E: Argument `Literal['x']` is not assignable to parameter `x` with type `int`
f.cache_clear()
"#,
);

functools_testcase!(
    test_lru_cache_method_binds_self,
    r#"
from functools import cache, lru_cache
from typing import assert_type

class C:
    @cache
    def cache_method(self, x: int) -> str:
        return ""

    @lru_cache(maxsize=4)
    def lru_method(self, x: int) -> str:
        return ""

c = C()
assert_type(C.cache_method(c, 1), str)
assert_type(c.cache_method(1), str)
c.cache_method("x")  # E: Argument `Literal['x']` is not assignable to parameter `x` with type `int`
c.cache_method.cache_info()
assert_type(C.lru_method(c, 1), str)
assert_type(c.lru_method(1), str)
c.lru_method("x")  # E: Argument `Literal['x']` is not assignable to parameter `x` with type `int`
c.lru_method.cache_clear()
"#,
);

testcase!(
    test_lru_cache_preserves_overload_signatures_in_stubs,
    cached_overload_env(),
    r#"
from pkg import HasProps
from typing import assert_type

assert_type(HasProps.properties(), set[str])
assert_type(HasProps.properties(_with_props=True), dict[str, object])
"#,
);
