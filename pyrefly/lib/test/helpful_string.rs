/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_helpful_string,
    TestEnv::new().enable_helpful_string_error(),
    r#"
from dataclasses import dataclass

class A: ...
f"{A()}"  # E: doesn't define a custom `__format__`, `__str__`, or `__repr__` method

def returns_none() -> None:
    pass
f"{returns_none()}"  # E: The string for `None` isn't helpful in a user-facing string

o = object()
f"{o}"  # E: The string for `object` isn't helpful in a user-facing string

def func(o: object) -> None: ...
f"{func}"  # E: doesn't define a custom `__format__`, `__str__`, or `__repr__` method

class WithStr:
    def __str__(self) -> str:
        return "ok"
f"{WithStr()}"

class WithRepr:
    def __repr__(self) -> str:
        return "ok"
f"{WithRepr()}"

class WithFormat:
    def __format__(self, format_spec: str) -> str:
        return "ok"
f"{WithFormat()}"

@dataclass
class Data:
    x: int
f"{Data(1)}"

f"{A()!s}"
f"{A()!r}"
f"{1}"
f"{'x'}"
"#,
);
