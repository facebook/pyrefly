/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::attrs_testcase;

attrs_testcase!(
    bug = "Pyrefly does not recognize attrs' automatic underscore stripping behavior for private attributes",
    private_attrs,
    r#"
import attr

@attr.s(auto_attribs=True)
class Example:
    _private: str
    public: int

# This is the correct usage per attrs behavior:
obj = Example(private="secret", public=42) # E: Missing argument `_private` in function `Example.__init__` # E: Unexpected keyword argument `private` in function `Example.__init__`
"#,
);

attrs_testcase!(
    test_attrs_basic,
    r#"
from typing import assert_type, reveal_type

import attr

@attr.s()
class A:
    x: int
    y: int | None = None


reveal_type(A.__init__)  # E: revealed type: (self: A, x: int, y: int | None = ...) -> None

a = A(1)
assert_type(a.x, int)
assert_type(a.y, int | None)
"#,
);

attrs_testcase!(
    test_attrs_attrib_pass,
    r#"
from typing import assert_type, reveal_type

import attr

@attr.s()
class A:
    x: int = attr.ib()
    y: int | None = attr.ib(None)


a = A(1)
assert_type(a.x, int)
assert_type(a.y, int | None)
"#,
);

attrs_testcase!(
    bug = "attr.ib inferred as Unknown in __init__ function",
    test_attrs_attrib_fail,
    r#"
from typing import assert_type, reveal_type

import attr

@attr.s()
class A:
    x: int = attr.ib()
    y: int | None = attr.ib(None)


reveal_type(A.__init__)  # E: revealed type: (self: A, x: Unknown, y: Unknown = ...) -> None
"#,
);
