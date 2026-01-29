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
    bug = "fix attr handling as shown in comments below ",
    test_attrs_basic,
    r#"
from typing import reveal_type

import attr

@attr.s()
class A:
    x: int | None = attr.ib(None)

@attr.s()
class B:
    x: int | None = None

# this should not be an error
A()  # E: Missing argument `x` in function `A.__init__`  
B() 

A("hello")  # should be an error
B("hello") # E: Argument `Literal['hello']` is not assignable to parameter `x` with type `int | None` in function `B.__init__`

"#,
);
