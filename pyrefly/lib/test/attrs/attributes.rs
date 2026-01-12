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
