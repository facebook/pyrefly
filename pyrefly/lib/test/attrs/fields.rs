/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::attrs_testcase;

attrs_testcase!(
    field_default_decorator,
    r#"
from attrs import define, field

@define
class C:
    a: dict = field()

    @a.default
    def _default_a(self):
        return {}

c = C()
"#,
);

attrs_testcase!(
    bug = "Recognize validator decorator",
    field_validator_decorator,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator # E: Object of class `int` has no attribute `validator`
    def _check_x(self, attribute, value):
        if value < 0:
            raise ValueError("x must be non-negative")
"#,
);
