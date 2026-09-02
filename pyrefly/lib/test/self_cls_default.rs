/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

// https://github.com/facebook/pyrefly/issues/3729: the implicit `self`/`cls` receiver of a
// method is supplied at call time, so a default value on it is unreachable and almost always a bug.

testcase!(
    test_self_param_with_default,
    r#"
class C:
    def m(self=1): ...  # E: cannot have a default value
"#,
);

testcase!(
    test_classmethod_cls_with_default,
    r#"
class C:
    @classmethod
    def m(cls=1): ...  # E: cannot have a default value
"#,
);

testcase!(
    test_positional_only_self_with_default,
    r#"
class C:
    def m(self=1, /): ...  # E: cannot have a default value
"#,
);

testcase!(
    test_dunder_new_cls_with_default,
    r#"
class C:
    def __new__(cls=1): ...  # E: cannot have a default value
"#,
);

// `__init_subclass__` is an implicit classmethod, so its `cls` receiver is also checked.
testcase!(
    test_dunder_init_subclass_cls_with_default,
    r#"
class C:
    def __init_subclass__(cls=1): ...  # E: cannot have a default value
"#,
);

// A property is still an instance method, so its `self` receiver is checked.
testcase!(
    test_property_self_with_default,
    r#"
class C:
    @property
    def p(self=1) -> int: ...  # E: cannot have a default value
"#,
);

testcase!(
    test_staticmethod_first_param_default_ok,
    r#"
class C:
    @staticmethod
    def m(x=1): ...
"#,
);

testcase!(
    test_top_level_function_default_ok,
    r#"
def f(x=1): ...
"#,
);

testcase!(
    test_non_receiver_param_default_ok,
    r#"
class C:
    def m(self, x=1): ...
    @classmethod
    def n(cls, y=2): ...
"#,
);
