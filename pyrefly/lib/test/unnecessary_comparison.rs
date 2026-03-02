/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_literal_both_sides_different,
    r#"
if 1 is 2:  # E: Identity comparison `1 is 2` is always False
    pass
if True is False:  # E: Identity comparison `True is False` is always False
    pass
"#,
);

testcase!(
    test_literal_none_different,
    r#"
if None is 5:  # E: Identity comparison `None is 5` is always False
    pass
if 5 is None:  # E: Identity comparison `5 is None` is always False
    pass
"#,
);

testcase!(
    test_singleton_same_value,
    r#"
# Singletons - always the same object, so warn
if True is True:  # E: Identity comparison `True is True` is always True
    pass
if False is False:  # E: Identity comparison `False is False` is always True
    pass
if True is not True:  # E: Identity comparison `True is not True` is always False
    pass
if False is not False:  # E: Identity comparison `False is not False` is always False
    pass
"#,
);

testcase!(
    test_non_singleton_same_value_ok,
    r#"
# Same integer/string literals - no warning (interning varies)
if 1 is 1:  # OK - interning behavior varies
    pass
if "hello" is "hello":  # OK - interning behavior varies
    pass
"#,
);

testcase!(
    test_literal_one_side_ok,
    r#"
def f(x: int):
    if x is 5:  # OK - x is not a literal, only 5 is
        pass
"#,
);

testcase!(
    test_same_class_type_ok,
    r#"
class MyClass:
    pass

def f(obj1: MyClass, obj2: MyClass):
    if obj1 is obj2:  # OK - valid identity check (same type)
        pass
"#,
);

testcase!(
    test_classdef_vs_class_instance,
    r#"
class A: pass
def f(x: int, y: str, z: A):
    if x is A:  # E: Identity comparison between an instance of `int` and class `A` is always False
        pass
    if y is not A:  # E: Identity comparison between an instance of `str` and class `A` is always True
        pass
    if z is A:  # E: Identity comparison between an instance of `A` and class `A` is always False
        pass
"#,
);

testcase!(
    test_classdef_vs_type_ok,
    r#"
class A: pass
def f(x: type):
    if x is A:  # OK - A (a ClassDef) is an instance of type
        pass
"#,
);

testcase!(
    test_classdef_vs_object_ok,
    r#"
class A: pass
def f(x: object):
    if x is A:  # OK - everything is a subtype of object
        pass
"#,
);
