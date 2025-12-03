/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests describing "natural inference" behavior, which dictates the inferred
// type of expressions with literal information, like `42` and `[42]`.
//
// For literal expressions, we define a "natural type" which we will infer
// instead of the most literal type. These natural types are:
// * int
// * str
// * bool
// * float
// * complex
// ... and, for enum values, the overall enum type.
//
// We choose to infer the natural type over the more specific literal type
// when we anticipate that a subsequent write should be compatible with a
// more general type.
//
// For example, the list literal expression `[42]` should have type `list[int]`,
// not `list[Literal[42]]`, because we expect that the user will write other
// values into the list.

use crate::testcase;

testcase!(
    test_class_attribute,
    r#"
from typing import assert_type

class C:
    x = 42

c = C()
assert_type(c.x, int)
c.x = 0  # OK
"#,
);

testcase!(
    test_class_attribute_final,
    r#"
from typing import assert_type, Final, Literal

class C:
    BAR: Final = "bar"

assert_type(C.BAR, Literal["bar"])
"#,
);

testcase!(
    test_container_list,
    r#"
from typing import assert_type

xs = [42]
assert_type(xs, list[int])
xs.append(0)  # OK
"#,
);

testcase!(
    test_container_dict,
    r#"
from typing import assert_type

d = {"key": 42}
assert_type(d, dict[str, int])
d["key"] = 0  # OK
"#,
);

testcase!(
    test_container_set,
    r#"
from typing import assert_type

s = {42}
assert_type(s, set[int])
s.add(0)  # OK
"#,
);

testcase!(
    test_container_final,
    r#"
from typing import assert_type, Final

x: Final = [42]
assert_type(x, list[int])
y = x
y.append(43)  # OK
"#,
);

testcase!(
    test_function_return,
    r#"
from typing import assert_type, Literal

def foo():
    return 42

assert_type(foo(), Literal[42])
"#,
);

testcase!(
    test_generic_inference,
    r#"
from typing import assert_type

class C[T]:
    def __init__(self, x: T) -> None:
        pass

assert_type(C(42), C[int])
"#,
);

testcase!(
    test_generic_tuple,
    r#"
from typing import assert_type

def f[T](x: T) -> list[T]: ...

assert_type(f((1,)), list[tuple[int]])
"#,
);
