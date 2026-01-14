/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_double_name_match,
    r#"
match 42:
    case x:  # E: name capture `x` makes remaining patterns unreachable
        pass
    case y:
        pass
print(y)  # E: `y` may be uninitialized
    "#,
);

testcase!(
    test_guard_narrowing_in_match,
    r#"
from typing import assert_type
def test(x: int | bytes | str):
    match x:
        case int():
            assert_type(x, int)
        case _ if isinstance(x, str):
            assert_type(x, str)
    "#,
);

testcase!(
    test_pattern_crash,
    r#"
# Used to crash, see https://github.com/facebook/pyrefly/issues/490
match None:
    case {a: 1}: # E: # E: # E:
        pass
"#,
);

testcase!(
    test_pattern_dict_key_enum,
    r#"
from enum import StrEnum

class MyEnumType(StrEnum):
    A = "a"
    B = "b"

def my_func(x: dict[MyEnumType, int]) -> int:
    match x:
        case {MyEnumType.A: a, MyEnumType.B: b}:
            return a + b
        case _:
            return 0
"#,
);

testcase!(
    test_non_exhaustive_flow_merging,
    r#"
from typing import assert_type, Literal
def foo(x: Literal['A'] | Literal['B']):
    match x: # E: Match on `Literal['A', 'B']` is not exhaustive
        case 'A':
            raise ValueError()
    assert_type(x, Literal['B'])
    "#,
);

testcase!(
    test_negation_of_guarded_pattern,
    r#"
from typing import assert_type, Literal
def condition() -> bool: ...
def foo(x: Literal['A'] | Literal['B']):
    match x: # E: Match on `Literal['A', 'B']` is not exhaustive
        case 'A' if condition():
            raise ValueError()
    assert_type(x, Literal['A', 'B'])
    "#,
);

testcase!(
    test_negated_exhaustive_class_match,
    r#"
from typing import assert_type

def f0(x: int | str):
    match x:
        case int():
            pass
        case _:
            assert_type(x, str)
"#,
);

testcase!(
    test_match_alias_narrows_subject,
    r#"
from typing import assert_never, assert_type

def my_method(str_or_int: str | int) -> str:
    match str_or_int:
        case str() as str_data:
            assert_type(str_or_int, str)
            return str_data
        case int() as int_data:
            assert_type(str_or_int, int)
            return str(int_data)
        case _:
            assert_never(str_or_int)
"#,
);

testcase!(
    test_class_match_with_args_not_exhaustive,
    r#"
from typing import assert_type

class C:
    val: int

def f0(x: C):
    match x:
        case C(val=1):
            pass
        case _:
            assert_type(x, C)
"#,
);

testcase!(
    test_class_match_with_guard_not_exhaustive,
    r#"
from typing import assert_type

def condition() -> bool: ...

def f0(x: int):
    match x:
        case int() if condition():
            pass
        case _:
            assert_type(x, int)
"#,
);

testcase!(
    test_class_match_with_positional_args_not_exhaustive,
    r#"
from typing import assert_type

class C:
    val: int
    __match_args__ = ("val",)
    def __init__(self, val: int):
        self.val = val

def f0(x: C):
    match x:
        case C(1):
            pass
        case _:
            assert_type(x, C)
"#,
);

testcase!(
    test_non_exhaustive_enum_match_warning,
    r#"
from enum import Enum

class Color(Enum):
    RED = "red"
    BLUE = "blue"

def describe(color: Color):
    match color:  # E: Missing cases: Color.BLUE
        case Color.RED:
            print("danger")

def describe_ok(color: Color):
    match color:
        case Color.RED:
            print("danger")
        case Color.BLUE:
            print("ok")
"#,
);

testcase!(
    test_non_exhaustive_literal_union_match_warning,
    r#"
from typing import Literal

def describe(color: Literal["red", "blue"]):
    match color:  # E: Missing cases: 'blue'
        case "red":
            print("danger")

def describe_ok(color: Literal["red", "blue"]):
    match color:
        case "red":
            print("danger")
        case "blue":
            print("ok")
"#,
);

testcase!(
    test_non_exhaustive_enum_match_facet_subject,
    r#"
from enum import Enum

class Color(Enum):
    RED = "red"
    BLUE = "blue"

class X:
    color: Color

def describe(x: X):
    match x.color: # E: Missing cases: Color.BLUE
        case Color.RED:
            print("danger")

def describe_ok(x: X):
    match x.color:
        case Color.RED:
            print("danger")
        case Color.BLUE:
            print("ok")
"#,
);

testcase!(
    test_non_exhaustive_literal_union_match_facet_subject,
    r#"
from typing import Literal

class X:
    color: Literal["red", "blue"]

def describe(x: X):
    match x.color:  # E: Missing cases: 'blue'
        case "red":
            print("danger")

def describe_ok(x: X):
    match x.color:
        case "red":
            print("danger")
        case "blue":
            print("ok")

def describe_ok_2(x: X):
    match x.color:
        case "red":
            print("danger")
        case _:
            print("default")
"#,
);
