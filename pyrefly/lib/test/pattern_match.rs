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
    test_enum_member_as_class_pattern,
    r#"
from enum import Enum

class Color(Enum):
    RED = "red"

def describe(color: Color) -> None:
    match color:  # E: Match on `Color` is not exhaustive
        case Color.RED():  # E: Expected class object, got `Literal[Color.RED]`
            pass
"#,
);

testcase!(
    test_protocol_class_pattern,
    r#"
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def describe(x: object) -> None:
    match x:
        case Drawable():  # E: Protocol `Drawable` is not decorated with @runtime_checkable and cannot be used with isinstance()
            pass
        case _:
            pass
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

testcase!(
    test_sequence_pattern_star_capture,
    r#"
from collections.abc import Sequence
from typing import assert_type

def test_seq_pattern(x: Sequence[int]) -> None:
    match x:
        case [*values]:
            assert_type(values, list[int])
"#,
);

testcase!(
    test_sequence_pattern_union,
    r#"
from collections.abc import Sequence
from typing import assert_type

def test_union_seq(x: int | Sequence[int]) -> None:
    match x:
        case int(value):
            assert_type(value, int)
        case [*values]:
            assert_type(values, list[int])
"#,
);

testcase!(
    test_sequence_pattern_fixed_length,
    r#"
from collections.abc import Sequence
from typing import assert_type

def test_fixed_len(x: Sequence[int]) -> None:
    match x:
        case [a, b]:
            assert_type(a, int)
            assert_type(b, int)
"#,
);

testcase!(
    test_sequence_pattern_mixed,
    r#"
from collections.abc import Sequence
from typing import assert_type

def test_mixed(x: Sequence[int]) -> None:
    match x:
        case [first, *middle, last]:
            assert_type(first, int)
            assert_type(middle, list[int])
            assert_type(last, int)
"#,
);

testcase!(
    test_sequence_pattern_str_excluded,
    r#"
from collections.abc import Sequence
from typing import assert_type

def test_str_not_sequence(x: str | Sequence[int]) -> None:
    # str is NOT matched by sequence patterns per PEP 634
    match x:
        case [*values]:
            # If we get here, x must be Sequence[int], not str
            assert_type(values, list[int])
        case _:
            # This is str since sequences are matched by the first case
            assert_type(x, str)
"#,
);

testcase!(
    test_sequence_pattern_list,
    r#"
from typing import assert_type

def test_list_pattern(x: list[int]) -> None:
    match x:
        case [*values]:
            assert_type(values, list[int])
"#,
);

testcase!(
    test_sequence_pattern_tuple,
    r#"
from typing import assert_type

def test_tuple_pattern(x: tuple[int, ...]) -> None:
    match x:
        case [*values]:
            assert_type(values, list[int])
"#,
);

testcase!(
    test_sequence_pattern_exhaustive_assert_never,
    r#"
from collections.abc import Sequence
from typing import assert_type, assert_never

def test_seq_pattern(x: Sequence[int]) -> None:
    match x:
        case [*values]:
            assert_type(values, list[int])
        case _:
            # This should be unreachable since all sequences match [*values]
            assert_never(x)
"#,
);

testcase!(
    test_sequence_pattern_union_exhaustive,
    r#"
from collections.abc import Sequence
from typing import assert_type, assert_never

def test_seq_pat_with_union(x: int | Sequence[int]) -> None:
    match x:
        case int(value):
            assert_type(value, int)
        case [*values]:
            assert_type(values, list[int])
        case _:
            # This should be unreachable since we've covered int and Sequence[int]
            assert_never(x)
"#,
);

testcase!(
    test_exhaustive_bool_match_warning,
    r#"
def describe(flag: bool):
    match flag:
        case True:
            return "yes"
        case False:
            return "no"
    # This should NOT warn about missing return (Phase 2 will fix this)
    # For now, we're just ensuring the NonExhaustiveMatch error doesn't fire
"#,
);

testcase!(
    test_non_exhaustive_bool_match_warning,
    r#"
def describe(flag: bool):
    match flag: # E: Match on `bool` is not exhaustive
        case True:
            pass
    # Missing False case
"#,
);

testcase!(
    test_exhaustive_union_with_none,
    r#"
def process(x: int | None):
    match x:
        case int():
            pass
        case None:
            pass
    # Should not warn - union is exhausted
"#,
);

testcase!(
    test_non_exhaustive_union_with_none,
    r#"
from typing import final

@final
class A:
    pass

@final
class B:
    pass

def process(x: A | B | None):
    match x: # E: Match on `A | B | None` is not exhaustive
        case A():
            pass
        case B():
            pass
    # Missing None case
"#,
);

testcase!(
    test_exhaustive_enum_no_missing_return,
    r#"
from enum import Enum

class Color(Enum):
    RED = "red"
    BLUE = "blue"

def describe(color: Color) -> str:
    match color:
        case Color.RED:
            return "It's red"
        case Color.BLUE:
            return "It's blue"
"#,
);

testcase!(
    test_non_exhaustive_enum_missing_return,
    r#"
from enum import Enum

class Color(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"

def describe(color: Color) -> str: # E: Function declared to return `str`, but one or more paths are missing an explicit `return`
    match color: # E: Match on `Color` is not exhaustive
        case Color.RED:
            return "It's red"
        case Color.BLUE:
            return "It's blue"
    #   (Missing GREEN case here)
"#,
);

testcase!(
    test_exhaustive_literal_union_no_missing_return,
    r#"
from typing import Literal

def describe(status: Literal["pending", "done"]) -> str:
    match status:
        case "pending":
            return "Still working"
        case "done":
            return "Finished"
"#,
);

// Test that an exhaustive match with a branch that doesn't return is correctly
// identified as having an implicit None return.
testcase!(
    test_exhaustive_enum_with_branch_missing_return,
    r#"
from enum import Enum

class Color(Enum):
    RED = "red"
    BLUE = "blue"

def describe(color: Color) -> str: # E: Function declared to return `str`, but one or more paths are missing an explicit `return`
    match color:
        case Color.RED:
            return "It's red"
        case Color.BLUE:
            pass  # Exhaustive but no return here
"#,
);

testcase!(
    test_exhaustive_literal_with_branch_missing_return,
    r#"
from typing import Literal

def describe(status: Literal["pending", "done"]) -> str: # E: Function declared to return `str`, but one or more paths are missing an explicit `return`
    match status:
        case "pending":
            return "Still working"
        case "done":
            pass  # Exhaustive but no return here
"#,
);

// Regression test: match on a complex expression (not a name) should not cause
// an internal error when checking for implicit returns. The subject `1 + 1` is
// a BinOp which cannot be converted to a narrowing subject.
testcase!(
    test_match_on_complex_expr_no_internal_error,
    r#"
def foo() -> str: # E: Function declared to return `str`, but one or more paths are missing an explicit `return`
    match 1 + 1:
        case 2:
            return "two"
"#,
);

testcase!(
    test_non_exhaustive_match_shows_missing_none,
    r#"
from typing import final

@final
class A:
    pass

def process(x: A | None):
    match x: # E: Match on `A | None` is not exhaustive
        case A():
            pass
"#,
);

testcase!(
    test_non_exhaustive_match_shows_missing_class,
    r#"
from typing import final

@final
class A:
    pass

@final
class B:
    pass

@final
class C:
    pass

def process(x: A | B | C):
    match x: # E: Match on `A | B | C` is not exhaustive
        case A():
            pass
"#,
);

testcase!(
    test_match_mapping_after_none,
    r#"
from typing import Any, assert_type

def test_dict_or_none(dict_or_none: dict[str, Any] | None):
    match dict_or_none:
        case None:
            pass
        case {"a": "b"}:
            # After matching None, dict_or_none is narrowed to dict[str, Any]
            assert_type(dict_or_none, dict[str, Any])
        case _:
            assert_type(dict_or_none, dict[str, Any])

def test_sequence_after_none(seq_or_none: list[int] | None):
    match seq_or_none:
        case None:
            pass
        case [first, *rest]:
            # After matching None, seq_or_none is narrowed to list[int]
            assert_type(seq_or_none, list[int])
        case _:
            assert_type(seq_or_none, list[int])
"#,
);

testcase!(
    test_match_mapping_before_none,
    r#"
from typing import Any, assert_type

def test_dict_first(dict_or_none: dict[str, Any] | None):
    match dict_or_none:
        case {"a": "b"}:
            # IsMapping narrows dict_or_none to dict[str, Any]
            assert_type(dict_or_none, dict[str, Any])
        case None:
            pass
        case _:
            pass
"#,
);
