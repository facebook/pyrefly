/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_is,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is None:
        assert_type(x, None)
    assert_type(x, str | None)
    "#,
);

testcase!(
    test_assert_narrow_message,
    r#"
from typing import assert_type
def test(x: str | None):
    assert x is None, x[0]
    assert_type(x, None)
    "#,
);

testcase!(
    test_truthy_falsy,
    r#"
from typing import assert_type, Literal
def f(x: str | None, y: bool):
    if x:
        assert_type(x, str)
    if y:
        assert_type(y, Literal[True])
    else:
        assert_type(y, Literal[False])
    "#,
);

testcase!(
    test_bool_simple,
    r#"
from typing import assert_type
def f(x: str | None):
    if bool(x):
        assert_type(x, str)
    "#,
);

testcase!(
    test_eq,
    r#"
from typing import assert_type
def f(x: str | None):
    if x == None:
        assert_type(x, None)
    "#,
);

testcase!(
    test_neq,
    r#"
from typing import assert_type
def f(x: str | None):
    if x != None:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_not,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is not None:
        assert_type(x, str)
    "#,
);

testcase!(
    test_if_else,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is None:
        assert_type(x, None)
    else:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_subtype,
    r#"
from typing import assert_type
class A: pass
class B(A): pass
def f(x: type[A]):
    if x is B:
        assert_type(x, type[B])
    "#,
);

testcase!(
    test_is_never,
    r#"
from typing import assert_type, Never
def f(x: str):
    if x is None:
        assert_type(x, Never)
    "#,
);

testcase!(
    test_is_not_bool_literal,
    r#"
from typing import assert_type, Literal, Never
def f1(x: bool):
    if x is not True:
        assert_type(x, Literal[False])
def f2(x: Literal[True] | str):
    if x is not True:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_not_enum_literal,
    r#"
from typing import assert_type, Literal
import enum
class E(enum.Enum):
    X = 1
    Y = 2
def f1(x: Literal[E.X, E.Y]):
    if x is not E.X:
        assert_type(x, Literal[E.Y])
def f2(x: E | int):
    if x is not E.X:
        assert_type(x, Literal[E.Y] | int)
    "#,
);

testcase!(
    test_is_final_literal_sentinel,
    r#"
from typing import Final, Literal, assert_type

S: Final[bool] = True
def f(x: bool):
    if x is S:
        assert_type(x, Literal[True])
    else:
        assert_type(x, Literal[False])
    if x is not S:
        assert_type(x, Literal[False])
    "#,
);

testcase!(
    test_is_final_object_sentinel,
    r#"
from typing import Final

class Sentinel:
    pass

S: Final[object] = Sentinel()

def takes_s(x: Sentinel) -> None: ...

def f(x: Sentinel | int):
    if x is S:
        takes_s(x)
    "#,
);

testcase!(
    test_ellipsis_is,
    r#"
from typing import reveal_type
from types import EllipsisType

def f(x: int | EllipsisType):
    if x is ...:
        reveal_type(x)  # E: Ellipsis
    else:
        reveal_type(x)  # E: int
    if x is not ...:
        reveal_type(x)  # E: int
    else:
        reveal_type(x)  # E: Ellipsis
    "#,
);

testcase!(
    test_ellipsis_eq,
    r#"
from typing import reveal_type
from types import EllipsisType

def f(x: int | EllipsisType):
    if x == ...:
        reveal_type(x)  # E: Ellipsis
    else:
        reveal_type(x)  # E: int
    if x != ...:
        reveal_type(x)  # E: int
    else:
        reveal_type(x)  # E: Ellipsis
    "#,
);

testcase!(
    test_tri_enum,
    r#"
from typing import assert_type, Literal
import enum
class E(enum.Enum):
    X = 1
    Y = 2
    Z = 3
def f(x: E):
    if x is E.X:
       assert_type(x, Literal[E.X])
    elif x is E.Y:
       assert_type(x, Literal[E.Y])
    else:
       assert_type(x, Literal[E.Z])
    "#,
);

testcase!(
    test_is_classdef,
    r#"
from typing import assert_type
class A: pass
class B: pass
def f1(x: type[A] | type[B]):
    if x is A:
        assert_type(x, type[A])
    else:
        # Note that we cannot narrow to `type[B]` here, as `type` is covariant and `x` may be a
        # subtype of `A`.
        assert_type(x, type[A] | type[B])
    "#,
);

testcase!(
    test_and,
    r#"
from typing import assert_type, Literal, Never
def f(x: bool | None):
    if x is True and x is None:  # E: Identity comparison `True is None` is always False
        assert_type(x, Never)
    else:
        assert_type(x, bool | None)
    "#,
);

testcase!(
    test_and_multiple_vars,
    r#"
from typing import assert_type, Literal
def f(x: bool | None, y: bool | None):
    if x is True and y is False:
        assert_type(x, Literal[True])
        assert_type(y, Literal[False])
    "#,
);

testcase!(
    test_or,
    r#"
from typing import assert_type, Literal
def f(x: bool | None):
    if x == True or x is None:
        assert_type(x, Literal[True] | None)
    else:
        assert_type(x, Literal[False])
    "#,
);

testcase!(
    test_or_multiple_vars,
    r#"
from typing import assert_type
def f(x: int | None, y: int | None) -> None:
    if x is None or y is None:
        assert_type(y, int | None)
        assert_type(x, int | None)
    else:
        assert_type(y, int)
        assert_type(x, int)
    "#,
);

testcase!(
    test_or_walrus_multiple_vars,
    r#"
from typing import assert_type
def f(x: None | int, y: None | int) -> None:
    if (z := x is None) or y is None:
        assert_type(x, None | int)
        assert_type(y, None | int)
        assert_type(z, bool)
    "#,
);

testcase!(
    test_elif,
    r#"
from typing import assert_type
def f(x: str | None, y: int | None):
    if x is None:
        assert_type(x, None)
        assert_type(y, int | None)
    elif y is None:
        assert_type(x, str)
        assert_type(y, None)
    else:
        assert_type(x, str)
        assert_type(y, int)
    "#,
);

testcase!(
    test_not,
    r#"
from typing import assert_type, Literal
def f(x: str | None):
    if not x is None:
        assert_type(x, str)
    else:
        assert_type(x, None)
    if not x:
        assert_type(x, None | Literal[""])
    else:
        assert_type(x, str)
    "#,
);

testcase!(
    test_exit,
    r#"
from typing import assert_type
import sys
import os
def test_sys_exit(x: str | None):
    if not x:
        sys.exit(1)
    assert_type(x, str)
def test_exit(x: str | None):
    if not x:
        exit(1)
    assert_type(x, str)
def test_quit(x: str | None):
    if not x:
        quit(1)
    assert_type(x, str)
def test_os_exit(x: str | None):
    if not x:
        os._exit(1)
    assert_type(x, str)
    "#,
);

testcase!(
    test_not_and,
    r#"
from typing import assert_type, Literal
def f(x: bool | None):
    if not (x is True and x is None):  # E: Identity comparison `True is None` is always False
        assert_type(x, Literal[False] | bool | None)
    "#,
);

testcase!(
    test_assert,
    r#"
from typing import assert_type
def f(x: str | None):
    assert x is not None
    assert_type(x, str)
    "#,
);

testcase!(
    test_prod_assert,
    r#"
from typing import assert_type
def prod_assert(x: object, msg: str | None = None): ...

def test_only(x: str | None) -> None:
    prod_assert(x is not None)
    assert_type(x, str)

def test_and_message(x: str | None) -> None:
    prod_assert(x is not None, "x is None")
    assert_type(x, str)
    "#,
);

testcase!(
    test_while_else,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    assert_type(x, None)
    x = f()
    assert_type(x, str | None)
else:
    assert_type(x, str)
assert_type(x, str)
    "#,
);

testcase!(
    test_while_break_no_else,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    break
assert_type(x, str | None)
    "#,
);

testcase!(
    test_while_break_else,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    if f():
        break
else:
    assert_type(x, str)
assert_type(x, str | None)
    "#,
);

testcase!(
    test_while_overwrite,
    r#"
from typing import assert_type, Literal
def f() -> str | None: ...
x = f()
while x is None:
    if f():
        x = 42
        break
assert_type(x, Literal[42] | str)
    "#,
);

testcase!(
    test_while_narrow,
    r#"
from typing import assert_type, Literal
def test(x: bool, z: bool):
    while x:
        assert_type(x, Literal[True])
    while y := z:
        assert_type(y, Literal[True])
        assert_type(z, Literal[True])
    "#,
);
testcase!(
    test_nested_function,
    r#"
from typing import assert_type
def foo(x: int | None) -> None:
    def include():
        if x is not None:
            assert_type(x, int)
    "#,
);

testcase!(
    test_multiple_is,
    r#"
from typing import assert_type, Never
def f(x: bool | None, y: bool | None):
    if x is None is None:
        assert_type(x, None)
    if y is None is True:  # E: Identity comparison `None is True` is always False
        assert_type(y, Never)
    "#,
);

testcase!(
    test_class_body,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
class C:
    if x is None:
        assert_type(x, None)
    "#,
);

testcase!(
    test_walrus_target,
    r#"
from typing import assert_type
def f() -> str | None:
    pass
if x := f():
    assert_type(x, str)
    "#,
);

testcase!(
    test_walrus_with_rhs_narrowing,
    r#"
from typing import assert_type, Literal
def f(x: None | int) -> int:
    # this is: y := (x is None)
    # `y` being truthy means `x is None` is also true
    if y := x is None:
        assert_type(y, Literal[True])
        x = 1
    return x
    "#,
);

testcase!(
    test_walrus_value,
    r#"
from typing import assert_type
def f(x: int | None):
    if y := x:
        assert_type(x, int)
        assert_type(y, int)
    "#,
);

testcase!(
    test_walrus_comparison,
    r#"
from typing import assert_type
def f() -> str | None:
    pass
if (x := f()) is None:
    assert_type(x, None)
    "#,
);

testcase!(
    test_walrus_comprehension_if_simple,
    r#"
from typing import assert_type
def f(xs: list[int | None]):
    ys = [y111 for x in xs if (y111 := x)]
    assert_type(ys, list[int])
    "#,
);

testcase!(
    test_walrus_comprehension_if_function,
    r#"
from typing import assert_type
def get_y(x: int | None) -> int | None:
    return x
def f(xs: list[int | None]):
    ys = [y111 for x in xs if (y111 := get_y(x))]
    assert_type(ys, list[int])
    "#,
);

testcase!(
    test_walrus_generator_if,
    r#"
from typing import Sequence
def foo(x: int) -> int | None:
    return (x + 5) if x % 2 else None
foos: Sequence[int] = tuple(
    maybe_foo
    for x in range(10)
    if (maybe_foo := foo(x)) is not None
)
    "#,
);

testcase!(
    test_walrus_ternary,
    r#"
from typing import assert_type
def get_y(x: int | None) -> int | None:
    return x
def f(x: int | None):
    val = y if (y := get_y(x)) else 0
    assert_type(val, int)
    "#,
);

testcase!(
    test_match_enum_fallback,
    r#"
from typing import assert_type, Literal
from enum import Enum
class E(Enum):
    X = 1
    Y = 2
    Z = 3
def f(e: E):
    match e:
        case E.X:
            assert_type(e, Literal[E.X])
        case E.Y:
            assert_type(e, Literal[E.Y])
        case _:
            assert_type(e, Literal[E.Z])
    "#,
);

testcase!(
    test_match_or,
    r#"
from typing import assert_type, Literal
def f(e: bool | None):
    match e:
        case True | None:
            assert_type(e, Literal[True] | None)
        case _:
            assert_type(e, Literal[False])
    "#,
);

testcase!(
    test_ternary,
    r#"
from typing import assert_type
def f(x: str | None, y: int):
    z = x if x else y
    assert_type(x, str | None)
    assert_type(y, int)
    assert_type(z, str | int)
    "#,
);

testcase!(
    test_is_supertype,
    r#"
from typing import Literal, assert_type
import enum
class E(enum.Enum):
    X = 1
def f(x: Literal[E.X], y: E):
    if x is y:
        assert_type(x, Literal[E.X])
    "#,
);

testcase!(
    test_isinstance,
    r#"
from typing import assert_type
def f(x: str | int):
    if isinstance(x, str):
        assert_type(x, str)
    else:
        assert_type(x, int)
    "#,
);

testcase!(
    test_dunder_bool_truthy_narrow,
    r#"
from typing import assert_type, Literal
class Falsey:
    def __bool__(self) -> Literal[False]:
        return False
class Truthy:
    def __bool__(self) -> Literal[True]:
        return True
def f(x: Falsey | Truthy):
    if x:
        assert_type(x, Truthy)
    else:
        assert_type(x, Falsey)
    "#,
);

testcase!(
    test_type_eq,
    r#"
from typing import assert_type
def f(x: str | int):
    if type(x) == str:
        assert_type(x, str)
    else:
        # x can still be a subclass of str
        assert_type(x, str | int)
    if type(x) is str:
        assert_type(x, str)
    else:
        # x can still be a subclass of str
        assert_type(x, str | int)

def verify_type(input: int):
    pass

def foo(x: int | None) -> None:
    assert type(x) is int
    verify_type(x)
    "#,
);

testcase!(
    test_isinstance_union,
    r#"
from typing import assert_type
def f(x: str | int | None):
    if isinstance(x, str | int):
        assert_type(x, str | int)
    else:
        assert_type(x, None)
    "#,
);

testcase!(
    test_isinstance_of_tuple,
    r#"
from typing import assert_type
def f(x):
    if isinstance(x, tuple | int):
        assert_type(x, tuple | int)
        if isinstance(x, tuple):
            assert_type(x, tuple)
        else:
            assert_type(x, int)
"#,
);

testcase!(
    test_isinstance_tuple_union_multiple,
    r#"
from typing import assert_type

def make_it() -> int | tuple[int, str] | tuple[bool] | tuple[str]:
    assert False

def test():
    c = make_it()
    if isinstance(c, tuple):
        assert_type(c, tuple[int, str] | tuple[bool] | tuple[str])
    else:
        assert_type(c, int)

def test_negative():
    c = make_it()
    if not isinstance(c, tuple):
        assert_type(c, int)
    else:
        assert_type(c, tuple[int, str] | tuple[bool] | tuple[str])
"#,
);

testcase!(
    test_isinstance_list_union_multiple,
    r#"
from typing import assert_type

def make_it() -> int | list[int] | list[bool] | list[str]:
    assert False

def test():
    c = make_it()
    if isinstance(c, list):
        assert_type(c, list[int] | list[bool] | list[str])
    else:
        assert_type(c, int)

def test_negative():
    c = make_it()
    if not isinstance(c, list):
        assert_type(c, int)
    else:
        assert_type(c, list[int] | list[bool] | list[str])
"#,
);

testcase!(
    test_isinstance_and_len_narrow,
    r#"
from typing import assert_type

def test(x: str | tuple[int, str] | tuple[int, str, str]):
    if isinstance(x, tuple) and len(x) == 3:
        assert_type(x, tuple[int, str, str])

def test2(x: str | tuple[int, str] | tuple[int, str, str]) -> str:
    if (not isinstance(x, tuple)) or len(x) != 3:
        assert_type(x, str | tuple[int, str])
        return "nope"
    assert_type(x, tuple[int, str, str])
    _, _, s = x
    return s
"#,
);

testcase!(
    test_isinstance_iterable_union_multiple,
    r#"
from typing import assert_type, Iterable

def make_it() -> int | tuple[int, str] | list[bool] | list[str]:
    assert False

def test():
    c = make_it()
    if isinstance(c, Iterable):
        assert_type(c, list[bool] | list[str] | tuple[int, str])
    else:
        assert_type(c, int)

def test_negative():
    c = make_it()
    if not isinstance(c, Iterable):
        assert_type(c, int)
    else:
        assert_type(c, list[bool] | list[str] | tuple[int, str])
"#,
);

testcase!(
    test_isinstance_mapping_list,
    r#"
from typing import Mapping, assert_type
def test(
    response: Mapping[str, object] | list[Mapping[str, object]],
) -> dict[str, object]:
    if isinstance(response, list):
        assert_type(response, list[Mapping[str, object]])
        return {
            "result": [
                {
                    "node_id": item["id"],
                    "node": item,
                }
                for item in response
            ]
        }
    else:
        assert_type(response, Mapping[str, object])
        return {
            "result": [
                {
                    "node_id": response["id"],
                    "node": response,
                }
            ]
        }
"#,
);

testcase!(
    test_isinstance_of_none,
    r#"
from typing import assert_type
def f(x):
    if isinstance(x, None | int):
        assert_type(x, None | int)
        if isinstance(x, int):
            assert_type(x, int)
        else:
            assert_type(x, None)

def g(x):
    isinstance(x, None) # E: `None` is not assignable to parameter
"#,
);

testcase!(
    test_isinstance_final_intersection,
    r#"
from typing import assert_never, final

@final
class A: ...
class B: ...

def f(x: A):
    if isinstance(x, B):
        assert_never(x)
    "#,
);

testcase!(
    test_isinstance_enum_intersection,
    r#"
from enum import Enum
from typing import assert_never, reveal_type

class E1(Enum):
    pass

class E2(Enum):
    X = 1

class A: ...

def f(x: A):
    if isinstance(x, E1):
        reveal_type(x) # E: A & E1
    if isinstance(x, E2):
        assert_never(x)
    "#,
);

testcase!(
    test_final_intersect_typevar,
    r#"
from typing import assert_type, Callable, final

@final
class C:
    x: int

def f[R](g: Callable[[], R]) -> R:
    x = g()
    if isinstance(x, C):
        assert_type(x.x, int)
    return x
    "#,
);

testcase!(
    test_isinstance_tuple,
    r#"
from typing import assert_type
def f(x: str | int | None):
    if isinstance(x, (str, int)):
        assert_type(x, str | int)
    else:
        assert_type(x, None)
    "#,
);

testcase!(
    test_isinstance_unbounded_tuple,
    r#"
from typing import assert_type

def test(x, y: tuple[type[int], ...]):
    if isinstance(x, y):
        assert_type(x, int)
"#,
);

testcase!(
    test_isinstance_type,
    r#"
from typing import assert_type, Any

def f(x: object, y: type[str]) -> None:
    if isinstance(x, y):
        assert_type(x, str)

def g(x: object, y: type[Any]) -> None:
    if isinstance(x, y):
        assert_type(x, Any)
"#,
);

testcase!(
    test_isinstance_type_negative_no_narrow,
    r#"
from typing import assert_type

def f(cls: type[int], x: int | str) -> None:
    if isinstance(x, cls):
        assert_type(x, int)
    else:
        # cls might be a subclass of int, so x can still be an int here
        assert_type(x, int | str)
"#,
);

testcase!(
    test_isinstance_type_negative_partial_narrow,
    r#"
from typing import assert_type

def f(cls: type[int], x: int | str | bytes) -> None:
    if isinstance(x, (cls, str)):
        assert_type(x, int | str)
    else:
        # cls might be a subclass of int, so x can still be an int here
        assert_type(x, int | bytes)

def g(cls: type[int], x: int | str | bytes) -> None:
    if isinstance(x, cls | str):
        assert_type(x, int | str)
    else:
        # cls might be a subclass of int, so x can still be an int here
        assert_type(x, int | bytes)
"#,
);

testcase!(
    test_is_not_instance_no_narrow,
    r#"
from typing import assert_type

def f(cls: type[int], x: int | str) -> None:
    if not isinstance(x, cls):
        # cls might be a subclass of int, so x can still be an int here
        assert_type(x, int | str)
    "#,
);

testcase!(
    test_is_not_instance_alias,
    r#"
from typing import TypeAlias, assert_type

X1 = int
def f(x: int | str) -> None:
    if not isinstance(x, X1):
        assert_type(x, str)

X2: TypeAlias = int
def g(x: int | str) -> None:
    if not isinstance(x, X2):
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_not_instance_of_final_class,
    r#"
from typing import assert_type, final
@final
class C: ...
def f(cls: type[C], x: C | int):
    if not isinstance(x, cls):
        # Because `C` is final, we can assume that `cls` is exactly `C` and has been narrowed away
        assert_type(x, int)
    "#,
);

testcase!(
    test_not_issubclass_no_narrow,
    r#"
from typing import assert_type

def f(cls: type[int], x: type[int] | type[str]):
    if not issubclass(x, cls):
        # cls might be a subclass of int, so x can still be int here
        assert_type(x, type[int] | type[str])
    "#,
);

testcase!(
    test_issubclass_union,
    r#"
from typing import assert_type
class A: ...
class B: ...
class C: ...
def f(x: type[A | B | C]):
    if issubclass(x, A | B):
        assert_type(x, type[A] | type[B])
    else:
        assert_type(x, type[C])
    "#,
);

testcase!(
    test_issubclass_tuple,
    r#"
from typing import assert_type
class A: ...
class B: ...
class C: ...
def f(x: type[A | B | C]):
    if issubclass(x, (A, B)):
        assert_type(x, type[A] | type[B])
    else:
        assert_type(x, type[C])
    "#,
);

testcase!(
    test_isinstance_alias,
    r#"
from typing import assert_type
X = int
def f(x: str | int):
    if isinstance(x, X):
        assert_type(x, int)
    "#,
);

testcase!(
    test_isinstance_alias_of_union,
    r#"
class A: ...
class B(A): ...
class C(A): ...

X = B | C

def f(x: A) -> X:
    if isinstance(x, X):
        return x
    raise ValueError()
    "#,
);

// Using scoped type aliases with isinstance is a runtime error.
testcase!(
    test_isinstance_alias_error,
    r#"
type X = int
type Y = int | str
isinstance(1, X)  # E: Expected class object
isinstance(1, Y)  # E: Expected class object
    "#,
);

testcase!(
    test_isinstance_error,
    r#"
from typing import assert_type
def f(x: int | list[int]):
    if isinstance(x, list[int]):  # E: Expected class object
        # Either `int | list[int]` or `list[int]` is acceptable for the narrowed type.
        assert_type(x, list[int])
    "#,
);

testcase!(
    test_isinstance_parameterized_type,
    r#"
from typing import assert_type
def f(x: int | list[int], y: type[list[int]]):
    # Note that a literal `list[int]` as the second argument is illegal, but this is ok because
    # `y` may be a class object at runtime.
    if isinstance(x, y):
        assert_type(x, list[int])
    "#,
);

testcase!(
    bug = "We mistakenly think y[0] is a parameterized type because of the square brackets",
    test_isinstance_subscript_bug,
    r#"
def f(x, y: list[type[list[int]]]):
    return isinstance(x, y[0])  # E: Expected class object
    "#,
);

testcase!(
    test_isinstance_aliased,
    r#"
from typing import assert_type
istype = isinstance
def f(x: int | str):
    if istype(x, int):
        assert_type(x, int)
    "#,
);

testcase!(
    test_guarded_attribute_access_and,
    r#"
class A:
    x: str
class B:
    pass
def f(x: A | B):
    return isinstance(x, A) and x.x
    "#,
);

testcase!(
    test_guarded_attribute_access_or,
    r#"
class A:
    x: str
def f(x: A | None):
    return x is None or x.x
    "#,
);

testcase!(
    test_and_chain_with_walrus,
    r#"
from typing import assert_type, Literal

class A: ...
class B: ...

def test(x: A | B):
    y = isinstance(x, A) and (z := True)
    assert_type(x, A | B)
    # Intended false negative for uninitialized local check.
    assert_type(z, Literal[True])
    "#,
);

testcase!(
    test_typeguard_basic,
    r#"
from typing import TypeGuard, assert_type
class Cat:
    color: str
class Dog:
    pass
def is_black_cat(x: Cat | Dog) -> TypeGuard[Cat]:
    return isinstance(x, Cat) and x.color == "black"
def f(x: Cat | Dog):
    if is_black_cat(x):
        assert_type(x, Cat)
    else:
        assert_type(x, Cat | Dog)
    is_black_cat(1)  # E: Argument `Literal[1]` is not assignable to parameter `x` with type `Cat | Dog` in function `is_black_cat`
    "#,
);

testcase!(
    test_typeis,
    r#"
from typing import TypeIs, assert_type
class Cat:
    color: str
class Dog:
    pass
def is_cat(x: Cat | Dog) -> TypeIs[Cat]:
    return isinstance(x, Cat)
def f(x: Cat | Dog):
    if is_cat(x):
        assert_type(x, Cat)
    else:
        assert_type(x, Dog)
    "#,
);

testcase!(
    test_typeis_union,
    r#"
from typing import TypeIs, assert_type
class A: ...
class B: ...
class C: ...
def is_a_or_b(x: object) -> TypeIs[A | B]:
    return isinstance(x, A) or isinstance(x, B)
def f(x:  A | B | C, y: A | C):
    if is_a_or_b(x):
        assert_type(x, A | B)
    else:
        assert_type(x, C)
    if is_a_or_b(y):
        assert_type(y, A)
    else:
        assert_type(y, C)
    "#,
);

testcase!(
    test_narrow_and,
    r#"
from typing import assert_type
foo: dict[str, str] = {}
if "foo" in foo and foo["foo"] is not "as":
    val = foo["foo"]
    assert_type(val, str)
"#,
);

testcase!(
    test_issubclass,
    r#"
from typing import assert_type, reveal_type
class A: ...
class B(A): ...
def f(x: type[B] | type[int]):
    if issubclass(x, A):
        reveal_type(x)  # E: type[(A & int) | B]
    else:
        assert_type(x, type[int])
    "#,
);

testcase!(
    test_issubclass_nondisjoint_classes,
    r#"
from typing import reveal_type

class A: ...
class B: ...

def f(x: type[A]):
    if issubclass(x, B):
        reveal_type(x)  # E: type[A & B]
    "#,
);

testcase!(
    test_issubclass_disjoint_classes,
    r#"
from typing import assert_type, Never
def f(x: type[int]):
    if issubclass(x, str):
        assert_type(x, type[Never])
    "#,
);

testcase!(
    test_call_after_issubclass,
    r#"
class A: ...
class B: ...
def f(x: type[A]):
    if issubclass(x, B):
        return x()
    "#,
);

testcase!(
    test_attribute_access_after_issubclass,
    r#"
from typing import assert_type
class A: ...
class B:
    b: int
def f(x: type[A]):
    if issubclass(x, B):
        assert_type(x.b, int)
    "#,
);

testcase!(
    test_issubclass_error,
    r#"
def f(x: int):
    if issubclass(x, int):  # E: Argument `int` is not assignable to parameter `cls` with type `type`
        return True
    "#,
);

testcase!(
    test_issubclass_bare_type,
    r#"
from typing import assert_type, Any

class Foo: ...

def test_bare_type(x: type) -> None:
    # `type` is equivalent to `type[Any]`, so issubclass can narrow it
    if issubclass(x, Foo):
        assert_type(x, type[Foo])

def test_type_any(x: type[Any]) -> None:
    if issubclass(x, Foo):
        assert_type(x, type[Foo])

def test_isinstance_then_issubclass(x: object) -> None:
    # Common pattern: check if x is a class, then check if it's a subclass of Foo
    if isinstance(x, type) and issubclass(x, Foo):
        assert_type(x, type[Foo])
    "#,
);

testcase!(
    test_issubclass_typevar_object,
    r#"
from typing import TypeVar

class Foo:
    @classmethod
    def check(cls) -> None:
        ...

T = TypeVar("T", bound=type[object])

def needs_foo(cls: type[Foo]) -> None:
    cls.check()

def check(t: T) -> T:
    if issubclass(t, Foo):
        needs_foo(t)
        t.check()
        return t
    return t
    "#,
);

testcase!(
    test_isinstance_typevar_intersection,
    r#"
def test[T: int | str](value: T) -> T:
    if isinstance(value, int):
        return value
    else:
        return value
    "#,
);

testcase!(
    test_issubclass_typevar_nondisjoint_classes,
    r#"
from typing import reveal_type

class A: ...
class B: ...

def f[T: type[A]](x: T) -> T:
    if issubclass(x, B):
        reveal_type(x)  # E: type[B] & T
    return x
    "#,
);

testcase!(
    test_issubclass_typevar_disjoint_classes,
    r#"
from typing import assert_type, Never
def f[T: type[int]](x: T) -> T:
    if issubclass(x, str):
        assert_type(x, type[Never])
    return x
    "#,
);

testcase!(
    test_issubclass_typevar_union,
    r#"
from typing import assert_type, Never
def f1[T: type[str]](x: T | type[bytes]) -> T | type[bytes]:
    if issubclass(x, int):
        assert_type(x, type[Never])
    return x

def f2[T: type[str] | type[bytes]](x: T) -> T:
    if issubclass(x, int):
        assert_type(x, type[Never])
    return x
    "#,
);

testcase!(
    test_typeguard_instance_method,
    r#"
from typing import TypeGuard, assert_type
class C:
    def is_positive_int(self, x: object) -> TypeGuard[int]:
        return isinstance(x, int) and x > 0
def f(c: C, x: int | str):
    if c.is_positive_int(x):
        assert_type(x, int)
    "#,
);

testcase!(
    test_typeguard_generic_function,
    r#"
from typing import TypeGuard, assert_type
def f[T](x: object, y: T, z: T) -> TypeGuard[int]: ...
def f2[T](x: object, y: T) -> TypeGuard[T]: ...
def g(x: int | str):
    if f(x, 0, 0):
        assert_type(x, int)
    if f2(x, ""):
        assert_type(x, str)
    "#,
);

testcase!(
    test_implicit_else,
    r#"
from typing import assert_type
def f(x: int | None):
    if not x:
        return
    assert_type(x, int)
    "#,
);

testcase!(
    test_narrowed_elif_test,
    r#"
def f(x: int | None, y: bool):
    if not x:
        pass
    elif x > 42:
        pass
"#,
);

testcase!(
    test_narrow_comprehension,
    r#"
from typing import assert_type
def f(xs: list[int | None]):
    ys = [x for x in xs if x]
    assert_type(ys, list[int])
"#,
);

// Note: the narrowing code isn't actually what's giving us this behavior,
// it comes from flow-aware type information taking precedence over static
// annotations. But the end result is narrowing behavior.
testcase!(
    test_assignment_and_narrowing,
    r#"
from typing import assert_type, Literal
def foo(x: int | str):
    y: int | str = x
    assert_type(x, int | str)
    assert_type(y, int | str)
    x = 42
    y = 42
    assert_type(x, Literal[42])
    assert_type(y, Literal[42])
    "#,
);

testcase!(
    test_bad_typeguard_return,
    r#"
from typing import TypeGuard
def f(x) -> TypeGuard[str]:
    return "oops"  # E: Returned type `Literal['oops']` is not assignable to expected return type `bool` of type guard functions
def g(x) -> TypeGuard[str]:  # E: Function declared to return `TypeGuard[str]` but is missing an explicit `return`
    pass
    "#,
);

testcase!(
    test_isinstance_any_second,
    r#"
from typing import Any
def f(x: int | str, y: Any):
    if isinstance(x, y):
        pass
    "#,
);

testcase!(
    test_isinstance_any_literally,
    r#"
from typing import Any
def f(x: int | str):
    if isinstance(x, Any): # E: Expected class object, got `Any`
        pass
    "#,
);

testcase!(
    test_isinstance_any_first,
    r#"
from typing import Any, assert_type
def f(x: Any):
    if isinstance(x, bool):
        assert_type(x, bool)
    else:
        assert_type(x, Any)
"#,
);

testcase!(
    test_listcomp_if_control_flow,
    r#"
class C: pass
class D(C): pass
def accepts_d(x: D) -> None: pass
def f(x: list[C], z: C):
    if accepts_d(z) and isinstance(z, D):  # E: Argument `C` is not assignable to parameter `x` with type `D`
        pass
    [y for y in x if (accepts_d(y) and isinstance(y, D))]  # E: Argument `C` is not assignable to parameter `x` with type `D` in function `accepts_d`
    [None for y in x if C.error]  # E: Class `C` has no class attribute `error`
    "#,
);

testcase!(
    test_unittest_assert,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_true(self) -> None:
        x = foo()
        self.assertTrue(x is not None)
        assert_type(x, int)

    def test_false(self) -> None:
        x = foo()
        self.assertFalse(x is None)
        assert_type(x, int)
"#,
);

testcase!(
    test_unittest_assert_none,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_is_none(self) -> None:
        x = foo()
        self.assertIsNone(x)
        assert_type(x, None)

    def test_is_not_none(self) -> None:
        x = foo()
        self.assertIsNotNone(x)
        assert_type(x, int)
"#,
);

testcase!(
    test_unittest_assert_isinstance,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_is_instance(self) -> None:
        x = foo()
        self.assertIsInstance(x, int)
        assert_type(x, int)

    def test_is_not_instance(self) -> None:
        x = foo()
        self.assertNotIsInstance(x, int)
        assert_type(x, None)
"#,
);

testcase!(
    test_unittest_assert_equal,
    r#"
from typing import assert_type, Literal
from unittest import TestCase
def foo() -> Literal[0, 1]: ...
class MyTest(TestCase):
    def test_equal(self) -> None:
        x = foo()
        self.assertEqual(x, 0)
        assert_type(x, Literal[0])

    def test_not_equal(self) -> None:
        x = foo()
        self.assertNotEqual(x, 0)
        assert_type(x, Literal[1])
"#,
);

testcase!(
    test_unittest_assert_is,
    r#"
from typing import assert_type, Literal
from unittest import TestCase
def foo() -> bool: ...
class MyTest(TestCase):
    def test_is(self) -> None:
        x = foo()
        self.assertIs(x, True)
        assert_type(x, Literal[True])

    def test_is_not(self) -> None:
        x = foo()
        self.assertIsNot(x, True)
        assert_type(x, Literal[False])
"#,
);

testcase!(
    test_unittest_assert_in,
    r#"
from typing import assert_type, Literal
from unittest import TestCase
def foo() -> Literal[1, 2, 3]: ...
class MyTest(TestCase):
    def test_in(self) -> None:
        x = foo()
        self.assertIn(x, [1, 2])
        assert_type(x, Literal[1, 2])

    def test_not_in(self) -> None:
        x = foo()
        self.assertNotIn(x, [1, 2])
        assert_type(x, Literal[3])
"#,
);

// Make sure we catch illegal arguments to isinstance and issubclass even when we aren't narrowing.
testcase!(
    test_validate_class_object_no_narrow,
    r#"
def f(x):
    return isinstance(x, list[int])  # E: Expected class object
def g(x):
    return issubclass(x, list[int])  # E: Expected class object
    "#,
);

testcase!(
    test_isinstance_type_typevar,
    r#"
from typing import assert_type
def f[T](x, y: type[T]) -> T:
    if isinstance(x, y):
        return x
    raise ValueError()
    "#,
);

testcase!(
    test_isinstance_type_self,
    r#"
from typing import Self, TypeGuard
class A:
    def f(self, x) -> TypeGuard[Self]:
        return isinstance(x, type(self))
    "#,
);

testcase!(
    test_or_negation,
    r#"
from typing import assert_type
def f(x: int | None, y: int | None):
    if x is None or y is None:
        pass
    else:
        assert_type(x, int)
        assert_type(y, int)
"#,
);

testcase!(
    test_narrow_to_anonymous_intersection,
    r#"
from typing import reveal_type
class A: pass
class B: pass
class C(A, B): pass  # not used, but demonstrates why the narrow is not Never
def f(x: A):
    if isinstance(x, B):
        reveal_type(x)  # E: A & B
"#,
);

testcase!(
    test_narrow_to_anonymous_intersection2,
    r#"
from typing import assert_type
class A:
    x: int
class B:
    y: str
def f(x: A) -> A:
    assert isinstance(x, B)
    assert_type(x.x, int)
    assert_type(x.y, str)
    return x
    "#,
);

testcase!(
    test_keep_anonymous_intersection_after_flow_merge,
    r#"
from typing import reveal_type
class A: ...
class B: ...
def f(a: A):
    assert isinstance(a, B)
    if hasattr(a, "value") and a.value is None:
        raise ValueError()
    reveal_type(a) # E: A & B
    "#,
);

testcase!(
    test_anonymous_intersection_with_union,
    r#"
class A: ...
class B: ...
class C: ...
def f(a: A | B):
    assert isinstance(a, C)
    g(a)
def g(a: A | B): ...
    "#,
);

testcase!(
    test_typed_dict_and_dict,
    r#"
from typing import TypedDict
class A(TypedDict):
    x: int
def f(a: A | list[int]):
    if isinstance(a, dict):
        return a["x"]
    "#,
);

testcase!(
    test_nested_or_with_multiple_vars,
    r#"
from typing import assert_type
class Foo: pass
class Bar(Foo): pass
def f(x: object, y: object) -> None:
    if isinstance(x, Bar) or (isinstance(y, Foo) and isinstance(x, Foo)):
        assert_type(x, Bar | Foo)
        assert_type(y, Foo | object)
    else:
        assert_type(x, object)
        assert_type(y, object)
"#,
);

testcase!(
    test_narrow_in,
    r#"
from typing import Literal, assert_type, Never
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
def test(x: Literal["foo", 1] | Color | bool | None, y: object, z: Literal["f", "g"]) -> None:
    if x in (1, Color.RED, True):
        assert_type(x, Literal[1, Color.RED, True])
    else:
        assert_type(x, Literal["foo", Color.BLUE, Color.GREEN, False] | None)
    if x in [1, 2, 3, 4]:
        assert_type(x, Literal[1])
    if x in [2, 3, 4]:
        assert_type(x, Never)
    if x in [y, 1]:
        # we only narrow if the list only contains literals
        assert_type(x, Literal["foo", 1, Color.BLUE, Color.GREEN, Color.RED] | bool | None)
    if z in "foo":
        # we only narrow if the RHS is a list, set, tuple literal
        assert_type(z, Literal["f", "g"])
    if y in {1, Color.RED, True}:
        assert_type(y, Literal[1, Color.RED, True])
    else:
        assert_type(y, object)

def test_type_objects(x: type[object]) -> None:
    if x in (int, float):
        assert_type(x, type[int] | type[float])

def test_type_objects_not_in(x: type[int] | type[float] | type[str]) -> None:
    if x not in (int, float):
        assert_type(x, type[int] | type[float] | type[str])
    else:
        assert_type(x, type[int] | type[float])

def test_type_objects_in_union(x: type[int] | type[float] | type[str]) -> None:
    if x in (int, float):
        assert_type(x, type[int] | type[float])
    else:
        assert_type(x, type[int] | type[float] | type[str])

def test_type_objects_mixed_with_literals(x: type[int] | type[float] | None, y: Literal[1] | type[int] | type[str]) -> None:
    if x in (int, None):
        assert_type(x, type[int] | None)
    else:
        assert_type(x, type[int] | type[float])
    if y in (1, int):
        assert_type(y, Literal[1] | type[int])
    else:
        assert_type(y, type[int] | type[str])
"#,
);

testcase!(
    test_narrow_in_with_starred,
    r#"
from typing import Literal, assert_type

def test(x: Literal["a", "b", "c", "d"]) -> None:
    y = ["a", "b"]
    # Starred expressions in `in` checks should not cause type errors,
    # and should not narrow (since we can't know all values at compile time)
    if x in [*y, "c"]:
        assert_type(x, Literal["a", "b", "c", "d"])
    if x not in [*y, "c"]:
        assert_type(x, Literal["a", "b", "c", "d"])

    # Also test in ternary expression
    z = "yes" if x in [*y, "c"] else "no"
    assert_type(z, Literal["yes", "no"])
"#,
);

testcase!(
    test_narrow_len,
    r#"
from typing import assert_type, Never, NamedTuple
class NT(NamedTuple):
    x: int
    y: int
def test(x: tuple[int, int], y: tuple[int, *tuple[int, ...], int], z: tuple[int, ...], nt: NT) -> None:
    if len(x) == 2:
        assert_type(x, tuple[int, int])
    else:
        assert_type(x, Never)
    if len(x) == 1:
        assert_type(x, Never)
    else:
        assert_type(x, tuple[int, int])
    if len(x) != 1:
        assert_type(x, tuple[int, int])
    if len(x) == x[0]:
        # only narrow if RHS is a literal
        assert_type(x, tuple[int, int])
    if len(y) == 2:
        assert_type(y, tuple[int, int])
    else:
        assert_type(y, tuple[int, *tuple[int, ...], int])
    if len(y) == 3:
        assert_type(y, tuple[int, int, int])
    if len(y) == 1:
        # this can never be true, since y has 2 concrete elements in the prefix/suffix
        assert_type(y, Never)
    if len(z) == 1:
        assert_type(z, tuple[int])
    else:
        assert_type(z, tuple[int, ...])
    if len(z) == 3:
        assert_type(z, tuple[int, int, int])
    if len(z) == 2 or len(z) == 3:
        assert_type(z, tuple[int, int] | tuple[int, int, int])
    if len(nt) == 2:
        assert_type(nt, NT)
    else:
        assert_type(nt, Never)
    if len(nt) == 1:
        assert_type(nt, Never)
    else:
        assert_type(nt, NT)
    u: tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...] = tuple(x)
    if len(u) > 1:
        assert_type(u, tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...])
    else:
        assert_type(u, tuple[int, ...])
    if len(u) >= 1:
        assert_type(u, tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...])
    else:
        assert_type(u, tuple[int, ...])
    if len(u) >= 0:
        assert_type(u, tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...])
    else:
        assert_type(u, Never)
    if len(u) > 0:
        assert_type(u, tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...])
    else:
        assert_type(u, tuple[int, ...])
    if len(u) < 1:
        assert_type(u, tuple[int, ...])
    else:
        assert_type(u, tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...])
    if len(u) <= 1:
        assert_type(u, tuple[int, ...])
    else:
        assert_type(u, tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...])
    if len(u) <= 0:
        assert_type(u, tuple[int, ...])
    else:
        assert_type(u, tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...])
    if len(u) < 0:
        assert_type(u, Never)
    else:
        assert_type(u, tuple[int, int] | tuple[int, *tuple[int, ...], int] | tuple[int, ...])
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1616
testcase!(
    test_dict_literal_key_isinstance_narrowing,
    r#"
from typing import Literal, reveal_type
def get_value(x: dict[Literal["value"], int] | int) -> int | None:
    if isinstance(x, dict):
        return x.get("value")
    reveal_type(x) # E: revealed type: int
    return x
    "#,
);

testcase!(
    test_isinstance_loop,
    r#"
from typing import assert_type, Any

def f(x: Any, ty: type[str], xs: Any):
    for _ in xs:
        if isinstance(x, ty):
            assert_type(x, str)
"#,
);

testcase!(
    test_isinstance_tuple_object,
    r#"
from typing import assert_type

def f(expr):
    tys = {str: "test", int: 1}
    if isinstance(expr, tuple(tys.keys())):
        assert_type(expr, int | str)
"#,
);

testcase!(
    test_issubclass_unknown_type,
    r#"
def f(a_type, handlers, type2):
    for _ in handlers:
        if issubclass(a_type, type2):
            pass
    "#,
);

testcase!(
    test_isinstance_uniontype,
    r#"
import types

def f(x, y: types.UnionType):
    isinstance(x, y)
"#,
);

testcase!(
    test_isinstance_selftype,
    r#"
from typing import reveal_type

class Foo:
    @classmethod
    def foo(cls, x):
        if isinstance(x, cls):
            reveal_type(x) # E: revealed type: Self@Foo
"#,
);

testcase!(
    test_isinstance_unpacked_tuple,
    r#"
from typing import assert_type

def f(x, y):
    isinstance(x, (str, *y))

def g(x, y: tuple[type[int], type[str]]):
    if isinstance(x, (bool, *y)):
        assert_type(x, bool | int | str)
"#,
);

testcase!(
    test_typeguard_argument_number,
    r#"
from typing import TypeGuard

def guard_no_arg() -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
    return True

def guard_one_arg(x) -> TypeGuard[int]:
    return True

def guard_two_args(x, y) -> TypeGuard[int]:
    return True

def guard_kw_arg(*, x) -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
    return True

class C:
    def guard_no_arg(self) -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
        return True

    def guard_one_arg(self, x) -> TypeGuard[int]:
        return True

    def guard_two_args(self, x, y) -> TypeGuard[int]:
        return True

    def guard_kw_arg(self, *, x) -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
        return True

    @classmethod
    def guard_no_arg_cls(cls) -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
        return True

    @classmethod
    def guard_one_arg_cls(cls, x) -> TypeGuard[int]:
        return True

    @classmethod
    def guard_two_args_cls(cls, x, y) -> TypeGuard[int]:
        return True

    @classmethod
    def guard_kw_arg_cls(cls, *, x) -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
        return True

    @staticmethod
    def guard_no_arg_static() -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
        return True

    @staticmethod
    def guard_one_arg_static(x) -> TypeGuard[int]:
        return True

    @staticmethod
    def guard_two_args_static(x, y) -> TypeGuard[int]:
        return True

    @staticmethod
    def guard_kw_arg_static(*, x) -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
        return True

class D:
    def guard_missing_self() -> TypeGuard[int]: # E: Type guard functions must accept at least one positional argument
        return True
"#,
);

testcase!(
    test_typeis_argument_number,
    r#"
from typing import TypeIs

def guard_no_arg() -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
    return True

def guard_one_arg(x) -> TypeIs[int]:
    return True

def guard_two_args(x, y) -> TypeIs[int]:
    return True

def guard_kw_arg(*, x) -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
    return True

class C:
    def guard_no_arg(self) -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
        return True

    def guard_one_arg(self, x) -> TypeIs[int]:
        return True

    def guard_two_args(self, x, y) -> TypeIs[int]:
        return True

    def guard_kw_arg(self, *, x) -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
        return True

    @classmethod
    def guard_no_arg_cls(cls) -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
        return True

    @classmethod
    def guard_one_arg_cls(cls, x) -> TypeIs[int]:
        return True

    @classmethod
    def guard_two_args_cls(cls, x, y) -> TypeIs[int]:
        return True

    @classmethod
    def guard_kw_arg_cls(cls, *, x) -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
        return True

    @staticmethod
    def guard_no_arg_static() -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
        return True

    @staticmethod
    def guard_one_arg_static(x) -> TypeIs[int]:
        return True

    @staticmethod
    def guard_two_args_static(x, y) -> TypeIs[int]:
        return True

    @staticmethod
    def guard_kw_arg_static(*, x) -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
        return True

class D:
    def guard_missing_self() -> TypeIs[int]: # E: Type guard functions must accept at least one positional argument
        return True
"#,
);

testcase!(
    test_typeis_subtyping,
    r#"
from typing import TypeIs

def bad_typeis(x: str) -> TypeIs[int]: # E: Return type `int` must be assignable to the first argument type `str`
    return isinstance(x, int)

# From the conformance tests
def also_bad_typeis(x: list[object]) -> TypeIs[list[int]]: # E: Return type `list[int]` must be assignable to the first argument type `list[object]`
    return all(isinstance(i, int) for i in x)

class C:
    def is_int(self, x: str) -> TypeIs[int]: # E: Return type `int` must be assignable to the first argument type `str`
        return isinstance(x, int)

    @classmethod
    def is_int_cls(cls, x: str) -> TypeIs[int]: # E: Return type `int` must be assignable to the first argument type `str`
        return isinstance(x, int)

    @staticmethod
    def is_int_static(x: str) -> TypeIs[int]: # E: Return type `int` must be assignable to the first argument type `str`
        return isinstance(x, int)
"#,
);

testcase!(
    test_while_try_except,
    r#"
from typing import assert_type
class Test:
    x: dict[str, str] | None
    def test(self) -> None:
        assert self.x is not None
        while True:
            try:
                assert_type(self.x, dict[str, str])
                x = self.x.get("asdf")
            except:
                pass
    "#,
);

testcase!(
    test_discriminated_union_key,
    r#"
from typing import TypedDict, assert_type, Literal

class UserDict(TypedDict):
    kind: Literal["user"]
    is_admin: Literal[False]

class AdminDict(TypedDict):
    kind: Literal["admin"]
    is_admin: Literal[True]

def test(x: UserDict | AdminDict):
    if x["kind"] == "user":
        assert_type(x, UserDict)
    else:
        assert_type(x, AdminDict)
    if x["is_admin"] is True:
        assert_type(x, AdminDict)
    else:
        assert_type(x, UserDict)
    "#,
);

testcase!(
    test_discriminated_union_attr,
    r#"
from typing import assert_type, Literal

class User:
    kind: Literal["user"]
    is_admin: Literal[False]

class Admin:
    kind: Literal["admin"]
    is_admin: Literal[True]

def test(x: User | Admin):
    if x.kind == "user":
        assert_type(x, User)
    elif x.kind == "admin":
        assert_type(x, Admin)

    if x.is_admin is True:
        assert_type(x, Admin)
    else:
        assert_type(x, User)
    "#,
);

testcase!(
    test_discriminated_union_index,
    r#"
from typing import assert_type, Literal

def test(x: tuple[Literal["user"], Literal[None]] | tuple[Literal["admin"], int]):
    if x[0] == "user":
        assert_type(x, tuple[Literal["user"], Literal[None]])
    else:
        assert_type(x, tuple[Literal["admin"], int])

    if x[1] is None:
        assert_type(x, tuple[Literal["user"], Literal[None]])
    else:
        assert_type(x, tuple[Literal["admin"], int])

    if x[1] is not None:
        assert_type(x, tuple[Literal["admin"], int])
    else:
        assert_type(x, tuple[Literal["user"], Literal[None]])
    "#,
);

testcase!(
    test_narrow_and_placeholder,
    r#"
from typing import assert_type

class A: pass
class B: pass

def test1(flag: bool, x: A | B) -> None:
    if isinstance(x, A) and flag:
        pass
    else:
        assert_type(x, A | B)

def test2(flag: bool, x: A | B) -> None:
    if flag and isinstance(x, A):
        pass
    else:
        assert_type(x, A | B)

def test3(x: A | B, y: A | B):
    if isinstance(x, A) and isinstance(y, B):
        pass
    else:
        assert_type(x, A | B)

def foo() -> bool: ...
def test4(x: int | None) -> None:
    if x is not None and foo():
        return
    assert_type(x, int | None)
    "#,
);

testcase!(
    test_truthy_falsy_builtins,
    r#"
from typing import assert_type, Literal
def test(a: int, b: str, c: bytes):
    if not a:
        assert_type(a, Literal[0])
    else:
        assert_type(a, int)
    assert_type(a, int)

    if not b:
        assert_type(b, Literal[""])
    else:
        assert_type(b, str)
    assert_type(b, str)

    if not c:
        assert_type(c, Literal[b""])
    else:
        assert_type(c, bytes)
    assert_type(c, bytes)

    if a:
        assert_type(a, int)
    else:
        assert_type(a, Literal[0])

    if b:
        assert_type(b, str)
    else:
        assert_type(b, Literal[""])

    if c:
        assert_type(c, bytes)
    else:
        assert_type(c, Literal[b""])
    "#,
);

testcase!(
    test_do_not_narrow_class_name,
    r#"
assert issubclass(list, object)
x: list[int] = [1]
    "#,
);

testcase!(
    test_disjoint_bases,
    r#"
from typing import assert_never
def f(x: int):
    if isinstance(x, str):
        # `int` and `str` are disjoint bases that cannot be multiply inherited from by the same class.
        assert_never(x)
    "#,
);

testcase!(
    test_literals_are_disjoint,
    r#"
from typing import Literal, LiteralString, assert_never
class A: ...
def f1(a: None):
    if isinstance(a, A):
        assert_never(a)
def f2(a: Literal[1]):
    if isinstance(a, A):
        assert_never(a)
def f3(a: LiteralString):
    if isinstance(a, A):
        assert_never(a)
    "#,
);

testcase!(
    test_callable,
    r#"
from typing import Callable
def f(x: int | Callable[[], int]) -> int:
    if callable(x):
        return x()
    else:
        return x
    "#,
);

testcase!(
    test_isinstance_local_var,
    r#"
from typing import assert_type, Literal
def f(x: int | str):
    isint = isinstance(x, int)
    if isint:
        assert_type(x, int)
        assert_type(isint, Literal[True])
    else:
        assert_type(x, str)
        assert_type(isint, Literal[False])
    "#,
);

testcase!(
    test_truthy_local_var,
    r#"
from typing import assert_type
def f(x: int | None):
    y = x
    if y:
        assert_type(x, int)
        assert_type(y, int)
    "#,
);

testcase!(
    test_reuse_local_var,
    r#"
from typing import assert_type
def f(x: int | str):
    isint = isinstance(x, int)
    if isint:
        assert_type(x, int)
    if isint:
        assert_type(x, int)
    "#,
);

testcase!(
    test_local_var_in_complex_expression,
    r#"
from typing import assert_type, Literal
def f(x: int | str, y: int | str, z: int | str):
    x_or_y_is_int = isinstance(x, int) or isinstance(y, int)
    if not x_or_y_is_int and isinstance(z, str):
        assert_type(x, str)
        assert_type(y, str)
        assert_type(z, str)
        assert_type(x_or_y_is_int, Literal[False])
    "#,
);

testcase!(
    test_circular_reference_to_name,
    r#"
from typing import assert_type, Literal
def f(x, y: bool, force: bool = True):
    force = force or y
    if not force and x:
        assert_type(force, Literal[False])
    "#,
);

testcase!(
    test_local_var_redefinition,
    r#"
from typing import assert_type
def f1(x: int | str):
    check = isinstance(x, int)
    check = isinstance(x, str)
    if check:
        assert_type(x, str)
def f2(x: int | str, y: int):
    check = isinstance(x, int)
    check += y  # truthiness is unknown now
    if check:
        assert_type(x, int | str)
    "#,
);

testcase!(
    test_change_expression_after_test,
    r#"
from typing import assert_type, Literal
def f1(x: int | str, y: int | str):
    isint = isinstance(x, int)
    x = y
    if isint:
        assert_type(x, int | str)
        assert_type(isint, Literal[True])
def f2(x: dict[str, int]):
    val = x.get("k")
    del x["k"]
    if val:
        assert_type(x.get("k"), int | None)
    "#,
);

// Regression test for a case that used to crash pyrefly due to duplicate narrow ranges
testcase!(
    test_redundant_elif,
    r#"
class A:
    x: int | None
def f(a: A, common: list[int]) -> None:
    prefer_device_type = a.x
    if prefer_device_type is not None:
        common_has_preferred = prefer_device_type in common
        if not common_has_preferred:
            return
        elif common_has_preferred:
            return
    "#,
);

// Regression test for a case that used to crash pyrefly due to duplicate narrow ranges
testcase!(
    test_nested_if,
    r#"
def f(other):
    f_other = isinstance(other, (float, str))
    if f_other:
        if not f_other:
            other = 3.14
    return other
    "#,
);

testcase!(
    test_property,
    r#"
class A:
    x: int
class B:
    @property
    def a(self) -> A | None: ...
def f(b: B) -> int:
    a = b.a
    return a.x if a else 0
    "#,
);

testcase!(
    test_dict_get,
    r#"
def f(x: dict[str, int]) -> int:
    v = x.get("k")
    if v:
        return v
    else:
        return 0
    "#,
);

testcase!(
    test_chained_isinstance,
    r#"
from typing import reveal_type
class A: ...
class B: ...
class C: ...
def f(x: A):
    if isinstance(x, B) and isinstance(x, C):
        reveal_type(x) # E: A & B & C
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1642
testcase!(
    test_typed_dict_truthiness_narrowing,
    r#"
from typing import TypedDict, assert_type, NotRequired
class RequiredDict(TypedDict):
    val: int
class EmptyDict(TypedDict):
    val: NotRequired[int]
def test_narrowing(x: RequiredDict | None, y: EmptyDict | None):
    xval = x and x['val']
    assert_type(xval, int | None)
    yval = y and y.get('val')
    assert_type(yval, int | None | EmptyDict)
"#,
);

testcase!(
    test_typevar_intersection,
    r#"
from collections.abc import Mapping, Sequence

def identity_on_mapping[M: Mapping](m: M) -> M: return m

def test_isinstance[T: Mapping[str, int] | Sequence[int]](arg: T) -> T:
    if isinstance(arg, Mapping):
        return identity_on_mapping(arg)
    return arg
    "#,
);

testcase!(
    test_match_intersection_against_constrained_typevar,
    r#"
class A: ...
class B: ...
class C: ...

def f[T: (A, B)](x: T) -> T:
    return x

def g(x: C):
    if isinstance(x, A):
        f(x)
    "#,
);

testcase!(
    test_len_gt_empty_string,
    r#"
def test(unknown):
    s = ""
    if unknown:
        s = unknown.foo
    if len(s) > 0:
      s[0]
    "#,
);

testcase!(
    test_typeis_narrow_to_intersection_not_never,
    r#"
from typing import TypeIs, reveal_type

class A: ...

def f(x: object) -> TypeIs[A]:
    return True

class B: ...

def g(b: B):
    if f(b):
        reveal_type(b)  # E: A & B
    "#,
);

testcase!(
    test_typeguard_return_without_annotation,
    r#"
from typing import TypeGuard

def is_int(x: int | str) -> TypeGuard[int]:
    return isinstance(x, int)

class X:
    def __init__(self, param: int | str) -> None:
        self.param = param

    # This function returns a TypeGuard value but does not have a TypeGuard annotation,
    # so it should not be validated as a TypeGuard function.
    # No "Type guard functions must accept at least one positional argument" error expected.
    def has_int(self):
        return is_int(self.param)
    "#,
);

testcase!(
    test_typeis_return_without_annotation,
    r#"
from typing import TypeIs

def is_int(x: int | str) -> TypeIs[int]:
    return isinstance(x, int)

class X:
    def __init__(self, param: int | str) -> None:
        self.param = param

    # This function returns a TypeIs value but does not have a TypeIs annotation,
    # so it should not be validated as a TypeIs function.
    # No "Type guard functions must accept at least one positional argument" error expected.
    def has_int(self):
        return is_int(self.param)
    "#,
);

testcase!(
    test_typeis_return_type,
    r#"
from typing import TypeIs, assert_type

def is_bool(x: int) -> TypeIs[bool]:
    return isinstance(x, bool)

assert_type(is_bool(0), bool)
    "#,
);

testcase!(
    test_typeguard_return_type,
    r#"
from typing import TypeGuard, assert_type

def is_str(x: object) -> TypeGuard[str]:
    return isinstance(x, str)

assert_type(is_str("hello"), bool)
    "#,
);

testcase!(
    test_typeguard_bad_specialization_no_duplicate,
    r#"
from typing import TypeGuard

def f[T: str](x: T) -> TypeGuard[T]:
    return True

def g(x: int | str):
    if f(x):  # E: `int | str` is not assignable to upper bound `str` of type variable `T`
        pass
    "#,
);

testcase!(
    test_isinstance_invalid_special_form,
    r#"
from typing import Final

def f(x: object):
    isinstance(x, Final)  # E: Expected class object, got special form `Final`
    "#,
);

testcase!(
    test_isinstance_valid_special_form,
    r#"
from typing import Protocol

def f(x: object):
    if isinstance(x, Protocol):
        pass  # No error - Protocol is valid for isinstance
    "#,
);

testcase!(
    test_narrow_to_unknown_name,
    r#"
class C:
    # expected error, leading to Unknown type
    x: XXX | None  # E: Could not find name `XXX`

def f(o: C):
    if o.x is not None:
        o.x.foo
    "#,
);

testcase!(
    test_narrow_to_intersection_of_mapping_and_iterable,
    r#"
from collections.abc import Iterable, Mapping
from typing import Any, assert_type

def test1(arg: Mapping[str, int] | Iterable[tuple[str, int]]) -> None:
    if isinstance(arg, Mapping):
        assert_type(arg, Mapping[str, int] | Mapping[tuple[str, int], Any])
    else:
        assert_type(arg, Iterable[tuple[str, int]])

def test2(arg: Mapping[str, int] | Iterable[tuple[str, int]]) -> None:
    if not isinstance(arg, Mapping):
        assert_type(arg, Iterable[tuple[str, int]])
    else:
        assert_type(arg, Mapping[str, int] | Mapping[tuple[str, int], Any])
    "#,
);

testcase!(
    test_narrow_sequence_to_tuple,
    r#"
from typing import Any, Sequence, assert_type

def f(inputs: Sequence[int]):
    assert isinstance(inputs, tuple)
    for x in inputs:
        assert_type(x, int)
    "#,
);
