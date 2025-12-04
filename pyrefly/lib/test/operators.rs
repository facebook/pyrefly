/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_binop,
    r#"
from typing import assert_type
def f(a: int, b: int) -> None:
    c = a + b
    assert_type(c, int)
    d = a + 1
    assert_type(d, int)
"#,
);

testcase!(
    test_bounded_type_var_comparison,
    r#"
def compare[T: int](x: T, y: T) -> bool:
    return x > y
"#,
);

testcase!(
    test_negative_literals,
    r#"
from typing import Literal
x: Literal[-1] = -1
"#,
);

testcase!(
    test_positive_literals,
    r#"
from typing import Literal
y = -1
x: Literal[-1] = +y
"#,
);
testcase!(
    test_positive_literals_error,
    r#"
from typing import Literal
y = 1
x: Literal[-1] = +y # E: `Literal[1]` is not assignable to `Literal[-1]`
"#,
);

testcase!(
    test_inversion_literals,
    r#"
from typing import Literal
y = -1
x: Literal[0] = ~y
"#,
);

testcase!(
    test_inversion_literals_error,
    r#"
from typing import Literal
y = -2
x: Literal[0] = ~y # E: `Literal[1]` is not assignable to `Literal[0]`
"#,
);

testcase!(
    test_union_unary_op,
    r#"
from typing import Literal, Union, assert_type
x: Literal[-1, 2] = 2
y = -x
assert_type(y, Literal[-2, 1])
"#,
);

testcase!(
    test_boolean_or_simple,
    r#"
from typing import assert_type
def f(x: int, y: str) -> None:
    z = x or y
    assert_type(z, int | str)
    "#,
);

testcase!(
    test_boolean_or_filter,
    r#"
from typing import assert_type, Literal

x = False or True
assert_type(x, Literal[True])

y = False or None
assert_type(y, None)

def f(a: None, b: int, c: str, cond: bool) -> None:
    if cond:
        b2 = a
    else:
        b2 = b
    d = b2 or c
    assert_type(d, int | str)
    "#,
);

testcase!(
    test_boolean_or_shortcircuit,
    r#"
from typing import assert_type, Literal
x = True or False
assert_type(x, Literal[True])
    "#,
);

testcase!(
    test_integer_or_shortcircuit,
    r#"
from typing import assert_type, Literal
x = 1 or 0
assert_type(x, Literal[1])
    "#,
);

testcase!(
    test_string_or_shortcircuit,
    r#"
from typing import assert_type, Literal
x = "a" or ""
assert_type(x, Literal["a"])
    "#,
);

testcase!(
    test_html_escape_or_str_regression,
    r#"
import html
from typing import assert_type

x = html.escape("") or "a"
assert_type(x, str)
    "#,
);

testcase!(
    test_boolean_and_simple,
    r#"
from typing import assert_type, Literal

def f(x: int, y: str) -> None:
    z = x and y
    assert_type(z, Literal[0] | str)
    "#,
);

testcase!(
    test_boolean_and_filter,
    r#"
from typing import assert_type, Literal
x = True and False
assert_type(x, Literal[False])

y = True and True
assert_type(y, Literal[True])
    "#,
);

testcase!(
    test_boolean_and_shortcircuit,
    r#"
from typing import assert_type, Literal
x = False and True
assert_type(x, Literal[False])
    "#,
);

testcase!(
    test_integer_and_shortcircuit,
    r#"
from typing import assert_type, Literal
x = 0 and 1
assert_type(x, Literal[0])
    "#,
);

testcase!(
    test_string_and_shortcircuit,
    r#"
from typing import assert_type, Literal
x = "" and "a"
assert_type(x, Literal[""])
    "#,
);

testcase!(
    bug = "Should narrow",
    test_boolean_operator_narrow,
    r#"
from typing import assert_type, Literal

def f(x: bool, y: int):
    assert_type(x and y, Literal[False] | int)

def g(x: bool, y: Literal['a'], z: Literal['b']):
    assert_type((x and y) or z, Literal['a', 'b'])

def h(x: int, y: str, z: bool, v: float):
    assert_type(x or y or z or v, int | float | str | Literal[True])
    assert_type(x and y and z and v, Literal[0, "", False] | float)
"#,
);

testcase!(
    test_boolean_union,
    r#"
from typing import assert_type, Literal

def f(x: int, y: str | Literal[False]):
    assert_type(x or y, int | str | Literal[False])
"#,
);

testcase!(
    test_unary_not_unknown,
    r#"
from typing import assert_type
def f(x):
    y = not x
    assert_type(y, bool)
    "#,
);

testcase!(
    test_unary_not_literal,
    r#"
from typing import assert_type, Literal

x1 = True
y1 = not x1
assert_type(y1, Literal[False])

x2 = False
y2 = not x2
assert_type(y2, Literal[True])

x3 = 1
y3 = not x3
assert_type(y3, Literal[False])

x4 = 0
y4 = not x4
assert_type(y4, Literal[True])

x5 = "a"
y5 = not x5
assert_type(y5, Literal[False])

x6 = ""
y6 = not x6
assert_type(y6, Literal[True])
    "#,
);

testcase!(
    test_unary_dunders,
    r#"
from typing import Literal, assert_type
class C:
    def __pos__(self) -> Literal[5]:
        return 5
    def __neg__(self) -> Literal[-5]:
        return -5
    def __invert__(self) -> Literal[100]:
        return 100
c = C()
assert_type(+c, Literal[5])
assert_type(-c, Literal[-5])
assert_type(~c, Literal[100])
    "#,
);

testcase!(
    test_unary_error,
    r#"
+None  # E: Unary `+` is not supported on `None`
+"oh no"  # E: Unary `+` is not supported on `Literal['oh no']`
-"oops"  # E: Unary `-` is not supported on `Literal['oops']`
class A:
    def __invert__(self, extra_arg):
        pass
~A()  # E: Unary `~` is not supported on `A`\n  Missing argument `extra_arg` in function `A.__invert__`
    "#,
);

testcase!(
    test_unary_enum,
    r#"
from enum import Enum
class A(Enum):
    X = 1
class B(Enum):
    X = 1
    def __pos__(self):
        return 0
+A.X  # E: Unary `+` is not supported on `Literal[A.X]`
+B.X  # OK
    "#,
);

testcase!(
    test_operator_error,
    r#"
class C: pass

x = C() + 1  # E:  `+` is not supported between `C` and `Literal[1]`"#,
);

testcase!(
    test_float_int_add,
    r#"
from typing import assert_type
x = 3 + 3.0
assert_type(x, float)
"#,
);

testcase!(
    test_float_int_compare,
    r#"
0 < 1.0
1.0 < 2
1 < 2
1.0 < 3.0
"#,
);

testcase!(
    test_inplace_operator_rhs_union,
    r#"
class A:
    def __radd__(self, other: int) -> int:
        return other
def f(x: int, y: int | A) -> None:
    x += y
    "#,
);

testcase!(
    test_unop_on_any,
    r#"
from typing import Any, assert_type
def f(x: Any):
    assert_type(-x, Any)
    "#,
);

testcase!(
    test_binop_on_any,
    r#"
from typing import Any, assert_type

def f(x: Any):
    assert_type(1 + x, Any)
    assert_type(x < 10, Any)

def f2(x: int | Any):
    assert_type(1 + x, int | Any)
    assert_type(x < 10,  bool | Any )
    "#,
);

testcase!(
    test_binop_type_var,
    r#"
from typing import TypeVar, reveal_type
T_co = TypeVar("T_co", covariant=True)
T_co == int
reveal_type(T_co) # E:  revealed type: TypeVar[T_co]
    "#,
);

testcase!(
    test_comparison_return_type,
    r#"
from typing import Literal, assert_type
class A:
  def __lt__(self, other):
    return 1
assert_type(A() < A(), Literal[1])
    "#,
);

testcase!(
    test_missing_binop_attr,
    r#"
class A:
    pass
class B:
    pass
A() + B()  # E: Cannot find `__add__` or `__radd__`
    "#,
);

// Both __add__ and __radd__ are tried, but it's less confusing to use __add__ when both fail.
testcase!(
    test_binop_error,
    r#"
from typing import Never, assert_type
class A:
    def __add__(self, other: Never) -> "A":
        return self
class B:
    def __radd__(self, other: Never) -> "B":
        return self
a = A() + B()  # E: `B` is not assignable to parameter `other` with type `Never` in function `A.__add__`
assert_type(a, A)
    "#,
);

// We try __iadd__ and some fallback dunders. When all fail, the least confusing option is to use __iadd__.
testcase!(
    test_iadd_error,
    r#"
from typing import Never, assert_type
class A:
    def __iadd__(self, other: Never):
        pass
class B:
    def __radd__(self, other: Never) -> "B":
        return self
a = A()
a += B()  # E: `B` is not assignable to parameter `other` with type `Never` in function `A.__iadd__`
    "#,
);

testcase!(
    test_custom_eq,
    r#"
from typing import assert_type
class A:
    def __eq__(self, other) -> bool:
        return True
assert_type(A() == 42, bool)
    "#,
);

testcase!(
    test_in_generator,
    r#"
'x' in (x for x in ['y'])
42 in (x for x in ['y'])  # E: `in` is not supported between `Literal[42]` and `Generator[str, None, None]`
    "#,
);

testcase!(
    test_unop_on_self,
    r#"
from typing import Self
class C:
    def __neg__(self) -> Self:
        return self
    def f(self) -> Self:
        return -self
    "#,
);

testcase!(
    test_dunder_bool_short_circuit_and_discard,
    r#"
from typing import assert_type, Literal
class Falsey:
    def __bool__(self) -> Literal[False]:
        return False
class Truthy:
    def __bool__(self) -> Literal[True]:
        return True
class NotBoolable:
    __bool__: int = 0

assert_type(Falsey() or int(), int)
# Note that although we evaluate `__bool__` and use the result for the boolean
# operation control flow, the resulting value is the actual `Truthy` instance
# and not a bool. This matches the runtime.
assert_type(Truthy() or int(), Truthy)
assert_type(int() if Truthy() else str(), int)
assert_type(int() if Falsey() else str(), str)

# Test the use of a non-boolean-convertable type in boolean operators.
#
# The runtime only uses truthiness in short-circuiting here, so it is actually
# legal to use a non-boolable value as the rightmost entry of a bool op.
assert_type(NotBoolable() or int(), int | NotBoolable)  # E: Expected `__bool__` to be a callable, got `int`
assert_type(int() or NotBoolable(), int | NotBoolable)
"#,
);

testcase!(
    test_tensor_type_lambda,
    r#"
from typing import Callable, cast, assert_type

class Tensor:
    __pow__ = cast(Callable[[Tensor, int], Tensor], lambda x, y: x)  # No redundant cast warning - types are not exactly equal

def f(x: Tensor, i: int):
    assert_type(x ** i, Tensor)

    "#,
);

testcase!(
    test_tensor_type_method,
    r#"
from typing import assert_type

def power(x: "Tensor", i: int) -> "Tensor":
    return x

class Tensor:
    __pow__ = power

def f(x: Tensor, i: int):
    assert_type(x ** i, Tensor)

    "#,
);

testcase!(
    test_magic_dunder_call_with_metaclass,
    r#"
from typing import assert_type

class Meta(type):
    def __add__(cls, other) -> int:
        return 0

class A(metaclass=Meta):
    @classmethod
    def __add__(cls, other) -> str:
        return ""

class B(A):
    pass

# The `+` operation uses `Meta.__add__`, unlike a direct `__add__` call, which uses `A.__add__`.
assert_type(B + B, int)
assert_type(B.__add__(B), str)
    "#,
);

testcase!(
    test_iter_var_annotation_with_getitem,
    r#"
class A:
    def __getitem__(self, i: int):
        return 0

x: int
for x in A():
    pass

y: str
for y in A():  # E: Cannot use variable `y` with type `str` to iterate over elements of type `Literal[0]`
    pass
    "#,
);

testcase!(
    test_contains_bad_getitem,
    r#"
class A:
    def __getitem__(self):
        return 0
0 in A()  # E: Expected 0 positional arguments, got 1 in function `A.__getitem__`
    "#,
);

testcase!(
    test_contains_getitem_wrong_type,
    r#"
class A:
    def __getitem__(self, i) -> int:
        return 0
"" in A()  # E: `Literal['']` is not assignable to contained type `int`
    "#,
);

testcase!(
    test_deprecated,
    r#"
from typing import assert_type, Self
from warnings import deprecated
class A:
    @deprecated("Super deprecated")
    def __add__(self, other) -> Self:
        return self
class B:
    def __radd__(self, other) -> Self:
        return self
assert_type(A() + B(), A)  # E: `A.__add__` is deprecated
    "#,
);

// https://github.com/facebook/pyrefly/issues/396
testcase!(
    test_bind_dunder_callable_simple,
    r#"
from typing import Callable

def foo_pow(base: "Foo", arg: "int | Foo") -> "Foo": return Foo()

class Foo:
    __pow__: Callable[["Foo", int], "Foo"] = foo_pow

def test(foo: Foo) -> None:
    foo ** 2
    "#,
);

// https://github.com/facebook/pyrefly/issues/396
testcase!(
    test_bind_dunder_callable_2,
    r#"
from typing import *
class ThisClassWorks:
    def __init__(self, flag: bool) -> None:
        self.flag = flag

    def __and__(self, other: Union["ThisClassWorks", bool]) -> "ThisClassWorks":
        if isinstance(other, bool):
            return ThisClassWorks(self.flag and other)
        return ThisClassWorks(self.flag and other.flag)

    def __rand__(self, other: Union["ThisClassWorks", bool]) -> "ThisClassWorks":
        if isinstance(other, bool):
            return ThisClassWorks(self.flag and other)
        return ThisClassWorks(self.flag and other.flag)


def _produce_and_func() -> Callable[
    ["ThisClassDoesNotWork", Union["ThisClassDoesNotWork", bool]],
    "ThisClassDoesNotWork",
]:
    def and_func(
        self: "ThisClassDoesNotWork", other: Union["ThisClassDoesNotWork", bool]
    ) -> "ThisClassDoesNotWork":
        if isinstance(other, bool):
            return ThisClassDoesNotWork(self.flag and other)
        return ThisClassDoesNotWork(self.flag and other.flag)

    return and_func


class ThisClassDoesNotWork:
    def __init__(self, flag: bool) -> None:
        self.flag = flag

    __and__ = _produce_and_func()
    __rand__ = _produce_and_func()

ThisClassWorks(True) & ThisClassWorks(False)
ThisClassWorks(True) & False
True & ThisClassWorks(False)
ThisClassDoesNotWork(True) & ThisClassDoesNotWork(False)
ThisClassDoesNotWork(True) & False
True & ThisClassDoesNotWork(False)
    "#,
);

testcase!(
    test_type_of_typevar_equality,
    r#"
def f[S, T](x: type[S], y: type[T]):
    return x == y
    "#,
);

testcase!(
    test_chained_in,
    r#"
class Foo:
    def __contains__(self, x: int) -> bool:
        ...

class Bar:
    def __contains__(self, x: Foo) -> bool:
        ...

def test(x: int, foo: Foo, bar: Bar) -> None:
    x in foo in bar  # Should be OK
    x in bar # E: `in` is not supported between `int` and `Bar`
    "#,
);

testcase!(
    test_chained_lt,
    r#"
class A:
    def __lt__(self, other: "B") -> bool: ...
class B:
    def __lt__(self, other: "C") -> bool: ...
class C: pass

def test(a: A, b: B, c: C) -> None:
    a < b < c  # Should be OK: (a < b) and (b < c)
    a < c      # E: `<` is not supported between `A` and `C`
    "#,
);
