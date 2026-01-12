/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_type_var_tuple_basic,
    r#"
from typing import TypeVarTuple, Generic, Any, assert_type
Ts = TypeVarTuple('Ts')
class Array(Generic[*Ts]): ...
def foo(*args: *Ts):
    assert_type(args, tuple[*Ts])
    x: tuple[Any, ...] = args
"#,
);

testcase!(
    test_type_var_tuple_multiple,
    r#"
from typing import TypeVarTuple, Generic, TypeAlias
Ts1 = TypeVarTuple('Ts1')
Ts2 = TypeVarTuple('Ts2')
class ArrayTwoParams(Generic[*Ts1, *Ts2]): ...  # E: Type parameters for class may not have more than one TypeVarTuple
class ArrayTwoParams2[*Ts1, *Ts2](): ...  # E: Type parameters for class may not have more than one TypeVarTuple
TA: TypeAlias = tuple[*Ts1] | tuple[*Ts2]  # E: Type parameters for type alias may not have more than one TypeVarTuple
def foo(t1: tuple[*Ts1], t2: tuple[*Ts2]): ...  # ok
"#,
);

testcase!(
    test_illegal_unpack,
    r#"
from typing import Unpack
x: Unpack[int] = 1  # E: `Unpack` is not allowed in this context
class X(Unpack[int]): ...  # E: `Unpack` is not allowed in this context  # E: Invalid base class: `Unpack`
y: tuple[Unpack[tuple[int, str]]] = (1, "2")  # OK
"#,
);

testcase!(
    bug = "We should disallow star-unpacking in invalid contexts",
    test_invalid_star,
    r#"
from typing import TypeVarTuple, Generic
Ts = TypeVarTuple('Ts')
*Ts
"#,
);

testcase!(
    test_type_var_tuple_class_field_and_constructor,
    r#"
class C1[T]:
    x: tuple[T, ...]
    def __init__(self, x: tuple[T, ...]) -> None:
        self.x = x
        self.y: T = x  # E: `tuple[T, ...]` is not assignable to attribute `y` with type `T`
class C2[*Ts]:
    x: tuple[*Ts]
    def __init__(self, x: tuple[*Ts]) -> None:
        self.x = x
        self.y: tuple[*Ts] = x
"#,
);

testcase!(
    test_require_unpack,
    r#"
from typing import TypeVarTuple, Unpack, Generic
class A[*Ts]: ...
class B[*Ts]:
    def test1(self) -> A[Ts]: ...  # E: `TypeVarTuple` must be unpacked
    def test2(self) -> A[*Ts]: ...
    def test3(self) -> A[Unpack[Ts]]: ...
    def test4(self) -> tuple[Ts]: ...  # E: `TypeVarTuple` must be unpacked
    def test5(self) -> tuple[*Ts]: ...
    def test6(self) -> tuple[Unpack[Ts]]: ...
Ts = TypeVarTuple('Ts')
class C(Ts): ...  # E: `TypeVarTuple` is not allowed in this context  # E: Invalid base class: `TypeVarTuple[Ts]`
class D(Generic[Ts]): ...  # E: `TypeVarTuple` must be unpacked
"#,
);

testcase!(
    test_type_var_tuple_instantiation,
    r#"
from typing import assert_type
class A[*Ts]:
    def x(self) -> tuple[*Ts]:
        raise Exception()
class B[T, *Ts]:
    def x(self) -> tuple[*Ts, T]:
        raise Exception()
def test(a1: A[int], a2: A[int, str], b: B[int, str, int]):
    assert_type(a1.x(), tuple[int])
    assert_type(a2.x(), tuple[int, str])
    assert_type(b.x(), tuple[str, int, int])
"#,
);

testcase!(
    test_type_var_tuple_solve,
    r#"
from typing import assert_type
class A[*Ts]:
    def x(self) -> tuple[*Ts]: ...
def test[*Ts](x: tuple[*Ts]) -> tuple[*Ts]:
    return x
assert_type(test((1, 2, 3)), tuple[int, int, int])
"#,
);

testcase!(
    test_type_var_tuple_subtype,
    r#"
from typing import assert_type
class A[*Ts]:
    def x(self) -> tuple[*Ts]:
        raise Exception()
def helper(x: A[int, str]): ...
def test[*Ts](x: A[int, str], y: A[str, str, str], z: A[*Ts]):
    helper(x)
    helper(y)  # E: Argument `A[str, str, str]` is not assignable to parameter `x` with type `A[int, str]`
    helper(z)  # E: Argument `A[*Ts]` is not assignable to parameter `x` with type `A[int, str]`
"#,
);

testcase!(
    test_type_var_tuple_iterate,
    r#"
from typing import TypeVarTuple

_Ts = TypeVarTuple("_Ts")
def f(*args: *_Ts) -> None:
    f(*args)
"#,
);

testcase!(
    test_type_var_tuple_legacy_unpack,
    r#"
from typing import Unpack, TypeVarTuple

_Ts = TypeVarTuple("_Ts")
class A:
    def f(self, *args: Unpack[_Ts]): ...
"#,
);

testcase!(
    test_type_var_tuple_unpack_quantified,
    r#"
from typing import Callable

def test[*Ts, T](f: Callable[[*Ts], T], t: tuple[*Ts], *args: *Ts):
    # we can unpack a quantified type var tuple if it matches the expected type exactly
    x: tuple[*Ts] = (*args,)
    f(*args)
    x = t
    x = (*t,)

    # This error message could be improved
    x = (*args, *args)  # E: `tuple[ElementOf[Ts] | Unknown, ...]` is not assignable to variable `x` with type `tuple[*Ts]`
    x = (*args, 1)  # E: `tuple[*Ts, Literal[1]]` is not assignable to variable `x` with type `tuple[*Ts]`
    x = (1, *args)  # E: `tuple[Literal[1], *Ts]` is not assignable to variable `x` with type `tuple[*Ts]`
    x = (*t, *t)  # E: `tuple[ElementOf[Ts] | Unknown, ...]` is not assignable to variable `x` with type `tuple[*Ts]`
    x = (*t, 1)  # E: `tuple[*Ts, Literal[1]]` is not assignable to variable `x` with type `tuple[*Ts]`
    x = (1, *t)  # E: `tuple[Literal[1], *Ts]` is not assignable to variable `x` with type `tuple[*Ts]`
    f(*args, *args)  # E: Expected at most one unpacked variadic argument
    f(1, *args)  # E: Unpacked argument `tuple[Literal[1], *Ts]` is not assignable to varargs type `tuple[*Ts]`
    f(*args, 1)  # E: Unpacked argument `tuple[*Ts, Literal[1]]` is not assignable to varargs type `tuple[*Ts]`
    f(*t, *t)  # E: Expected at most one unpacked variadic argument
    f(1, *t)  # E: Unpacked argument `tuple[Literal[1], *Ts]` is not assignable to varargs type `tuple[*Ts]`
    f(*t, 1)  # E: Unpacked argument `tuple[*Ts, Literal[1]]` is not assignable to varargs type `tuple[*Ts]`
"#,
);

testcase!(
    test_type_var_tuple_callable_resolves_to_empty,
    r#"
from typing import Callable, assert_type

def test[*Ts, T](f: Callable[[*Ts], T], *args: *Ts) -> tuple[*Ts]:
    x: T = f(*args)
    return (*args,)

assert_type(test(lambda: 1), tuple[()])

fun: Callable[[int], int] = lambda x: x + 1
assert_type(test(fun, 1), tuple[int])
"#,
);

testcase!(
    test_type_var_tuple_resolves_to_empty,
    r#"
from typing import Callable, assert_type

def test[*Ts](*args: *Ts) -> tuple[*Ts]:
    return (*args,)

assert_type(test(), tuple[()])
assert_type(test(1), tuple[int])
"#,
);

testcase!(
    test_type_var_tuple_slice_empty,
    r#"
from typing import assert_type

def f[*Ts](*x: *Ts) -> None:
    assert_type(x[1:0], tuple[()])
"#,
);

testcase!(
    test_type_var_tuple_bound_method_resolves_to_empty,
    r#"
from collections.abc import Callable, Awaitable
from typing import TypeVarTuple

class Nursery:
    def start_soon[*PosArgsT](self, func: Callable[[*PosArgsT], Awaitable[None]], *args: *PosArgsT) -> None:
        pass

class B:
    async def bound(self) -> None:
        pass

    def fn(self):
        n = Nursery()
        n.start_soon(self.bound)
"#,
);

testcase!(
    test_type_var_tuple_subscript,
    r#"
from typing import assert_type

def test[*Ts](prefix_only: tuple[int, str, *Ts], prefix_and_suffix: tuple[int, str, *Ts, bool, str], suffix_only: tuple[*Ts, bool, str, int]):
    # Positive indexing in prefix
    assert_type(prefix_only[0], int)
    assert_type(prefix_only[1], str)
    assert_type(prefix_and_suffix[0], int)
    assert_type(prefix_and_suffix[1], str)

    # Negative indexing in suffix
    assert_type(suffix_only[-1], int)
    assert_type(suffix_only[-2], str)
    assert_type(suffix_only[-3], bool)
    assert_type(prefix_and_suffix[-1], str)
    assert_type(prefix_and_suffix[-2], bool)

    # Slice within prefix
    assert_type(prefix_only[0:1], tuple[int])
    assert_type(prefix_only[0:2], tuple[int, str])
    assert_type(prefix_only[1:2], tuple[str])
    assert_type(prefix_and_suffix[0:1], tuple[int])
    assert_type(prefix_and_suffix[0:2], tuple[int, str])
    assert_type(prefix_and_suffix[1:2], tuple[str])

    # Slice ending in prefix
    assert_type(prefix_only[:1], tuple[int])
    assert_type(prefix_only[:2], tuple[int, str])
    assert_type(prefix_and_suffix[:1], tuple[int])
    assert_type(prefix_and_suffix[:2], tuple[int, str])

    # Slice starting in prefix
    assert_type(prefix_only[1:], tuple[str, *Ts])
    assert_type(prefix_and_suffix[1:], tuple[str, *Ts, bool, str])
    assert_type(prefix_only[2:], tuple[*Ts])
    assert_type(prefix_and_suffix[2:], tuple[*Ts, bool, str])

    # Slice ending in suffix
    assert_type(prefix_and_suffix[:-1], tuple[int, str, *Ts, bool])
    assert_type(prefix_and_suffix[:-2], tuple[int, str, *Ts])
    assert_type(prefix_and_suffix[1:-1], tuple[str, *Ts, bool])

    # Unhandled cases (these should fall back and not crash)
    prefix_only[5:]
    prefix_and_suffix[5:6]
    prefix_and_suffix[5:6:2]
    suffix_only[10:20]
    prefix_and_suffix[-10:-5]
    prefix_only[::2]
    suffix_only[::-1]
    prefix_only[:-10]
    prefix_only[:10]
    prefix_and_suffix[1:10:3]
    prefix_and_suffix[-1:]
    prefix_only[-5]
    prefix_and_suffix[-5]
    prefix_and_suffix[10]

"#,
);

testcase!(
    test_type_var_tuple_swap,
    r#"
def test[T, *Ts](t1: tuple[T, *Ts], t2: tuple[*Ts, T]):
    t1 = (t2[-1], *t2[:-1])
    t2 = (*t1[1:], t1[0])
"#,
);
