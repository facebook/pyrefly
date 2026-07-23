/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_tuple,
    r#"
from typing import assert_type, Literal

x = (1, "2")
assert_type(x, tuple[Literal[1], Literal["2"]])

y: tuple[int, Literal["3"]] = (1, "3")
"#,
);

testcase!(
    test_index_literal,
    r#"
from typing import assert_type, Literal

x = (1, "2")
assert_type(x[0], Literal[1])
assert_type(x[1], Literal["2"])
assert_type(x[-2], Literal[1])
assert_type(x[-1], Literal["2"])
"#,
);

testcase!(
    test_invalid_ellipsis,
    r#"
from typing import assert_type, Any
def test(
    x1: tuple[int, ...], # OK
    x2: tuple[...],  # E: Invalid position for `...`
    x3: tuple[int, ..., ...],  # E: Invalid position for `...`
    x4: tuple[int, ..., int],  # E: Invalid position for `...`
    x5: tuple[int, int, ...],  # E: Invalid position for `...`
    x6: tuple[..., int],  # E: Invalid position for `...`
    x7: tuple[*tuple[int], ...]  # E: `...` cannot be used with an unpacked `TypeVarTuple` or tuple
):
    assert_type(x2, tuple[Any, ...])
    assert_type(x3, tuple[Any, ...])
    assert_type(x4, tuple[Any, ...])
    assert_type(x5, tuple[Any, ...])
    assert_type(x6, tuple[Any, ...])
    assert_type(x7, tuple[Any, ...])
"#,
);

testcase!(
    test_index,
    r#"
from typing import assert_type

def foo(x: tuple[int, str], y: tuple[int, ...], z: tuple[int, *tuple[str, ...], bool], idx: int) -> None:
    assert_type(x[idx], int | str)
    assert_type(y[idx], int)
    assert_type(z[idx], bool | int | str)
    x["nonsense"]  # E: Cannot index into `tuple[int, str]`
    y["nonsense"]  # E: Cannot index into `tuple[int, ...]`
"#,
);

testcase!(
    test_empty_tuple,
    r#"
from typing import assert_type
assert_type((), tuple[()])
"#,
);

testcase!(
    test_tuple_base,
    r#"
from typing import Any
class Base1(tuple[Any, ...]): ...
class Base2(tuple[int, ...]): ...
class Base3(tuple[int, int]): ...
class Base4(tuple[str, int]): ...

class Child1(Base1, Base2): ...
class Child2(Base1, Base3): ...
class Child3(Base3, Base4): ...  # E: Class `Child3` has inconsistent type arguments for base class `Iterable`
class Child4(Base2, Base3): ...
"#,
);

testcase!(
    test_tuple_base_subtype,
    r#"
from typing import *
class Size(tuple[int, ...]): ...
def f(x: tuple[int, ...]): ...
def g(x: Size):
    f(x)
"#,
);

testcase!(
    test_tuple_base_narrow,
    r#"
from typing import *
class A(tuple[int, str]): ...
class B(tuple[int, str, bool]): ...
def test(x: A | B):
    if len(x) == 2:
        assert_type(x, A)
    else:
        assert_type(x, B)
"#,
);

testcase!(
    test_tuple_base_index,
    r#"
from typing import *
class A(tuple[int, str]): ...
def test(x: A):
    assert_type(x[0], int)
    assert_type(x[1], str)
"#,
);

testcase!(
    test_unparameterized,
    r#"
from typing import assert_type, Any, Tuple
def foo(x: tuple, y: Tuple) -> None:
    assert_type(x, tuple[Any, ...])
    assert_type(y, tuple[Any, ...])
"#,
);

testcase!(
    test_tuple_type_attr_base,
    r#"
from typing import Any
def is_namedtuple_cls(cls: Any):
    if issubclass(cls, tuple):
        print(cls.__bases__)
"#,
);

testcase!(
    test_tuple_bad_unpack,
    r#"
from typing import Any, Iterable
def f(x: int) -> int: ...
def test(y: int):
    x: tuple[int, ...] = (3, *y, 4)  # E: Expected an iterable, got `int`
    x: tuple[int, ...] = (3, *y, f("x"))  # E: Expected an iterable, got `int`  # E: Argument `Literal['x']` is not assignable to parameter `x` with type `int` in function `f`
"#,
);

testcase!(
    test_unpack_starred_not_iterable_in_tuple,
    r#"
def test():
    x: int = 42
    y = (*x,)  # E: Expected an iterable, got `int`
"#,
);

testcase!(
    test_unpack_index_out_of_bounds,
    r#"
def test(x: tuple[int]) -> None:
  y, z = x  # E: Cannot unpack
"#,
);

testcase!(
    test_unpack_in_literal,
    r#"
from typing import Any, assert_type, Literal
def test(x: tuple[int, ...], y: str) -> None:
  assert_type(("foo", *(1, 1)), tuple[Literal['foo'], Literal[1], Literal[1]])
  assert_type((1, *x, 2), tuple[Literal[1], *tuple[int, ...], Literal[2]])
  assert_type((1, *x, *x, 3), tuple[Literal[1], *tuple[int, ...], Literal[3]])
  assert_type((1, *x, y, *x, 3), tuple[Literal[1], *tuple[int | str, ...], Literal[3]])
"#,
);

testcase!(
    test_unbounded_solve,
    r#"
from typing import Any
def test(x: tuple[int, str], y: tuple[int, ...], z: tuple[Any, ...]) -> None:
  a: tuple[int, int] = z
  b: tuple[int | str, ...] = x
  c: tuple[int | str, ...] = y
  d: tuple[int, ...] = x  # E: `tuple[int, str]` is not assignable to `tuple[int, ...]`
"#,
);

testcase!(
    test_unpacked_solve,
    r#"
from typing import Any
def test(a: tuple[int, bool, str], b: tuple[Any, ...], c: tuple[int, *tuple[bool, ...], str]) -> None:
  x1: tuple[int, *tuple[bool, ...], str] = a
  x2: tuple[int, *tuple[bool | str, ...]] = a
  x3: tuple[*tuple[int | bool, ...], str] = a
  x4: tuple[int, bool, *tuple[str, ...]] = a
  x5: tuple[*tuple[int, ...], bool, str] = a
  x6: tuple[int, *tuple[bool, ...], str] = b
  x7: tuple[int, *tuple[bool, ...], str] = c
  x8: tuple[int, *tuple[bool | str, ...]] = c
  x9: tuple[*tuple[int | bool, ...], str] = c
  x10: tuple[*tuple[int], *tuple[bool], *tuple[str]] = a
  x11: tuple[int, *tuple[bool, str]] = a
  x12: tuple[*tuple[int, bool, str]] = a
  x13: tuple[*tuple[int, ...], *tuple[bool], *tuple[str]] = a
  x14: tuple[*tuple[int, ...], *tuple[bool, ...], *tuple[str]] = a  # E: Only one unbounded type is allowed to be unpacked
"#,
);

testcase!(
    test_slice_literal,
    r#"
from typing import assert_type, Literal

x = (5, 6, 7)

assert_type(x[0:0], tuple[()])
assert_type(x[0:1], tuple[Literal[5]])
assert_type(x[0:2], tuple[Literal[5], Literal[6]])
assert_type(x[0:3], tuple[Literal[5], Literal[6], Literal[7]])

assert_type(x[1:1], tuple[()])
assert_type(x[1:2], tuple[Literal[6]])
assert_type(x[1:3], tuple[Literal[6], Literal[7]])

assert_type(x[2:2], tuple[()])
assert_type(x[2:3], tuple[Literal[7]])

assert_type(x[3:3], tuple[()])

assert_type(x[:0], tuple[()])
assert_type(x[:1], tuple[Literal[5]])
assert_type(x[:2], tuple[Literal[5], Literal[6]])
assert_type(x[:3], tuple[Literal[5], Literal[6], Literal[7]])

assert_type(x[0:], tuple[Literal[5], Literal[6], Literal[7]])
assert_type(x[1:], tuple[Literal[6], Literal[7]])
assert_type(x[2:], tuple[Literal[7]])
assert_type(x[3:], tuple[()])
"#,
);

testcase!(
    test_slice_negative,
    r#"
from typing import assert_type, Literal

x = (5, 6, 7)

# Negative end index
assert_type(x[:-1], tuple[Literal[5], Literal[6]])
assert_type(x[:-2], tuple[Literal[5]])
assert_type(x[:-3], tuple[()])

# Negative start index
assert_type(x[-1:], tuple[Literal[7]])
assert_type(x[-2:], tuple[Literal[6], Literal[7]])
assert_type(x[-3:], tuple[Literal[5], Literal[6], Literal[7]])

# Both negative
assert_type(x[-3:-1], tuple[Literal[5], Literal[6]])
assert_type(x[-2:-1], tuple[Literal[6]])

# Mixed positive and negative
assert_type(x[0:-1], tuple[Literal[5], Literal[6]])
assert_type(x[1:-1], tuple[Literal[6]])
assert_type(x[-2:3], tuple[Literal[6], Literal[7]])
"#,
);

testcase!(
    test_unbounded_tuple_hint,
    r#"
x1: tuple[str, ...] = ("ok",)
x2: tuple[int, ...] = ("err",)  # E: `tuple[Literal['err']]` is not assignable to `tuple[int, ...]`
    "#,
);

testcase!(
    test_superclass_tuple_hint,
    r#"
from typing import Iterable, Literal
x1: Iterable[Literal['ok']] = ("ok",)
x2: Iterable = ("ok",)
x3: object = ("ok",)
x4: Iterable[int] = ("err",)  # E: `tuple[Literal['err']]` is not assignable to `Iterable[int]`
x5: list[int] = ("err",)  # E: `tuple[Literal['err']]` is not assignable to `list[int]`
    "#,
);

testcase!(
    test_empty_tuple_hint,
    r#"
from typing import Iterable
x: Iterable[str] = ()
    "#,
);

testcase!(
    test_unpack_union,
    r#"
from typing import assert_type
def f() -> tuple[int, str] | tuple[bool, ...]: ...
(x, y) = f()
assert_type(x, int | bool)
assert_type(y, str | bool)

(x, y, z) = f()  # E: Cannot unpack
    "#,
);

testcase!(
    test_iterate_union,
    r#"
from typing import assert_type
def f() -> tuple[int, str] | tuple[bool, ...]: ...
for x in f():
    assert_type(x, int | bool | str)
    "#,
);

testcase!(
    test_tuple_parent,
    r#"
from typing import Any, assert_type
class C1(tuple[int, ...]):
    pass
class C2(tuple[int, int]):
    pass
for x in C1():
    assert_type(x, int)
for x in C2():
    assert_type(x, int)
    "#,
);

testcase!(
    test_tuple_short_unpack,
    r#"
*a, b, c = (1,) # E: Cannot unpack tuple[Literal[1]] (of size 1) into 2+ values
"#,
);

testcase!(
    test_tuple_with_never_element_preserves_shape,
    r#"
from typing import Literal, NoReturn, assert_type

def f(x: NoReturn) -> None:
    t = (x, 1)
    assert_type(t[1], Literal[1])
"#,
);

testcase!(
    test_unpacked_tuple_subtype,
    r#"
from typing import Sequence
def test[*Ts](x1: tuple[int, *tuple[str, ...]], x2: tuple[*Ts]) -> None:
    y1: Sequence[int | str] = x1
    y2: tuple[int | str, ...] = x1
    y3: tuple[object, ...] = x2
"#,
);

testcase!(
    test_unpack_typevar_bound_to_tuple,
    r#"
from typing import reveal_type
def f[Z: tuple[str, int]](x: Z):
    u, v = x
    reveal_type(u)  # E: revealed type: str
    reveal_type(v)  # E: revealed type: int
"#,
);

testcase!(
    test_unpack_typevar_bound_to_tuple_three_elements,
    r#"
from typing import reveal_type
def f[Z: tuple[str, int, bytes]](x: Z):
    a, b, c = x
    reveal_type(a)  # E: revealed type: str
    reveal_type(b)  # E: revealed type: int
    reveal_type(c)  # E: revealed type: bytes
"#,
);

testcase!(
    test_unpack_typevar_bound_to_unbounded_tuple,
    r#"
from typing import reveal_type
def f[Z: tuple[int, ...]](x: Z):
    a, b = x
    reveal_type(a)  # E: revealed type: int
    reveal_type(b)  # E: revealed type: int
"#,
);

testcase!(
    test_unpack_typevar_bound_to_tuple_starred,
    r#"
from typing import reveal_type
def f[Z: tuple[str, int, bytes]](x: Z):
    a, *b = x
    reveal_type(a)  # E: revealed type: str
    reveal_type(b)  # E: revealed type: list[bytes | int]
"#,
);

testcase!(
    test_unpack_constrained_typevar_tuple,
    r#"
from typing import TypeVar, reveal_type
Z = TypeVar("Z", tuple[str, int], tuple[bool, bytes])
def f(x: Z):
    a, b = x
    reveal_type(a)  # E: revealed type: bool | str
    reveal_type(b)  # E: revealed type: bytes | int
"#,
);

testcase!(
    test_unpack_typevar_unbounded_not_iterable,
    r#"
def f[Z](x: Z):
    a, b = x  # E: Type `object` is not iterable
"#,
);

testcase!(
    test_unpack_typevar_bound_not_iterable,
    r#"
def f[Z: int](x: Z):
    a, b = x  # E: Type `int` is not iterable
"#,
);

testcase!(
    test_tuple_slice_non_literal,
    r#"
from typing import assert_type
def test(x: tuple[int, str, bool], y: tuple[int, ...], start: int, stop: int, step: int):
    assert_type(x[start:stop:step], tuple[int | str | bool, ...])
    assert_type(y[start:stop:step], tuple[int, ...])
"#,
);

testcase!(
    test_slice_subset,
    r#"
def f(x: slice) -> None:
    pass
def g(x: slice[int, int, int]) -> None:
    f(x)
"#,
);

testcase!(
    test_tuple_constructor,
    r#"
from typing import Any, Iterable
def test(y: Iterable[Any], z: Iterable[int]):
    x: tuple[int, int] = tuple(y)
    x = tuple(z)  # E: `tuple[int, ...]` is not assignable to variable `x` with type `tuple[int, int]`
"#,
);

testcase!(
    test_tuple_constructor_assert_type,
    r#"
from typing import assert_type, Iterable
def test(x: Iterable[int]) -> None:
    assert_type(tuple(x), tuple[int, ...])
"#,
);

testcase!(
    bug = "TODO: handle generator from fixed-length heterogeneous iterable",
    test_tuple_constructor_preserves_fixed_length,
    r#"
from typing import assert_type
def test(xs: tuple[int, int]) -> None:
    ys: tuple[int, int] = tuple(xs)
    assert_type(tuple(xs), tuple[int, int])
    zs: tuple[int, int] = tuple(x for x in xs)  # E: `tuple[int, ...]` is not assignable to `tuple[int, int]`
"#,
);

testcase!(
    test_tuple_constructor_concat,
    r#"
from typing import assert_type, Iterable, Literal
def test(x: Iterable[int]) -> None:
    assert_type(tuple(x) + (3,), tuple[*tuple[int, ...], Literal[3]])
"#,
);

testcase!(
    test_namedtuple_constructor_nominal,
    r#"
from typing import NamedTuple, assert_type
class Point(NamedTuple):
    x: int
    y: int
p = Point(1, 2)
assert_type(p, Point)
"#,
);

testcase!(
    test_tuple_subclass_constructor_nominal,
    r#"
from typing import assert_type
class MyTuple(tuple[int, ...]): pass
m = MyTuple([1, 2])
assert_type(m, MyTuple)
"#,
);

testcase!(
    test_star_unpack_single_unbounded_tuple,
    r#"
from typing import assert_type
def test(x: tuple[int, ...]) -> None:
    y = (*x,)
"#,
);

testcase!(
    test_star_unpack_union_of_tuples,
    r#"
from typing import assert_type
def f() -> tuple[int, ...] | tuple[str, ...]:
    ...
x = (*f(),)
"#,
);

testcase!(
    test_tuple_aug_assign,
    r#"
def test() -> None:
    x: tuple[object, ...] = (1,)
    x += (2, "y")
    y: tuple[int, ...] = (1,)
    y += (2, "y")  # E: Augmented assignment result `tuple[*tuple[int, ...], Literal[2], Literal['y']]` is not assignable to `tuple[int, ...]`
"#,
);

testcase!(
    test_tuple_concat,
    r#"
from typing import assert_type
def test(x: tuple[int, str], y: tuple[bool, ...], z: tuple[int, *tuple[str, ...], bool]) -> None:
    assert_type(x + x, tuple[int, str, int, str])
    assert_type(x + y, tuple[int, str, *tuple[bool, ...]])
    assert_type(x + z, tuple[int, str, int, *tuple[str, ...], bool])
    assert_type(y + x, tuple[*tuple[bool, ...], int, str])
    assert_type(y + y, tuple[bool, ...])
    assert_type(y + z, tuple[*tuple[bool | int | str, ...], bool])
    assert_type(z + x, tuple[int, *tuple[str, ...], bool, int, str])
    assert_type(z + y, tuple[int, *tuple[str | bool, ...]])
    assert_type(z + z, tuple[int, *tuple[str | bool | int, ...], bool])
"#,
);

testcase!(
    test_tuple_concat_union,
    r#"
from typing import assert_type
def test(x: tuple[int] | tuple[str]) -> None:
    assert_type(x + x, tuple[int, int] | tuple[str, str] | tuple[int, str] | tuple[str, int])
"#,
);

testcase!(
    test_tuple_repeat,
    r#"
from typing import assert_type, Literal

assert_type((42,) * 2, tuple[Literal[42], Literal[42]])
assert_type(2 * (42,), tuple[Literal[42], Literal[42]])
assert_type((1, "x") * 2, tuple[Literal[1], Literal["x"], Literal[1], Literal["x"]])
assert_type((1,) * 0, tuple[()])
assert_type((1,) * -1, tuple[()])
assert_type((1,) * 257, tuple[Literal[1], ...])
"#,
);

testcase!(
    test_unpack_tuple_with_double_def,
    r#"
from typing import Unpack, Any
def f(*args: Unpack[tuple[Any, ...]]):
    pass

def f():
     pass
"#,
);

testcase!(
    test_tuple_equivalence,
    r#"
from typing import assert_type

def f(x: tuple):
    assert_type(x, tuple)

def g(x):
    if isinstance(x, tuple):
        assert_type(x, tuple)
"#,
);

#[test]
fn test_tuple_concat_large_union_no_crash() -> anyhow::Result<()> {
    let code = r#"
a: int | list[int] | tuple[int, ...] | bool
a + (a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a)
"#;
    let (state, handle_fn) = TestEnv::one("main", code).to_state();
    let handle = handle_fn("main");
    state.transaction().get_errors(&[handle]);
    Ok(())
}

testcase!(
    test_tuple_class_type,
    r#"
from typing import Any

def f(x: type[tuple[Any, ...]]):
    return x() # Ok
    "#,
);

testcase!(
    test_bad_tuple_index,
    r#"
def f(x: tuple[int, int], y: tuple[int, ...]):
    x[(1, 2)]  # E: No matching overload found for function `tuple.__getitem__`
    y[(1, 2)]  # E: No matching overload found for function `tuple.__getitem__`
    "#,
);

testcase!(
    test_typevartuple_subclass_index,
    r#"
from typing import assert_type, TypeVarTuple
Ts = TypeVarTuple('Ts')
class TupleChild(tuple[*Ts]): ...
def f(x: TupleChild[int, str]):
    assert_type(x[0], int)
    assert_type(x[1], str)
    x[2]  # E: Index 2 out of range for tuple with 2 elements
    "#,
);

testcase!(
    test_tuple_subclass_getitem_override,
    r#"
from typing import assert_type

class Foo(tuple[int, ...]):
    def __getitem__(self, name: str) -> int:  # E: `Foo.__getitem__` has type
        ...

def test(foo: Foo) -> None:
    assert_type(foo["test"], int)
    "#,
);

testcase!(
    test_tuple_subclass_inherited_getitem_override,
    r#"
from typing import assert_type

class Parent(tuple[int, ...]):
    def __getitem__(self, name: str) -> int:  # E: `Parent.__getitem__` has type
        ...

class Child(Parent):
    pass

def test(c: Child) -> None:
    assert_type(c["test"], int)
    "#,
);

testcase!(
    test_starred_empty_tuple_no_panic,
    r#"
(),*()
    "#,
);

// https://github.com/facebook/pyrefly/issues/273
// https://discuss.python.org/t/unbounded-tuple-unions/92472
testcase!(
    test_union_empty_tuple_and_variadic_tuple,
    r#"
type Eq0 = tuple[()]
type Eq1 = tuple[int]
type Ge0 = tuple[int, ...]
type Ge1 = tuple[int, *Ge0]

def test(eq0: Eq0, eq1: Eq1, ge0: Ge0, ge1: Ge1) -> None:
    eq0_ge1__eq0: Eq0 | Ge1 = eq0
    eq0_ge1__eq1: Eq0 | Ge1 = eq1
    eq0_ge1__ge0: Eq0 | Ge1 = ge0
    eq0_ge1__ge1: Eq0 | Ge1 = ge1
    "#,
);

testcase!(
    test_giant_tuple_literal,
    r#"
# literal tuples with >256 elements get inferred as `tuple[Any, ...]`

from typing import assert_type, Any
x = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256)
assert_type(x, tuple[Any, ...])
"#,
);

testcase!(
    test_assign_unknown_tuple_to_concrete_tuple,
    r#"
def f(x):
    y: tuple[float, float] = tuple(x)
    "#,
);

testcase!(
    test_assign_varlength_tuple_to_concrete_tuple_error,
    r#"
from typing import Any, Iterable
def f(x: Iterable[float]):
    y: tuple[float, float] = tuple(x)  # E: `tuple[float, ...]` is not assignable to `tuple[float, float]`
    "#,
);

testcase!(
    test_tuple_iterable_mismatch,
    r#"
from typing import Iterable
def f(x: tuple[str, ...]): ...
def g(x: Iterable[int]):
    f(tuple(x))  # E: `tuple[int, ...]` is not assignable to parameter `x` with type `tuple[str, ...]`
    "#,
);

testcase!(
    test_tuple_constraint_mismatch,
    r#"
def f[T: (int, str)](x: tuple[T, ...], y: tuple[T, T]):
    pass
f((1, 2), ("", ""))  # E: `tuple[Literal[''], Literal['']]` is not assignable to parameter `y` with type `tuple[int, int]`
    "#,
);

testcase!(
    test_hint_influences_tuple_type,
    r#"
from typing import Literal
CONSTS = ("a", "b")
x: tuple[Literal["a", "b"], ...] = tuple(CONSTS)

    "#,
);

testcase!(
    test_callable_tuple_mismatch,
    r#"
from typing import Callable
def make_tuple[T](x: T) -> tuple[T, ...]:
    return (x,)
f: Callable[[int], tuple[int, str]] = make_tuple  # E: `[T](x: T) -> tuple[T, ...]` is not assignable to `(int) -> tuple[int, str]`
    "#,
);

// Regression test: widening wide tuple unions keeps type complexity linear at control-flow joins.
// Without it, conditionally appending to a tuple produces an exponential number of concrete tuple
// variants and causes a OOM
testcase!(
    test_many_conditional_tuple_appends,
    r#"
class D0: pass
class D1: pass
class D2: pass
class D3: pass
class D4: pass
class D5: pass
class D6: pass
class D7: pass
class D8: pass
class D9: pass
class D10: pass
class D11: pass
class D12: pass
class D13: pass
class D14: pass
class D15: pass
class D16: pass
class D17: pass
class D18: pass
class D19: pass
class D20: pass
class D21: pass
class D22: pass
class D23: pass
class D24: pass
class D25: pass
class D26: pass
class D27: pass
class D28: pass
class D29: pass
class D30: pass
class D31: pass
class D32: pass
class D33: pass
class D34: pass
class D35: pass
class D36: pass
class D37: pass
class D38: pass
class D39: pass
class D40: pass
class D41: pass
class D42: pass
class D43: pass
class D44: pass
class D45: pass
class D46: pass
class D47: pass
class D48: pass
class D49: pass
class D50: pass
class D51: pass
class D52: pass
class D53: pass
class D54: pass
class D55: pass
class D56: pass
class D57: pass
class D58: pass
class D59: pass
class D60: pass
class D61: pass
class D62: pass
class D63: pass
class D64: pass

def repro(conds: list[bool]):
    z = ()
    if conds[0]: z += (D0(),)
    if conds[1]: z += (D1(),)
    if conds[2]: z += (D2(),)
    if conds[3]: z += (D3(),)
    if conds[4]: z += (D4(),)
    if conds[5]: z += (D5(),)
    if conds[6]: z += (D6(),)
    if conds[7]: z += (D7(),)
    if conds[8]: z += (D8(),)
    if conds[9]: z += (D9(),)
    if conds[10]: z += (D10(),)
    if conds[11]: z += (D11(),)
    if conds[12]: z += (D12(),)
    if conds[13]: z += (D13(),)
    if conds[14]: z += (D14(),)
    if conds[15]: z += (D15(),)
    if conds[16]: z += (D16(),)
    if conds[17]: z += (D17(),)
    if conds[18]: z += (D18(),)
    if conds[19]: z += (D19(),)
    if conds[20]: z += (D20(),)
    if conds[21]: z += (D21(),)
    if conds[22]: z += (D22(),)
    if conds[23]: z += (D23(),)
    if conds[24]: z += (D24(),)
    if conds[25]: z += (D25(),)
    if conds[26]: z += (D26(),)
    if conds[27]: z += (D27(),)
    if conds[28]: z += (D28(),)
    if conds[29]: z += (D29(),)
    if conds[30]: z += (D30(),)
    if conds[31]: z += (D31(),)
    if conds[32]: z += (D32(),)
    if conds[33]: z += (D33(),)
    if conds[34]: z += (D34(),)
    if conds[35]: z += (D35(),)
    if conds[36]: z += (D36(),)
    if conds[37]: z += (D37(),)
    if conds[38]: z += (D38(),)
    if conds[39]: z += (D39(),)
    if conds[40]: z += (D40(),)
    if conds[41]: z += (D41(),)
    if conds[42]: z += (D42(),)
    if conds[43]: z += (D43(),)
    if conds[44]: z += (D44(),)
    if conds[45]: z += (D45(),)
    if conds[46]: z += (D46(),)
    if conds[47]: z += (D47(),)
    if conds[48]: z += (D48(),)
    if conds[49]: z += (D49(),)
    if conds[50]: z += (D50(),)
    if conds[51]: z += (D51(),)
    if conds[52]: z += (D52(),)
    if conds[53]: z += (D53(),)
    if conds[54]: z += (D54(),)
    if conds[55]: z += (D55(),)
    if conds[56]: z += (D56(),)
    if conds[57]: z += (D57(),)
    if conds[58]: z += (D58(),)
    if conds[59]: z += (D59(),)
    if conds[60]: z += (D60(),)
    if conds[61]: z += (D61(),)
    if conds[62]: z += (D62(),)
    if conds[63]: z += (D63(),)
    if conds[64]: z += (D64(),)
    return z
"#,
);
