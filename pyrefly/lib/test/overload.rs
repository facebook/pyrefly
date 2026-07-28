/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every SPELL

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_py,
    r#"
from typing import overload, assert_type

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...

def f(x):
    return x

assert_type(f(1), int)

def anywhere():
    assert_type(f(1), int)
    "#,
);

testcase!(
    test_useless_overload_body,
    r#"
from typing import overload

# should warn

@overload
def returns_expr(x: int) -> int:  # E: `@overload` bodies should not contain executable logic
    return x + 1

@overload
def returns_expr(x: str) -> str:
    ...

@overload
def raises_other(x: int) -> int:  # E: `@overload` bodies should not contain executable logic
    raise ValueError("bad")

@overload
def raises_other(x: str) -> str:
    ...

@overload
def has_assignment(x: int) -> int:  # E: `@overload` bodies should not contain executable logic
    x = 1
    return x

@overload
def has_assignment(x: str) -> str:
    ...

@overload
def has_multiple_stmts(x: int) -> int:  # E: `@overload` bodies should not contain executable logic
    print("side effect")
    return x

@overload
def has_multiple_stmts(x: str) -> str:
    ...

def returns_expr(x: int | str) -> int | str:
    return x

def raises_other(x: int | str) -> int | str:
    return x

def has_assignment(x: int | str) -> int | str:
    return x

def has_multiple_stmts(x: int | str) -> int | str:
    return x

# should not warn 

@overload
def body_pass(x: int) -> int:
    pass

@overload
def body_pass(x: str) -> str:
    ...

def body_pass(x: int | str) -> int | str:
    return x

@overload
def body_ellipsis(x: int) -> int:
    ...

@overload
def body_ellipsis(x: str) -> str:
    ...

def body_ellipsis(x: int | str) -> int | str:
    return x

@overload
def body_docstring_only(x: int) -> int:
    """This is fine."""

@overload
def body_docstring_only(x: str) -> str:
    ...

def body_docstring_only(x: int | str) -> int | str:
    return x

@overload
def body_raise_not_impl(x: int) -> int:
    raise NotImplementedError

@overload
def body_raise_not_impl(x: str) -> str:
    ...

def body_raise_not_impl(x: int | str) -> int | str:
    return x

@overload
def body_raise_not_impl_msg(x: int) -> int:
    raise NotImplementedError("not done")

@overload
def body_raise_not_impl_msg(x: str) -> str:
    ...

def body_raise_not_impl_msg(x: int | str) -> int | str:
    return x

@overload
def body_return_not_impl(x: int) -> int:
    return NotImplemented

@overload
def body_return_not_impl(x: str) -> str:
    ...

def body_return_not_impl(x: int | str) -> int | str:
    return x

@overload
def body_docstring_then_pass(x: int) -> int:
    """docstring stripped first, then pass is trivial."""
    pass

@overload
def body_docstring_then_pass(x: str) -> str:
    ...

def body_docstring_then_pass(x: int | str) -> int | str:
    return x
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2867
testcase!(
    test_urlunparse_prefers_string_overload_for_parse_result,
    r#"
from typing import assert_type
from urllib.parse import urlparse, urlunparse

def sanitize_url(url: str) -> str:
    parsed = urlparse(url)
    assert_type(urlunparse(parsed), str)
    sanitized = parsed._replace(netloc="example.com")
    assert_type(urlunparse(sanitized), str)
    return urlunparse(sanitized)
    "#,
);

testcase!(
    test_branches,
    r#"
from typing import assert_type, overload
def test(x: bool):
    if x:
        def f(x: str) -> bytes: ...
    else:
        @overload
        def f(x: int) -> int: ...
        @overload
        def f(x: str) -> str: ...
        def f(x: int | str) -> int | str:
            return x
    def g(x: str):
        assert_type(f(x), bytes | str)
    "#,
);

fn env_with_stub() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path(
        "foo",
        "foo.pyi",
        r#"
from typing import overload

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...
    "#,
    );
    t
}

testcase!(
    test_pyi,
    env_with_stub(),
    r#"
from typing import assert_type
import foo
assert_type(foo.f(1), int)
    "#,
);

testcase!(
    test_protocol,
    r#"
from typing import Protocol, assert_type, overload

class P(Protocol):
    @overload
    def m(self, x: int) -> int: ...
    @overload
    def m(self, x: str) -> str: ...

def test(o: P):
    assert_type(o.m(1), int)
    "#,
);

testcase!(
    test_method,
    r#"
from typing import assert_type, overload

class C:
    @overload
    def m(self, x: int) -> int: ...
    @overload
    def m(self, x: str) -> str: ...
    def m(self, x: int | str) -> int | str:
        return x

def test(o: C):
    assert_type(o.m(1), int)
    "#,
);

testcase!(
    test_overload_method_name_matches_class,
    r#"
from typing import assert_type, overload

class A:
    @overload
    def B(self, x: int) -> B: ...
    @overload
    def B(self, x: str) -> B: ...
    def B(self, x):
        return B()

class B:
    x: int

assert_type(A().B(0).x, int)
assert_type(A().B("1").x, int)
    "#,
);

testcase!(
    test_overload_arg_errors,
    r#"
from typing import overload, assert_type

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x: int | str) -> int | str: ...

def g(x: str) -> int: ...
def h(x: str) -> str: ...

assert_type(f(g(0)), int) # E: Argument `Literal[0]` is not assignable to parameter `x` with type `str`
assert_type(f(h(0)), str) # E: Argument `Literal[0]` is not assignable to parameter `x` with type `str`
"#,
);

testcase!(
    test_overload_default_incompatible,
    r#"
from typing import Literal, overload

@overload
def foo(a: Literal[True] = ...) -> None: ...  # E: Default `Literal[False]` from implementation is not assignable to overload parameter `a` with type `Literal[True]`
@overload
def foo(a: Literal[False]) -> int: ...
def foo(a: bool = False) -> None | int:
    return 1 if not a else None
"#,
);

testcase!(
    test_overload_default_matches_one_signature,
    r#"
from typing import overload

@overload
def f(x: int = ..., y: None = ...) -> None: ...

# In this situation, we're forced to write `y: int = ...` to avoid a
# "parameter without a default follows parameter with a default" syntax error,
# even though any call that omits `y` will match the first overload above.
# So we should not error here even though `y`'s `None` default is not an `int`.
@overload
def f(x: int = ..., y: int = ...) -> int: ...

def f(x: int = 0, y: int | None = None):
    if y is None:
        return None
    return x + y

# Make sure using a `None` default instead of `...` in the overload works as well.
@overload
def g(x: int = ..., y: None = None) -> None: ...
@overload
def g(x: int = ..., y: int = ...) -> int: ...
def g(x: int = 0, y: int | None = None):
    if y is None:
        return None
    return x + y
"#,
);

testcase!(
    test_overload_default_does_not_match_second_signature,
    r#"
from typing import overload

@overload
def f(x: int, y: None = ...) -> str: ...

# False negative: an `f()` call (no args) matches this overload, but `y`'s default of `None` is
# inconsistent with its type of `int` in the signature. We allow this in order to avoid a false
# positive in test_overload_default_matches_one_signature.
@overload
def f(x: int = ..., y: int = ...) -> int: ...

def f(x: int = 0, y: int | None = None) -> str | int: ...
"#,
);

// See `dataclasses_transform_field.py` in the conformance test suite.
// This pattern is used to indicate that the value of one parameter is implicitly specified by
// another, which is useful for expressing dataclass transforms.
testcase!(
    test_overload_default_do_not_check_non_ellipsis,
    r#"
from typing import Literal, overload

@overload
def f(x: int, y: Literal[True] = True): ...
@overload
def f(x: None, y: Literal[False] = False): ...
def f(x: int | None, y: bool = False): ...
    "#,
);

testcase!(
    test_overload_missing_implementation,
    r#"
from typing import overload, assert_type

@overload
def f(x: int) -> int: ... # E: Overloaded function must have an implementation
@overload
def f(x: str) -> str: ...

# still behaves like an overload
assert_type(f(0), int)
assert_type(f(""), str)
"#,
);

testcase!(
    test_overload_static_config,
    r#"
from typing import overload, assert_type
import sys

@overload
def f(x: int) -> int: ... # E: Overloaded function must have an implementation

if sys.version_info >= (3, 11):
    @overload
    def f(x: str) -> str: ...
else:
    @overload
    def f(x: int, int) -> bool: ...

if sys.version_info >= (3, 12):
    @overload
    def f() -> None: ...

assert_type(f(0), int)
assert_type(f(""), str)
assert_type(f(), None)
f(0, 0) # E: No matching overload found
"#,
);

testcase!(
    test_only_one_overload,
    r#"
from typing import overload, Protocol

@overload
def f(x: int) -> int: ...  # E: Overloaded function needs at least two @overload declarations
def f(x: int) -> int:
    return x

@overload
def g(x: int) -> int: ...  # E: Overloaded function must have an implementation  # E: Overloaded function needs at least two @overload declarations

class P(Protocol):
    @overload
    def m(x: int) -> int: ...  # E: Overloaded function needs at least two @overload declarations
"#,
);

testcase!(
    test_overload_ignore,
    r#"
from typing import Never, overload, assert_type

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x: int | str) -> int | str:
    return x

x = f("foo") # type: ignore
# intentionally blank: make sure we don't ignore the assert_type below
assert_type(x, str)
"#,
);

testcase!(
    test_typeguard,
    r#"
from typing import assert_type, overload, TypeGuard

class Animal: ...
class Mammal(Animal): ...
class Cat(Mammal): ...
class Bird(Animal): ...
class Robin(Bird): ...

@overload
def f(x: Mammal) -> TypeGuard[Cat]: ...
@overload
def f(x: Bird) -> TypeGuard[Robin]: ...
def f(x: Animal) -> bool: ...

class A:
    @overload
    def f(self, x: Mammal) -> TypeGuard[Cat]: ...
    @overload
    def f(self, x: Bird) -> TypeGuard[Robin]: ...
    def f(self, x: Animal) -> bool: ...

def g(meow: Mammal, chirp: Bird):
    if f(meow):
        assert_type(meow, Cat)
    if A().f(chirp):
        assert_type(chirp, Robin)
    "#,
);

testcase!(
    test_classmethod,
    r#"
from typing import assert_type, overload
class A:
    @overload
    @classmethod
    def f(cls, x: int) -> int: ...

    @overload
    @classmethod
    def f(cls, x: str) -> str: ...

    @classmethod
    def f(cls, x: int | str) -> int | str:
        return x

assert_type(A().f(1), int)
    "#,
);

testcase!(
    test_invalid_decoration,
    r#"
from typing import overload

def decorate(f) -> float:
    return 0

@overload
def f(x: str) -> str: ...

@decorate
@overload
def f(x: int) -> int: ...  # E: `f` has type `float` after decorator application, which is not callable

def f(x: str | int) -> str | int:
    return x
    "#,
);

testcase!(
    test_decoration,
    r#"
from typing import Callable, assert_type, overload

def decorate(f) -> Callable[[int], int]:
    return lambda x: x

@overload
@decorate
def f(x: bytes) -> bytes: ...

@overload
def f(x: str) -> str: ...

def f(x: object) -> object:
    return x

assert_type(f(0), int)
f(b"")  # E: No matching overload found for function `f`
    "#,
);

testcase!(
    test_overload_assignable_to_callable_union,
    r#"
from typing import Callable, overload

@overload
def foo(x: int) -> str: ...
@overload
def foo(x: str) -> str: ...
def foo(x: int | str) -> str:
    return str(x)

bar: Callable[[int | str], str] = foo
baz: Callable[[int | str | bytes], str] = foo  # E: not assignable
    "#,
);

testcase!(
    test_overload_assignable_to_callable_union_multi_param,
    r#"
from typing import Callable, overload

@overload
def foo(x: int, y: bytes) -> int: ...
@overload
def foo(x: int, y: str) -> int: ...
def foo(x: int, y: bytes | str) -> int:
    return 0

bar: Callable[[int, bytes | str], int] = foo
    "#,
);

testcase!(
    test_overload_assignable_to_callable_return_supertype,
    r#"
from typing import Callable, overload

@overload
def foo(x: int) -> bool: ...
@overload
def foo(x: str) -> bool: ...
def foo(x: int | str) -> bool:
    return False

bar: Callable[[int | str], int] = foo
baz: Callable[[int | str], str] = foo  # E: not assignable
    "#,
);

testcase!(
    test_overload_assignable_to_callable_return_union,
    r#"
from typing import Callable, overload

@overload
def foo(x: int) -> int: ...
@overload
def foo(x: str) -> str: ...
def foo(x: int | str) -> int | str:
    return x

bar: Callable[[int | str], int | str] = foo
baz: Callable[[int | str], int | str | bytes] = foo
qux: Callable[[int | str], int] = foo  # E: not assignable
    "#,
);

testcase!(
    test_final_decoration_on_top_level_function,
    r#"
from typing import assert_type, final, overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
@final  # E: `@final` can only be used on methods
def f(x: int | str) -> int | str:
    return x

assert_type(f(0), int)
    "#,
);

testcase!(
    test_overload_inconsistent_override_final,
    r#"
from typing import overload, override, Any, final

class Base:
    def f(x: Any) -> Any: pass
    def f2(x: Any) -> Any: pass

class C(Base):
    # OK
    @override
    @overload
    def f(x: int) -> int: ...  # E: Overloaded function must have an implementation
    @overload
    def f(x: str) -> str: ...

    # not on the first overload
    @overload
    def f2(x: int) -> int: ...  # E: Overloaded function must have an implementation
    @overload
    @override
    def f2(x: str) -> str: ...  # E: `@override` should be applied to the first overload only.

    # OK
    @overload
    def f3(x: int) -> int: ...
    @overload
    def f3(x: str) -> str: ...
    @final
    def f3(x: int | str):
        return x

    # not on the implementation
    @final
    @overload
    def f3(x: int) -> int: ...  # E: `@final` should only be applied to the implementation of an overloaded function
    @overload
    def f3(x: str) -> str: ...
    def f3(x: int | str):
        return x
    "#,
);

testcase!(
    test_overload_inconsistent_staticmethod_classmethod,
    r#"
from typing import overload

class C:
    # missing from an overload
    @overload
    def f(x: int) -> int: ...  # E: Overloaded function must have an implementation  # E: If `@staticmethod` is present on one overload, all overloads must have that decorator
    @staticmethod
    @overload
    def f(x: str) -> str: ...

    # OK
    @staticmethod
    @overload
    def f2(x: int) -> int: ...  # E: Overloaded function must have an implementation
    @staticmethod
    @overload
    def f2(x: str) -> str: ...

    # OK
    @classmethod
    @overload
    def f3(x: int) -> int: ...
    @classmethod
    @overload
    def f3(x: str) -> str: ...
    @classmethod
    def f3(x: int | str):  # E: `f3` method cls type `int | str` is not a valid `type[...]` annotation
        return x

    # missing from implementation
    @classmethod
    @overload
    def f4(x: int) -> int: ...
    @classmethod
    @overload
    def f4(x: str) -> str: ...
    def f4(x: int | str):  # E: If `@classmethod` is present on any overload or the implementation, it should be on every overload and the implementation
        return x

    # missing from an overload
    @classmethod
    @overload
    def f5(x: int) -> int: ...
    @overload
    def f5(x: str) -> str: ...  # E: If `@classmethod` is present on any overload or the implementation, it should be on every overload and the implementation
    @classmethod
    def f5(x: int | str):  # E: `f5` method cls type `int | str` is not a valid `type[...]` annotation
        return x
    "#,
);

testcase!(
    test_defaultdict_constructor_overload_1,
    r#"
from collections import defaultdict
from typing import DefaultDict
x: DefaultDict[int, list[int]] = defaultdict(list)
    "#,
);

testcase!(
    test_defaultdict_constructor_overload_2,
    r#"
from collections import defaultdict
x: dict[int, int] = defaultdict(int)
    "#,
);

testcase!(
    test_defaultdict_constructor_overload_3,
    r#"
import collections
std_aggs: dict[int, tuple[list[str], list[str]]] = collections.defaultdict(
    lambda: ([], [])
)
std_aggs[0][1].append('foo')
    "#,
);

testcase!(
    test_constructor_overload_with_hint,
    r#"
from typing import Callable, overload
class defaulty[K, V]:
    @overload
    def __init__(self: defaulty[str, V], **kwargs: V) -> None: ... # E: `__init__` method self type cannot reference class type parameter `V`
    @overload
    def __init__(self, default_factory: Callable[[], V] | None, /) -> None: ...
    def __init__(self, *args, **kwargs) -> None:
        return None
badge: defaulty[bool, list[str]] = defaulty(list)
    "#,
);

testcase!(
    test_pass_generic_class_to_overload,
    r#"
from typing import Iterable, Iterator, Literal, overload, Self
from _typeshed import SupportsAdd

@overload
def f(x: Iterable[Literal[1]]) -> None: ...
@overload
def f(x: Iterable[SupportsAdd]) -> None: ...
def f(x) -> None: ...

class C[T](Iterable[T]):
    def __new__(cls, x: T) -> Self: ...
    def __iter__(self) -> Iterator[T]: ...

def g(x: int):
    f(C(x))
    "#,
);

testcase!(
    test_overload_type_form_inference,
    r#"
from typing import assert_type, overload

class C: ...

@overload
def foo[T](x: type[T]) -> T: ...  # E: Overloaded function must have an implementation
@overload
def foo(x: int) -> int: ...

def bar[T](x: type[T]) -> T: ...

assert_type(foo(C), C)
assert_type(bar(C), C)
    "#,
);

testcase!(
    test_overload_exponential,
    r#"
# This used to take an exponential amount of time to type check

from typing import overload, Any
class X: ...

@overload
def f(a: int) -> X: ...
@overload
def f(a: str) -> X: ...
def f(a: Any) -> X: return X()

def exponential() -> Any:
    f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(X())))))))))))))))))))))))) # E: # E: # E: # E: # E: # E: # E: # E: # E: # E: # E: # E:
"#,
);

testcase!(
    test_implementation_with_overload,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...

@overload
def f(x: int | str) -> int | str: # E: @overload decorator should not be used on function implementation  # E: `@overload` bodies should not contain executable logic
    return x
    "#,
);

testcase!(
    test_implementation_before_overload,
    r#"
from typing import overload

def f(x: int | str) -> int | str: # E: @overload declarations must come before function implementation
    return x

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...
    "#,
);

testcase!(
    test_overload_with_docstring,
    r#"
from typing import overload, Any

@overload
def foo(a: int) -> int: ...
@overload
def foo(a: str) -> str:
    """Docstring"""
def foo(*args, **kwargs) -> Any:
    pass
    "#,
);

testcase!(
    test_overload_with_docstring2,
    r#"
from typing import overload, Any

@overload
def foo(a: int) -> int: ...
@overload
def foo(a: str) -> str:  # E: `@overload` bodies should not contain executable logic
    """Docstring"""
    return 123             # E: Returned type `Literal[123]` is not assignable to declared return type `str`
def foo(*args, **kwargs) -> Any:
    pass
    "#,
);

testcase!(
    test_overload_with_docstring3,
    r#"
def foo() -> int: # E: Function declared to return `int` but is missing an explicit `return`
    """hello"""
    "#,
);

testcase!(
    test_return_consistency_explicit_return,
    r#"
from typing import overload

@overload
def f1(x: int) -> int: ...
@overload
def f1(x: str) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `int`
def f1(x: int | str) -> int:
    return int(x)

@overload
def f2(x: int) -> int: ...
@overload
def f2(x: str) -> str: ...
def f2(x: int | str) -> int | str:
    return x
    "#,
);

testcase!(
    test_return_consistency_inferred_return,
    r#"
from typing import overload

@overload
def f1(a: int) -> int: ...
@overload
def f1(a: str) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `int`
def f1(a):
    return 1

@overload
def f2(a: int) -> int: ...
@overload
def f2(a: str) -> str: ...
def f2(a):
    return 1 if a else ""
    "#,
);

testcase!(
    test_generic_overloads_are_consistent,
    r#"
from typing import Any, overload

@overload
def f(a: int, b: Any) -> float: ...
@overload
def f[T](a: float, b: T) -> T: ...
def f[T](a: float, b: T) -> T: ...
    "#,
);

testcase!(
    test_overloads_are_consistent_with_generic_impl,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f[T](x: T) -> T:
    return x
    "#,
);

testcase!(
    test_param_consistency,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> int: ...  # E: Implementation signature `(x: int) -> int` does not accept all arguments that overload signature `(x: str) -> int` accepts
def f(x: int) -> int:
    return x
    "#,
);

testcase!(
    test_typevar_bound_consistency,
    r#"
from typing import overload

@overload
def f[T: str](x: T) -> T: ...  # E: `str` is not assignable to upper bound `bytes` of type variable `T`
@overload
def f[T: bytes](x: T) -> T: ...
def f[T: bytes](x: T) -> T:
    return x
    "#,
);

testcase!(
    test_generic_implementation_multiple_typevars,
    r#"
from typing import overload

@overload
def f(x: int, y: str) -> tuple[str, int]: ...
@overload
def f(x: bool, y: float) -> tuple[float, bool]: ...
def f[T1, T2](x: T1, y: T2) -> tuple[T2, T1]:
    return (y, x)

@overload
def g(x: int, y: str) -> tuple[str, int]: ...  # E: `tuple[str, int]` is not assignable to implementation return type `tuple[int, str]`
@overload
def g(x: bool, y: float) -> tuple[float, bool]: ...  # E: `tuple[float, bool]` is not assignable to implementation return type `tuple[bool, float]`
def g[T1, T2](x: T1, y: T2) -> tuple[T1, T2]:
    return (x, y)
    "#,
);

testcase!(
    test_generic_overload_nongeneric_impl,
    r#"
from typing import Any, overload

@overload
def f[T](x: T, y=None) -> T: ...  # E: does not accept all arguments  # E: not assignable to implementation return type
@overload
def f(x, y) -> Any: ...
def f(x: int, y=None) -> int:
    return x

@overload
def g[T: int](x: T, y=None) -> T: ...
@overload
def g(x, y) -> Any: ...
def g(x: int, y=None) -> int:
    return x
    "#,
);

testcase!(
    test_overload_typeddict_errors,
    r#"
from typing import Any, TypedDict, overload, assert_type

class TD(TypedDict):
    x: int

@overload
def foo(d: TD) -> None: ...
@overload
def foo(d: int) -> None: ...
def foo(d: TD | int) -> None: ...

assert_type(foo({ "x": "foo" }), Any) # E: No matching overload found for function `foo`
    "#,
);

testcase!(
    test_generic_impl_input_inconsistent,
    r#"
from typing import overload, TypeVar
T = TypeVar("T")
S = TypeVar("S")

# The implementation signature is intentionally inconsistent with this overload signature
# (`exception` is kw-only in the implementation) so that we can test how legacy TypeVars are
# printed in the error message.
@overload
def catch(exception: T) -> T: ...  # E: Implementation signature `(f: S | None = None, *, exception: T) -> S | T` does not accept all arguments that overload signature `(exception: T) -> T` accepts

@overload
def catch(f: S, *, exception: T) -> S | T: ...

def catch(f: S | None = None, *, exception: T) -> S | T: ...
    "#,
);

testcase!(
    test_abstract,
    r#"
from abc import ABC, abstractmethod
from typing import Literal, overload

class Derp(ABC):
    @overload
    @abstractmethod
    def f(self, m: Literal["x"] = "x") -> int: ...
    @overload
    @abstractmethod
    def f(self, m: str) -> str: ...
    @abstractmethod
    def f(self, m: str = "x") -> int | str: ...

def test(x: Derp, m: Literal["y"] = "y") -> str:
    return x.f(m)
    "#,
);

testcase!(
    test_expand_union,
    r#"
from typing import assert_type, overload
@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x: int | str) -> int | str:
    return x

def g(x: int | str):
    y = f(x)
    assert_type(y, int | str)
    "#,
);

testcase!(
    test_expand_second_arg,
    r#"
from typing import assert_type, overload

@overload
def f(x: int, y: int) -> int: ...
@overload
def f(x: int, y: str) -> str: ...
def f(x: int, y: int | str) -> int | str:
    return y

def g(y: int | str):
    assert_type(f(0, y), int | str)
    "#,
);

testcase!(
    test_expand_twice,
    r#"
from typing import assert_type, overload

@overload
def f(x: int, y: int) -> int: ...
@overload
def f(x: int, y: str) -> int: ...
@overload
def f(x: str, y: int) -> str: ...
@overload
def f(x: str, y: str) -> str: ...
def f(x: int | str, y: int | str) -> int | str:
    return x

def g(x: int | str, y: int | str):
    assert_type(f(x, y), int | str)
    "#,
);

testcase!(
    test_expand_bool,
    r#"
from typing import assert_type, overload, Literal

@overload
def f(x: Literal[True]) -> Literal['True']: ...
@overload
def f(x: Literal[False]) -> Literal['False']: ...
def f(x: bool) -> str:
    return str(x)

def g(x: bool):
    assert_type(f(x), Literal['True', 'False'])
    "#,
);

testcase!(
    test_expand_enum,
    r#"
from enum import Enum
from typing import assert_type, overload, Literal

class E(Enum):
    X = 1
    Y = 2

@overload
def f(x: Literal[E.X]) -> Literal['X']: ...
@overload
def f(x: Literal[E.Y]) -> Literal['Y']: ...
def f(x: E) -> str:
    return x.name

def g(x: E):
    assert_type(f(x), Literal['X', 'Y'])
    "#,
);

testcase!(
    test_expand_one_member_enum,
    r#"
from enum import Enum
from typing import assert_type, overload, Literal

class E(Enum):
    X = 1

@overload
def f(x: Literal[E.X]) -> Literal['X']: ...
@overload
def f(x: str) -> str: ...
def f(x: E | str) -> str:
    return str(x)

def g(x: E):
    assert_type(f(x), Literal['X'])
    "#,
);

testcase!(
    test_expand_type_union,
    r#"
from typing import assert_type, overload

class A: ...
class B: ...

@overload
def f(x: type[A]) -> A: ...
@overload
def f(x: type[B]) -> B: ...
def f(x: type[A | B]) -> A | B:
    return x()

def g(x: type[A | B]):
    assert_type(f(x), A | B)
    "#,
);

testcase!(
    test_expand_tuple,
    r#"
from typing import assert_type, overload, Literal

@overload
def f(x: tuple[int, Literal[True]]) -> str: ...
@overload
def f(x: tuple[int, Literal[False]]) -> int: ...
def f(x: tuple[int, bool]) -> int | str:
    return str(x[0]) if x[1] else x[0]

def g(x: tuple[int, bool]):
    assert_type(f(x), str | int)
    "#,
);

testcase!(
    test_expand_refines_generic_match_dropping_never,
    TestEnv::new().enable_legacy_overload_expansion(),
    r#"
from typing import assert_type, overload, NoReturn, TypeVar
T = TypeVar("T")

@overload
def not_none(x: None) -> NoReturn: ...
@overload
def not_none(x: T) -> T: ...
def not_none(x: T | None) -> T:
    raise NotImplementedError

def g(x: int | None):
    assert_type(not_none(x), int)
    "#,
);

testcase!(
    test_no_refine_matched_overload_by_default,
    r#"
from typing import assert_type, overload, NoReturn, TypeVar
T = TypeVar("T")

@overload
def not_none(x: None) -> NoReturn: ...
@overload
def not_none(x: T) -> T: ...
def not_none(x: T | None) -> T:
    raise NotImplementedError

def g(x: int | None):
    assert_type(not_none(x), int | None)
    "#,
);

testcase!(
    test_expand_refines_when_unrelated_arg_expands_first,
    TestEnv::new().enable_legacy_overload_expansion(),
    r#"
from typing import assert_type, Never, overload, TypeVar
T = TypeVar("T")
@overload
def f(x: object, y: None) -> Never: ...
@overload
def f(x: object, y: T) -> T: ...
def f(x: object, y: T | None) -> T:
    if y is None:
        raise AssertionError
    return y
def test(x: int | str, y: int | None) -> None:
    # `x` expands first without dropping an arm; refinement must keep going to `y`.
    assert_type(f(x, y), int)
    "#,
);

testcase!(
    test_expand_refine_only_does_not_broaden,
    TestEnv::new().enable_legacy_overload_expansion(),
    r#"
from typing import assert_type, overload

@overload
def f(x: int) -> bytes: ...
@overload
def f(x: object) -> str: ...
def f(x: object) -> bytes | str:
    raise NotImplementedError

def g(x: int | str):
    assert_type(f(x), str)
    "#,
);

testcase!(
    test_expand_no_refine_without_uninhabited_arm,
    TestEnv::new().enable_legacy_overload_expansion(),
    r#"
from typing import assert_type

def f(a: list[int] | list[str], b: list[bytes]):
    assert_type(zip(a, b), zip[tuple[int | str, bytes]])
    "#,
);

testcase!(
    test_wrong_arity,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: int, y: int) -> int: ...
def f(x: int, y: int = 0) -> int:
    return x + y

f(0, 1, 2)  # E: (x: int, y: int) -> int [closest match]\n  Expected at most 2 arguments, got 3
    "#,
);

testcase!(
    test_wrong_method_arity,
    r#"
from typing import overload

class A:
    @overload
    def f(self, x: int, y: str = ..., /, *, z: float = ...) -> int: ...
    @overload
    def f(self, x: str, y: int = ..., /, *, z: float = ...) -> str: ...
    def f(self, x: int | str, y: int | str = "", /, *, z: float = 0.0) -> int | str: ...

A().f()  # E: Expected at least 1 argument, got 0
A().f(0, "1", 0.0)  # E: Expected at most 2 positional arguments, got 3
A().f(0, y="1", z=0.0)  # E: Expected at most 1 keyword argument, got 2
    "#,
);

testcase!(
    test_unpack_nothing,
    r#"
from typing import assert_type, overload

@overload
def f(x: int, y: int) -> int: ...
@overload
def f(x: str) -> str: ...
@overload
def f(x: float) -> float: ...
def f(x, y=0) -> int | str | float: ...

assert_type(f("", *()), str)
assert_type(f(0.0, **{}), float)
    "#,
);

testcase!(
    test_unpack_required,
    r#"
from typing import assert_type, overload

@overload
def f(x: str, /) -> str: ...
@overload
def f(x: int, y: int, /) -> int: ...
def f(x, y=0) -> str | int: ...

def g(*args: int):
    assert_type(f(0, *args), int)
    "#,
);

testcase!(
    test_select_using_args,
    r#"
from typing import assert_type, overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: int, *args: int) -> str: ...
def f(x: int, *args: int) -> int | str: ...

def g(x: int, y: tuple[()], z: tuple[int, ...]):
    assert_type(f(x, *y), int)
    assert_type(f(x, *z), str)
    "#,
);

testcase!(
    test_select_using_kwargs,
    r#"
from typing import assert_type, overload, TypedDict

@overload
def f(x: int) -> int: ...
@overload
def f(x: int, **kwargs: int) -> str: ...
def f(x: int, **kwargs: int) -> int | str: ...

class Empty(TypedDict):
    pass

def g(x: int, y: Empty, z: dict[str, int]):
    assert_type(f(x, **y), int)
    assert_type(f(x, **z), str)
    "#,
);

testcase!(
    test_materialization_eliminates_overload,
    r#"
from typing import Any, assert_type, overload

@overload
def f(x: list[Any]) -> int: ...
@overload
def f(x: list[str]) -> str: ...
def f(x: list[Any]) -> int | str: ...

def g(x: list[Any]):
    # Because all materializations of `list[Any]` (the argument) are assignable to `list[Any]` (the
    # type of parameter `x` in the first overload), we can eliminate the second overload, leaving
    # us with the first overload with return type `int`.
    assert_type(f(x), int)
    "#,
);

testcase!(
    test_materialization_does_not_eliminate_overload,
    r#"
from typing import Any, assert_type, overload

@overload
def f(x: list[int]) -> int: ...
@overload
def f(x: list[str]) -> str: ...
def f(x: Any) -> int | str: ...

def g(x: list[Any]):
    # There's no overload for which all materializations of `list[Any]` are assignable to the
    # parameter type, so we keep all overloads. Their return types are not equivalent, so we fall
    # back to `Any`.
    assert_type(f(x), Any)

    "#,
);

testcase!(
    test_callable_param_materialization,
    r#"
from typing import Any, assert_type, Callable, Never, overload

@overload
def f1(x: Callable[[int], None]) -> int: ...
@overload
def f1(x: Callable[[str], None]) -> str: ...
def f1(x: Any) -> int | str: ...

@overload
def f2(x: Callable[[Any], None]) -> int: ...
@overload
def f2(x: Callable[[str], None]) -> str: ...
def f2(x: Any) -> int | str: ...

@overload
def f3(x: Callable[[Never], None]) -> int: ...
@overload
def f3(x: Callable[[str], None]) -> str: ...
def f3(x: Any) -> int | str: ...

def g(x: Callable[[Any], None]):
    assert_type(f1(x), Any)
    assert_type(f2(x), int)
    assert_type(f3(x), int)
    "#,
);

testcase!(
    test_callable_ellipsis_materialization,
    r#"
from typing import Any, assert_type, Callable, overload, Protocol

class EverythingCallback(Protocol):
    def __call__(self, *args, **kwargs) -> None: ...

@overload
def f1(x: EverythingCallback) -> int: ...
@overload
def f1(x: Callable[[], None]) -> str: ...
def f1(x: Any) -> int | str: ...

@overload
def f2(x: Callable[[EverythingCallback], None]) -> int: ...
@overload
def f2(x: Callable[[Callable[[], None]], None]) -> str: ...
def f2(x: Any) -> int | str: ...

@overload
def f3(x: Callable[..., None]) -> int: ...
@overload
def f3(x: Callable[[], None]) -> str: ...
def f3(x: Any) -> int | str: ...

def g(x: Callable[..., None], y: Callable[[Callable[..., None]], None]):
    assert_type(f1(x), int)
    assert_type(f2(y), int)
    assert_type(f3(x), int)
    "#,
);

testcase!(
    test_list_vs_sequence_materialization,
    r#"
from typing import Any, assert_type, overload, Sequence

@overload
def f1(x: list[object]) -> int: ...
@overload
def f1(x: list[Any]) -> str: ...
def f1(x: Any) -> int | str: ...

@overload
def f2(x: Sequence[object]) -> int: ...
@overload
def f2(x: Sequence[Any]) -> str: ...
def f2(x: Any) -> int | str: ...

def g(x: list[Any]):
    assert_type(f1(x), Any)
    assert_type(f2(x), int)
    "#,
);

testcase!(
    test_tuple_materialization,
    r#"
from typing import Any, assert_type, overload

@overload
def f1(x: tuple[Any, ...]) -> int: ...
@overload
def f1(x: tuple[()]) -> str: ...
def f1(x: Any) -> int | str: ...

@overload
def f2(x: tuple[int, ...]) -> int: ...
@overload
def f2(x: tuple[str, ...]) -> str: ...
def f2(x: Any) -> int | str: ...

def g(x: tuple[Any, ...]):
    assert_type(f1(x), int)
    assert_type(f2(x), Any)
    "#,
);

testcase!(
    test_abstractmethod_does_not_need_implementation,
    r#"
from typing import overload
from abc import ABC, abstractmethod

class A(ABC):
    @overload
    @abstractmethod
    def f(self, x: int) -> int: ...
    @overload
    @abstractmethod
    def f(self, x: str) -> str: ...

class B(ABC):
    @abstractmethod
    @overload
    def f(self, x: int) -> int: ...
    @abstractmethod
    @overload
    def f(Self, x: str) -> str: ...
    "#,
);

testcase!(
    test_overload_error_shows_argument_types,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x): return x

# Test with wrong type - should show argument type in error
f(3.14)  # E: No matching overload found for function `f` called with arguments: (float)

# Test with keyword argument
f(x=3.14)  # E: No matching overload found for function `f` called with arguments: (x=float)

# Test with multiple arguments (Pyrefly infers literal types for constants)
@overload
def g(x: int, y: int) -> int: ...
@overload
def g(x: str, y: str) -> str: ...
def g(x, y): return x

g(1, "hello")  # E: No matching overload found for function `g` called with arguments: (Literal[1], Literal['hello'])
g(x=1, y="hello")  # E: No matching overload found for function `g` called with arguments: (x=Literal[1], y=Literal['hello'])
    "#,
);

testcase!(
    test_overload_error_shows_call_error,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x): return x

f(3.14)  # E: `float` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_varargs_materialization,
    r#"
from typing import Any, assert_type, overload

@overload
def f1(*args: int) -> int: ...
@overload
def f1(*args: str) -> str: ...
def f1(*args) -> int | str: ...

@overload
def f2(*args: Any) -> int: ...
@overload
def f2(*args: str) -> str: ...
def f2(*args) -> int | str: ...

def g(*args):
    # ambiguous whether this matches `f1`'s first or second overload, so fall back to return type Any
    assert_type(f1(*args), Any)
    # this matches `f2`'s first overload via https://typing.python.org/en/latest/spec/overload.html#step-5
    assert_type(f2(*args), int)

def h(args: Any):
    # make sure the entire iterable being `Any` works
    assert_type(f1(*args), Any)
    assert_type(f2(*args), int)
    "#,
);

testcase!(
    test_kwargs_materialization,
    r#"
from typing import Any, assert_type, overload

@overload
def f1(**kwargs: int) -> int: ...
@overload
def f1(**kwargs: str) -> str: ...
def f1(**kwargs) -> int | str: ...

@overload
def f2(**kwargs: Any) -> int: ...
@overload
def f2(**kwargs: str) -> str: ...
def f2(**kwargs) -> int | str: ...

def g(**kwargs):
    # ambiguous whether this matches `f1`'s first or second overload, so fall back to return type Any
    assert_type(f1(**kwargs), Any)
    # this matches `f2`'s first overload via https://typing.python.org/en/latest/spec/overload.html#step-5
    assert_type(f2(**kwargs), int)

def h(kwargs: Any):
    # make sure the entire mapping being `Any` works
    assert_type(f1(**kwargs), Any)
    assert_type(f2(**kwargs), int)
    "#,
);

testcase!(
    test_unsolved_typevar,
    r#"
from typing import overload
class Foo[T]:
    @overload
    def test(self, obj: None, cls: type[T]) -> str: ...
    @overload
    def test(self, obj: T, cls: type[T]) -> int: ...
    def test(self, obj: T | None, cls: type[T]) -> str | int: ...
    "#,
);

testcase!(
    test_literal_selection,
    r#"
import contextlib
import os
from typing import AnyStr, IO, Generator, Literal, assert_type, overload

@overload
@contextlib.contextmanager
def atomic_file(
    dest: str | os.PathLike[str], mode: Literal["wb", "w+b"] = ..., **kwargs
) -> Generator[IO[bytes], None, None]: ...
@overload
@contextlib.contextmanager
def atomic_file(
    dest: str | os.PathLike[str], mode: Literal["w", "w+", "wt", "w+t"], **kwargs
) -> Generator[IO[str], None, None]: ...
@overload
@contextlib.contextmanager
def atomic_file(
    dest: str | os.PathLike[str], mode: str, **kwargs
) -> Generator[IO[AnyStr], None, None]: ...

@contextlib.contextmanager
def atomic_file(
    dest: str | os.PathLike[str], mode: str = "w+b", **kwargs
) -> Generator[IO, None, None]:
    ...

with atomic_file("foo", "w") as f:
    assert_type(f, IO[str])
    "#,
);

testcase!(
    test_literalstring_or_str_overloads,
    r#"
from typing import Any, LiteralString, overload

class PathLike[T]: ...

def normpath[T](path: PathLike[T]) -> T: ...

@overload
def relpath(path: LiteralString) -> LiteralString: ...
@overload
def relpath(path: str) -> str: ...
def relpath(path) -> Any: ...

def f(path: Any, data: Any) -> dict[str, Any]:
    outputs = {}
    relative_normalized_path = relpath(normpath(path))
    outputs[relative_normalized_path] = data
    return outputs
    "#,
);

testcase!(
    test_one_overload_is_typeis,
    r#"
from typing import TypeIs, assert_type, overload

@overload
def f(x: str) -> str: ...
@overload
def f(x: int) -> TypeIs[bool]: ...
def f(x):
    if isinstance(x, str):
        return x
    else:
        return isinstance(x, bool)

def g(x: str, y: int):
    assert_type(f(x), str)
    assert_type(f(y), bool)
    if f(x):
        assert_type(x, str)
    if f(y):
        assert_type(y, bool)
    "#,
);

testcase!(
    test_filter_selects_typeis_overload,
    r#"
import ast
from typing import TypeIs, assert_type

type IsDef = ast.FunctionDef | ast.ClassDef

def is_def(n: object) -> TypeIs[IsDef]:
    return isinstance(n, ast.FunctionDef | ast.ClassDef)

def f(node: ast.ClassDef):
    for child in filter(is_def, node.body):
        assert_type(child, ast.ClassDef | ast.FunctionDef)
    "#,
);

testcase!(
    test_tuple_any_with_tuple_ambigious_overload,
    r#"
from typing import Any, Literal, Never, overload, assert_type

@overload
def ndim(shape: tuple[Never, ...]) -> int: ...
@overload
def ndim(shape: tuple[int]) -> Literal[1]: ...
@overload
def ndim(shape: tuple[int, int]) -> Literal[2]: ...
@overload
def ndim(shape: tuple[int, ...]) -> int: ...
def ndim(shape: tuple[int, ...]) -> int:
    return len(shape)

def demo_gradual(s: tuple[Any, ...]):
    assert_type(ndim(s), int)

def demo_one(s: tuple[int]):
    assert_type(ndim(s), Literal[1])

def demo_two(s: tuple[int, int]):
    assert_type(ndim(s), Literal[2])

def demo_variadic(s: tuple[int, ...]):
    assert_type(ndim(s), int)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2600.
testcase!(
    test_materialization_does_not_leak_into_partial_contained,
    r#"
def f(x: None):
    config = {}
    # Intentionally introduce Error types.
    for k, v in x:  # E: Type `None` is not iterable
        config.setdefault(k, v)
    for k in config:
        if k == "hello":
            pass
    "#,
);

testcase!(
    test_resolve_ambiguous_precise,
    r#"
from typing import Any, overload, assert_type

class A[T]:  # covariant
    def get(self) -> T: ...

@overload
def op(l: A[None], r: A[None]) -> A[None]: ...
@overload
def op(l: A[None], r: A[Any]) -> A[None]: ...
@overload
def op(l: A[Any], r: A[None]) -> A[None]: ...
@overload
def op(l: A[Any], r: A[Any]) -> A[Any]: ...
def op(l, r) -> A[None | Any]: ...

def test(x: A[None], y: A[Any]) -> None:
    assert_type(op(x, x), A[None])
    assert_type(op(x, y), A[None])
    assert_type(op(y, x), A[None])
    assert_type(op(y, y), A[Any])
    "#,
);

testcase!(
    test_resolve_ambiguous_spec_compliant,
    TestEnv::new().enable_spec_compliant_overloads(),
    r#"
from typing import Any, overload, assert_type

class A[T]:  # covariant
    def get(self) -> T: ...

@overload
def op(l: A[None], r: A[None]) -> A[None]: ...
@overload
def op(l: A[None], r: A[Any]) -> A[None]: ...
@overload
def op(l: A[Any], r: A[None]) -> A[None]: ...
@overload
def op(l: A[Any], r: A[Any]) -> A[Any]: ...
def op(l, r) -> A[None | Any]: ...

def test(x: A[None], y: A[Any]) -> None:
    assert_type(op(x, x), A[None])
    assert_type(op(x, y), A[None])
    assert_type(op(y, x), A[None])
    assert_type(op(y, y), Any)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2043
testcase!(
    test_overload_paramspec_unify_with_ellipsis_callable,
    r#"
from collections.abc import Callable
from typing import ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")

@overload
def foo(func: Callable[P, R]) -> Callable[P, R]: ...
@overload
def foo() -> None: ...

def foo(func: None | Callable[..., R] = None) -> None | Callable[..., R]:
    return func
    "#,
);

// Even though argument `x` has an unknown type, `f1(x)` and `f2(x)` are not considered ambiguous
// because parameter `x` has the same type in all candidate overloads.
testcase!(
    test_eliminate_non_ambiguous,
    r#"
from typing import assert_type, overload

@overload
def f1(x: str) -> str: ...
@overload
def f1(x: str, *args) -> int: ...
def f1(x, *args) -> str | int: ...

@overload
def f2(x: str) -> str: ...
@overload
def f2(x: str, **kwargs) -> int: ...
def f2(x, **kwargs) -> str | int: ...

def g(x):
    assert_type(f1(x), str)
    assert_type(f2(x), str)
    "#,
);

// In spec-compliant mode, because argument `x` has unknown type in `f1(x)` and `f2(x)`, the calls
// are considered ambiguous and both overloads match, so we fall back to `Any` for the return type.
// See test_eliminate_non_ambiguous for the non-spec-compliant behavior.
testcase!(
    test_eliminate_non_ambiguous_spec_compliant,
    TestEnv::new().enable_spec_compliant_overloads(),
    r#"
from typing import Any, assert_type, overload

@overload
def f1(x: str) -> str: ...
@overload
def f1(x: str, *args) -> int: ...
def f1(x, *args) -> str | int: ...

@overload
def f2(x: str) -> str: ...
@overload
def f2(x: str, **kwargs) -> int: ...
def f2(x, **kwargs) -> str | int: ...

def g(x):
    assert_type(f1(x), Any)
    assert_type(f2(x), Any)
    "#,
);

testcase!(
    test_do_not_eliminate_variadic,
    r#"
from typing import Any, assert_type, overload

@overload
def f(x: str) -> str: ...
@overload
def f(x: str | bytes, *args: int) -> int: ...
def f(x: str | bytes, *args: int) -> str | int:
    return 0

def g(x):
    assert_type(f(x), Any)
    "#,
);

testcase!(
    test_eliminate_overload_using_argument_count_even_with_error,
    r#"
from typing import assert_type, overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: int, y: str) -> str: ...
def f(x, y="") -> int | str: ...

def g(x: int | None):
    # We should report the mismatch between `int` and `int | None` rather than "No matching overload",
    # since we know only the first overload can match based on argument count.
    e = f(x)  # E: `int | None` is not assignable to parameter `x` with type `int`
    # Even though the call failed, we know that the second overload cannot match based on argument
    # count, so we should use the return type from the first overload.
    assert_type(e, int)
    "#,
);

testcase!(
    test_match_overload_using_literal_arg,
    r#"
from typing import Any, assert_type, Literal, overload

@overload
def f(x: int, y: Literal['y']) -> int: ...
@overload
def f(x: int, y: Literal['n']) -> str: ...
@overload
def f(x: int, y: str) -> int | str: ...
def f(x, y) -> int | str: ...

def g(x: Any):
    assert_type(f(x, 'y'), int)
    "#,
);

// In non-spec-compliant mode (see test_match_overload_using_literal_arg), step 5 skips
// materializing arguments whose parameter type is the same across all candidate overloads, so `x`
// would not be materialized, and the first overload would win. In spec-compliant mode, `x` is
// materialized, and the call is ambiguous.
testcase!(
    test_match_overload_using_literal_arg_spec_compliant,
    TestEnv::new().enable_spec_compliant_overloads(),
    r#"
from typing import Any, assert_type, Literal, overload

@overload
def f(x: int, y: Literal['y']) -> int: ...
@overload
def f(x: int, y: Literal['n']) -> str: ...
@overload
def f(x: int, y: str) -> int | str: ...
def f(x, y) -> int | str: ...

def g(x: Any):
    assert_type(f(x, 'y'), Any)
    "#,
);

testcase!(
    test_nested_call_to_generic_overload,
    r#"
from typing import Literal, overload

def check[T](actual: T) -> T: ...

class Kitten: ...
class Puppy: ...

@overload
def read_csv[S](
    filepath_or_buffer: str,
    *,
    iterator: Literal[True],
) -> Puppy: ...
@overload
def read_csv(
    filepath_or_buffer: str,
    *,
    iterator: Literal[False],
) -> Kitten: ...
def read_csv(*args, **kwargs) -> Puppy | Kitten: ...

def test_types_csv(path: str) -> None:
    check(read_csv(path, iterator=False))
    "#,
);

testcase!(
    test_partial_inference_through_overload,
    r#"
from typing import assert_type, overload
x = []
y = "y"
@overload
def f(x: list[int], y: int): ...
@overload
def f(x: list[str], y: str): ...
def f(x, y): ...
f(x, y)
assert_type(x, list[str])
    "#,
);

testcase!(
    test_round_after_narrow,
    r#"
def test(value: int | float | None) -> str:
    if value is None:
        return ''
    value = round(value, 4)
    i = int(value)
    return str(i)
    "#,
);

testcase!(
    test_match_overload_with_unknown_type_from_missing_import,
    r#"
from typing import Any, assert_type, overload, TypeVar, TypeAliasType
from collections.abc import Sequence
import nonexistent as nmod  # type: ignore

T = TypeVar("T")

Opaque = TypeAliasType("Opaque", nmod.Foo[T], type_params=(T,))
MyType = TypeAliasType("MyType", Opaque[T] | Sequence[T], type_params=(T,))

class Inexact: ...
S = TypeVar("S", bound=Inexact)

@overload
def f(a: MyType[S]) -> S: ...
@overload
def f(a: MyType[int]) -> float: ...
def f(a: object) -> object: ...

x: list[int] = []
# This agrees with pyright and ty. (Mypy says `Never`.)
# Because of `Opaque`, we should match the first overload with `S` unsolved.
assert_type(f(x), Any)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/3161
testcase!(
    test_overload_unpacked_tuple_varargs,
    r#"
from typing import overload, assert_type

@overload
def f(*args: *tuple[int]) -> int: ...
@overload
def f(*args: *tuple[int, int]) -> tuple[int, int]: ...
def f(*args) -> int | tuple[int, int]:
    return 1

assert_type(f(1), int)
assert_type(f(1, 2), tuple[int, int])
    "#,
);

testcase!(
    test_reject_overload_with_specialization_error,
    r#"
from typing import overload

@overload
def f[T: str](x: T) -> T: ...
@overload
def f(x: int) -> int: ...
def f(x):
    return x

def g(x: float):
    f(x)  # E: No matching overload
    "#,
);

testcase!(
    test_callback_protocol_with_overloads_and_bounded_typevar,
    r#"
from typing import Callable, Protocol, overload

class Base: ...

class HasCall(Protocol):
    @overload
    def __call__[T: Base](self, arg: T) -> T: ...
    @overload
    def __call__(self, arg: float) -> float: ...

def takes(f: Callable[[float], float]) -> None: ...

def repro(p: HasCall):
    takes(p)
    "#,
);

testcase!(
    test_overload_unpack_kwargs_with_explicit_impl_params,
    r#"
from typing import Literal, Protocol, Required, TypedDict, TypeVar, Unpack, overload

class A: ...
class B: ...
class C(Protocol):
    def member(self) -> str: ...

T = TypeVar("T")

class ExcludeC(TypedDict, closed=True, total=False):
    pass_through: bool
    only_a: bool
    only_c: Literal[False]
    include_c: Literal[False] | None

class OnlyC(TypedDict, closed=True, total=False):
    pass_through: bool
    only_a: bool
    only_c: Required[Literal[True]]
    include_c: bool | None

class IncludeC(TypedDict, closed=True, total=False):
    pass_through: bool
    only_a: bool
    only_c: Literal[False]
    include_c: Required[Literal[True]]

class IncludeB(TypedDict, closed=True, total=False):
    pass_through: bool
    only_a: Literal[False]
    only_c: Literal[False]
    include_c: bool | None

class IncludeABC(TypedDict, closed=True, total=False):
    pass_through: bool
    only_a: Literal[False]
    only_c: Literal[False]
    include_c: Required[Literal[True]]

class PassThrough(TypedDict, closed=True, total=False):
    pass_through: Required[Literal[True]]
    only_a: bool
    only_c: bool
    include_c: bool | None

# This typed dict is open, so it may contain unknown keys and is not safe to match
# against an implementation signature without **kwargs
class PassThroughOpen(TypedDict, total=False):
    pass_through: Required[Literal[True]]
    only_a: bool
    only_c: bool
    include_c: bool | None

@overload
def func(obj: A, **kwds: Unpack[ExcludeC]) -> A: ...
@overload
def func(obj: C, **kwds: Unpack[OnlyC]) -> C: ...
@overload
def func(obj: C, **kwds: Unpack[IncludeC]) -> C: ...
@overload
def func(obj: B, **kwds: Unpack[IncludeB]) -> B: ...
@overload
def func(obj: A | C, **kwds: Unpack[IncludeC]) -> A | C: ...
@overload
def func(obj: A | B | C, **kwds: Unpack[IncludeABC]) -> A | B | C: ...
@overload
def func(obj: T, **kwds: Unpack[PassThrough]) -> T: ...
@overload
def func(obj: T, **kwds: Unpack[PassThroughOpen]) -> T: ... # E:
def func(
    obj: A | B | C | T,
    *,
    pass_through: bool = False,
    only_a: bool = False,
    only_c: bool = False,
    include_c: bool | None = None,
) -> A | B | C | T:
    raise NotImplementedError
    "#,
);

testcase!(
    test_overload_error_shows_relevant_signature_part,
    r#"
from typing import overload

@overload
def f(*args: int, **kwargs: str) -> int: ...
@overload
def f(*args: str, **kwargs: int) -> str: ...
def f(*args, **kwargs):
    return args[0]

f(4.2)  # E: (*args: int, ...) -> int [closest match]\n    (*args: str, ...) -> str
f(x=4.2)  # E: (..., **kwargs: str) -> int [closest match]\n    (..., **kwargs: int) -> str

# When any arguments are unpacked, we conservatively fall back to showing full signatures
f(*[4.2])  # E: (*args: int, **kwargs: str) -> int [closest match]\n    (*args: str, **kwargs: int) -> str
f(**{"x": 4.2})  # E: (*args: int, **kwargs: str) -> int [closest match]\n    (*args: str, **kwargs: int) -> str
    "#,
);

testcase!(
    test_overload_error_shows_unpacked_kwargs,
    r#"
from typing import overload, TypedDict, Unpack

class TD(TypedDict):
    y: int

@overload
def f(x: int = ..., **kwargs: Unpack[TD]) -> int: ...
@overload
def f(x: str = ..., **kwargs: Unpack[TD]) -> str: ...
def f(x=0, **kwargs):
    return x

f(y=4.2)  # E: (..., **kwargs: Unpack[TD]) -> int [closest match]\n    (..., **kwargs: Unpack[TD]) -> str
    "#,
);

testcase!(
    test_overload_error_does_not_truncate_on_different_param_names,
    r#"
from typing import overload

@overload
def f(x: int, /) -> int: ...
@overload
def f(y: str, /) -> str: ...
def f(x, /): return x

# Make sure we show 'y' in the second overload even though 'x' was matched in the first overload
f(4.2)  # E: (x: int, /) -> int [closest match]\n    (y: str, /) -> str
    "#,
);

testcase!(
    test_overload_error_truncates_method_signatures,
    r#"
from typing import overload

class A:
    @overload
    def f(self, x: str, y: int = ...) -> str: ...
    @overload
    def f(self, x: int, y: str = ...) -> int: ...
    def f(self, x, y=0): return x

A().f(4.2)  # E: (x: str, ...) -> str [closest match]\n    (x: int, ...) -> int
    "#,
);

testcase!(
    test_overload_error_shows_missing_parameter,
    r#"
from typing import Any, overload

@overload
def f(x: int, y: str = ..., z: float = ...) -> int: ...
@overload
def f(x: str, y: int = ..., z: float = ...) -> str: ...
def f(x, y="", z=0.0): return x

y: Any = ...
f(y=y)  # E: (x: int, y: str = ..., ...) -> int [closest match]\n    (x: str, y: int = ..., ...) -> str
    "#,
);

testcase!(
    test_arg_error_isolation,
    r#"
from typing import Callable, Literal, assert_type, overload

@overload
def f(tag: Literal[1], xs: list[Callable[[int], str]]) -> int: ...

@overload
def f(tag: Literal[2], xs: list[Callable[[str], str]]) -> str: ...

def f(tag: Literal[1, 2], xs: list[Callable[..., str]]) -> int | str:
    return 0 if tag == 1 else ""

assert_type(f(2, [lambda value: value + "x"]), str)
    "#,
);
