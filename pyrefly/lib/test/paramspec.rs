/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Many of these tests come from <https://typing.readthedocs.io/en/latest/spec/generics.html#paramspec>.

use crate::testcase;

testcase!(
    test_callable_param_spec,
    r#"
from typing import Callable, ParamSpec
P = ParamSpec("P")
def test(f: Callable[P, None]) -> Callable[P, None]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> None:
        f(*args, **kwargs)
    return inner
"#,
);

testcase!(
    test_param_spec_solve,
    r#"
from typing import Callable, ParamSpec, Concatenate, assert_type
def f1[**P](x: Callable[P, int]) -> Callable[P, int]: ...
def f2[**P](x: Callable[Concatenate[int, P], int]) -> Callable[P, int]: ...
def f3[**P](x: Callable[P, int]) -> Callable[Concatenate[int, P], int]: ...
def f4[**P](x: Callable[Concatenate[int, P], int]) -> Callable[Concatenate[str, P], int]: ...

def test(x1: Callable[[int, str], int], x2: Callable[..., int]):
    assert_type(f1(x1), Callable[[int, str], int])
    assert_type(f1(x2), Callable[..., int])

    assert_type(f2(x1), Callable[[str], int])
    assert_type(f2(x2), Callable[..., int])

    assert_type(f3(x1), Callable[[int, int, str], int])
    assert_type(f3(x2), Callable[Concatenate[int, ...], int])

    assert_type(f4(x1), Callable[[str, str], int])
    assert_type(f4(x2), Callable[Concatenate[str, ...], int])
"#,
);

testcase!(
    test_paramspec_middleware_kwargs_mismatch,
    r#"
from typing import Protocol, Any

class ASGIApp:
    pass

class _MiddlewareFactory[**P](Protocol):
    def __call__(self, app: ASGIApp, /, *args: P.args, **kwargs: P.kwargs) -> Any: ...

class FastAPI:
    def add_middleware[**P](
        self,
        middleware_class: _MiddlewareFactory[P],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        pass

class CORSMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        pass


def use_middleware() -> None:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        x="",  # E: Unexpected keyword argument `x`
    )
"#,
);

testcase!(
    test_param_spec_generic_function,
    r#"
from typing import Callable, reveal_type
def identity[**P, R](x: Callable[P, R]) -> Callable[P, R]:
    return x
def foo[T](x: T, y: T) -> T:
    return x
foo2 = identity(foo)
reveal_type(foo2)  # E: revealed type: (x: @_, y: @_) -> @_
"#,
);

testcase!(
    bug = "Generic class constructors don't work with ParamSpec",
    test_param_spec_generic_constructor,
    r#"
from typing import Callable, reveal_type
def identity[**P, R](x: Callable[P, R]) -> Callable[P, R]:
  return x
class C[T]:
  x: T
  def __init__(self, x: T) -> None:
    self.x = x
c2 = identity(C)
reveal_type(c2)  # E: revealed type: (x: Unknown) -> C[Unknown]
x: C[int] = c2(1)
"#,
);

testcase!(
    test_param_spec_invariance,
    r#"
from typing import Callable, reveal_type
def identity[**P, R](x: Callable[P, R]) -> Callable[P, R]:
  return x
def test(f1: Callable[[int], None] | Callable[[str], None]):
  f3 = identity(f1)  # E: Argument `((int) -> None) | ((str) -> None)` is not assignable to parameter `x` with type `(int) -> None` in function `identity`
"#,
);

testcase!(
    test_function_concatenate,
    r#"
from typing import Callable, ParamSpec, Concatenate
P = ParamSpec("P")
def f(t: Callable[Concatenate[int, P], int]):
    pass
"#,
);

testcase!(
    test_simple_paramspec,
    r#"
from typing import Callable, ParamSpec, assert_type, reveal_type

P = ParamSpec("P")

def changes_return_type_to_str(x: Callable[P, int]) -> Callable[P, str]: ...

def returns_int(a: str, b: bool) -> int: ...

f = changes_return_type_to_str(returns_int)
reveal_type(f)  # E: revealed type: (a: str, b: bool) -> str

f("A", True)
f(a="A", b=True)
f("A", "A")  # E: Argument `Literal['A']` is not assignable to parameter `b` with type `bool`
assert_type(f("A", True), str)
"#,
);

testcase!(
    test_paramspec_in_right_place,
    r#"
from typing import Callable, Concatenate, ParamSpec

P = ParamSpec("P")

def foo(x: P) -> P: ...                           # E: `ParamSpec` is not allowed in this context # E: `ParamSpec` is not allowed in this context
def foo(x: Concatenate[int, P]) -> int: ...       # E: `Concatenate[int, P]` is not allowed in this context
def foo(x: Callable[Concatenate[P, P], int]) -> int: ...  # E: `ParamSpec` is not allowed in this context
def foo(x: list[P]) -> None: ...                  # E: `ParamSpec` cannot be used for type parameter
def foo(x: Callable[[int, str], P]) -> None: ...  # E: `ParamSpec` is not allowed in this context
def foo(x: Callable[[P, str], int]) -> None: ...  # E: `ParamSpec` is not allowed in this context
"#,
);

testcase!(
    test_paramspec_generic,
    r#"
from typing import Callable, Concatenate, ParamSpec, Generic, TypeVar

T = TypeVar("T")
P = ParamSpec("P")
P_2 = ParamSpec("P_2")

class X(Generic[T, P]):
  f: Callable[P, int]
  x: T

def f1(x: X[int, P_2]) -> str: ...                    # Accepted
def f2(x: X[int, Concatenate[int, P_2]]) -> str: ...  # Accepted
def f3(x: X[int, [int, bool]]) -> str: ...            # Accepted
def f4(x: X[int, ...]) -> str: ...                    # Accepted
def f5(x: X[int, int]) -> str: ...                    # E: Expected a valid ParamSpec expression

class X2[T, **P]:
  f: Callable[P, int]
  x: T

def f6(x: X2[int, [int, bool]]) -> str: ...           # Accepted
"#,
);

testcase!(
    test_paramspec_omit_brackets,
    r#"
from typing import Callable, Generic, ParamSpec

P = ParamSpec("P")

class Z(Generic[P]):
  f: Callable[P, int]

def f(x: Z[[int, str, bool]]) -> str: ...
def f(x: Z[int, str, bool]) -> str: ...
"#,
);

testcase!(
    test_paramspec_repeated,
    r#"
from typing import Callable, ParamSpec, reveal_type

P = ParamSpec("P")

def foo(x: Callable[P, int], y: Callable[P, int]) -> Callable[P, bool]: ...

def x_y(x: int, y: str) -> int: ...
def y_x(y: int, x: str) -> int: ...

reveal_type(foo(x_y, x_y)) # E: revealed type: (x: int, y: str) -> bool
               # Should return (x: int, y: str) -> bool
               # (a callable with two positional-or-keyword parameters)

foo(x_y, y_x) # E: Argument `(y: int, x: str) -> int` is not assignable to parameter `y` with type `(x: int, y: str) -> int`
               # Could return (a: int, b: str, /) -> bool
               # (a callable with two positional-only parameters)
               # This works because both callables have types that are
               # behavioral subtypes of Callable[[int, str], int].
               # Could also fail, which is what we do.

def keyword_only_x(*, x: int) -> int: ...
def keyword_only_y(*, y: int) -> int: ...
foo(keyword_only_x, keyword_only_y) # Rejected # E: Argument `(*, y: int) -> int` is not assignable to parameter `y` with type `(*, x: int) -> int`
"#,
);

testcase!(
    test_paramspec_constructor,
    r#"
from typing import TypeVar, Generic, Callable, ParamSpec, reveal_type

P = ParamSpec("P")
U = TypeVar("U")

class Y(Generic[U, P]):
  f: Callable[P, str]
  prop: U

  def __init__(self, f: Callable[P, str], prop: U) -> None:
    self.f = f
    self.prop = prop

def a(q: int) -> str: ...

reveal_type(Y(a, 1))   # E: revealed type: Y[int, [q: int]]
reveal_type(Y(a, 1).f) # E: revealed type: (q: int) -> str
"#,
);

// We have different formatting to what the spec suggests, but the same answers.
testcase!(
    test_simple_concatenate,
    r#"
from typing import Callable, Concatenate, ParamSpec, reveal_type

P = ParamSpec("P")

def bar(x: int, *args: bool) -> int: ...
def add(x: Callable[P, int]) -> Callable[Concatenate[str, P], bool]: ...
reveal_type(add(bar))       # Should return (a: str, /, x: int, *args: bool) -> bool # E: revealed type: (str, x: int, *args: bool) -> bool

def remove(x: Callable[Concatenate[int, P], int]) -> Callable[P, bool]: ...
reveal_type(remove(bar))    # E: revealed type: (*args: bool) -> bool

def transform(
  x: Callable[Concatenate[int, P], int]
) -> Callable[Concatenate[str, P], bool]: ...
reveal_type(transform(bar)) # Should return (a: str, /, *args: bool) -> bool # E: revealed type: (str, *args: bool) -> bool
"#,
);

testcase!(
    bug = "P.args and P.kwargs should only work when P is in scope",
    test_paramspec_component_usage,
    r#"
from typing import Callable, ParamSpec

P = ParamSpec("P")

def puts_p_into_scope(f: Callable[P, int]) -> None:
  def inner(*args: P.args, **kwargs: P.kwargs) -> None: pass     # Accepted
  def mixed_up(*args: P.kwargs, **kwargs: P.args) -> None: pass  # E: `ParamSpec` **kwargs is only allowed in a **kwargs annotation # E: `ParamSpec` *args is only allowed in an *args annotation
  def misplaced(x: P.args) -> None: pass                         # E: `ParamSpec` *args is only allowed in an *args annotation

def out_of_scope(*args: P.args, **kwargs: P.kwargs) -> None: # Rejected
  pass
"#,
);

testcase!(
    test_paramspec_together,
    r#"
from typing import Callable, ParamSpec

P = ParamSpec("P")

def puts_p_into_scope(f: Callable[P, int]) -> None:
  stored_args: P.args                           # E: `ParamSpec` *args is only allowed in an *args annotation
  stored_kwargs: P.kwargs                       # E: `ParamSpec` **kwargs is only allowed in a **kwargs annotation
  def just_args(*args: P.args) -> None:         # E: `ParamSpec` *args and **kwargs must be used together
    pass
  def just_kwargs(**kwargs: P.kwargs) -> None:  # E: `ParamSpec` *args and **kwargs must be used together
    pass
"#,
);

testcase!(
    test_paramspec_decorator,
    r#"
from typing import Callable, ParamSpec, assert_type, Any

P = ParamSpec("P")

def decorator(f: Callable[P, int]) -> Callable[P, None]:
  def foo(*args: P.args, **kwargs: P.kwargs) -> None:
    assert_type(f(*args, **kwargs), int)    # Accepted, should resolve to int
    f(*kwargs, **args)    # Rejected # E: Expected *-unpacked P.args and **-unpacked P.kwargs
    f(1, *args, **kwargs) # Rejected # E: Expected 0 positional arguments, got 1

  # Allow tuple[Any, ...] to match P.args, and dict[str, Any] to match P.kwargs
  def bar(t1: tuple[Any, ...], t2: tuple[object, ...], d1: dict[str, Any], d2: dict[str, object]) -> None:
    f(*t1, **d1) # OK
    f(*t2, **d2) # # E: Expected *-unpacked P.args and **-unpacked P.kwargs
  return foo              # Accepted
"#,
);

testcase!(
    test_paramspec_args_kwargs_subscriptable,
    r#"
from typing import Callable, ParamSpec, TypeVar, assert_type

FP = ParamSpec('FP')
FR = TypeVar('FR')

def decorate(f: Callable[FP, FR]) -> Callable[FP, FR]:
  def wrapper(*args: FP.args, **kwargs: FP.kwargs) -> FR:
    if len(args) > 0:
      assert_type(args[0], object)
    if 'foo' in kwargs:
      assert_type(kwargs['foo'], object)
    return f(*args, **kwargs)
  return wrapper
"#,
);

testcase!(
    test_paramspec_named_arguments_concatenate,
    r#"
from typing import Callable, ParamSpec

P = ParamSpec("P")

def outer(f: Callable[P, None]) -> Callable[P, None]:
  def foo(x: int, *args: P.args, **kwargs: P.kwargs) -> None:
    f(*args, **kwargs)

  def bar(*args: P.args, **kwargs: P.kwargs) -> None:
    foo(1, *args, **kwargs)   # Accepted
    foo(x=1, *args, **kwargs) # Rejected # E: Expected 1 more positional argument # E: Unexpected keyword argument `x`

  return bar
"#,
);

testcase!(
    test_paramspec_different_origins,
    r#"
from typing import Callable, ParamSpec

P1 = ParamSpec("P1")
P2 = ParamSpec("P2")

def foo(x: int, *args: P1.args, **kwargs: P2.kwargs) -> None: ...  # E: *args and **kwargs must come from the same `ParamSpec`
"#,
);

testcase!(
    test_paramspec_twice,
    r#"
from typing import Callable, ParamSpec

P = ParamSpec("P")

def twice(f: Callable[P, int], *args: P.args, **kwargs: P.kwargs) -> int:
  return f(*args, **kwargs) + f(*args, **kwargs)

def a_int_b_str(a: int, b: str) -> int:
  return a

twice(a_int_b_str, 1, "A")     # Accepted

twice(a_int_b_str, b="A", a=1) # Accepted

twice(a_int_b_str, "A", 1)     # Rejected # E: `Literal['A']` is not assignable to parameter `a` with type `int` # E: `Literal[1]` is not assignable to parameter `b` with type `str`
"#,
);

testcase!(
    test_paramspec_callable_infer_ellipsis,
    r#"
from typing import Callable, ParamSpec

P = ParamSpec("P")
def f(f: Callable[P, int], *args: P.args, **kwargs: P.kwargs) -> int:
  return f(*args, **kwargs) + f(*args, **kwargs)
x: Callable[..., int] = lambda: 1

f(x, 1)"#,
);

testcase!(
    bug = "we can pass in anything for the ParamSpec component of the concatenation",
    test_paramspec_callable_infer_concatenate,
    r#"
from typing import Callable, ParamSpec, Concatenate

P = ParamSpec("P")
def f(f: Callable[P, int], *args: P.args, **kwargs: P.kwargs) -> int:
    return f(*args, **kwargs)
P2 = ParamSpec("P2")
def g(x: Callable[Concatenate[int, P2], int], *args: P2.args, **kwargs: P2.kwargs):
    f(x, 1)
    f(x)  # E: Expected 1 more positional argument in function `f`
    f(x, 1, 2)  # Not OK, we aren't sure the 2nd param is an int
"#,
);

testcase!(
    test_functools_wraps_paramspec,
    r#"
from functools import wraps

def f(fn):
    @wraps(fn)
    def wrapped_fn(x):
        return fn(x)

    return wrapped_fn
"#,
);

testcase!(
    test_bad_paramspec_in_concatenate,
    r#"
from typing import Callable, Concatenate

X = Callable[Concatenate[int, "oops"], int]  # E: Expected a `ParamSpec`  # E: Could not find name `oops`

def f(x: X, y):
    x(0)
  "#,
);

testcase!(
    test_in_alias,
    r#"
from typing import Callable, ParamSpec, TypeAlias
P = ParamSpec("P")
TA: TypeAlias = Callable[P, None]
def f(f: TA):
    # f can be called with anything
    f()
    f("")
    f(1, 2, 3, 4, 5)
    "#,
);

testcase!(
    test_in_try_except,
    r#"
from collections.abc import Callable
def run[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    while True:
        try:
            return func(*args,**kwargs)
        except Exception:
            pass
    raise NotImplementedError("Unreachable")
  "#,
);

testcase!(
    test_param_spec_component_subtype,
    r#"
def test[**P](*args: P.args, **kwargs: P.kwargs) -> None:
  x: tuple[object, ...] = args
  y: dict[str, object] = kwargs
"#,
);

testcase!(
    test_param_spec_component_iter,
    r#"
from typing import *

R = TypeVar("R")
P = ParamSpec("P")

def wrap_any(f, *args, **kwargs):
    return f(*args, **kwargs)

def wrap(f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    return wrap_any(f, *args, **kwargs)
"#,
);

testcase!(
    test_param_spec_ellipsis,
    r#"
from typing import Callable
def test[**P](v: Callable[P, None]):
    a: Callable[..., None] = v
    b: Callable[P, None] = a
"#,
);

testcase!(
    test_param_spec_empty,
    r#"
def foo[**P](x: int = 0, *args: P.args, **kwargs: P.kwargs):
    pass
foo()
"#,
);

testcase!(
    test_param_spec_optional_prefix_arg,
    r#"
from typing import Concatenate, ParamSpec, TypeVar, Callable

class T:
    def c1(self) -> None:
        pass

Res = TypeVar("Res", covariant=True)
P = ParamSpec("P")

def caller(
    vals: list[T],
    action: Callable[Concatenate[T, P], Res],
    extra_arg: bool=True,
    *args: P.args,
    **kwargs: P.kwargs
) -> None:
    for val in vals:
        if extra_arg:
            _ = action(val, *args, **kwargs)

caller([], T.c1)
"#,
);
