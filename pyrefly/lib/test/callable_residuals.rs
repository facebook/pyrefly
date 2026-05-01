/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Callable residuals represent type parameters of higher order functions
 * that "capture" complex structure in an argument that is a callable
 *  type, when that callable is generic or overloaded.
 *
 * The resulting "residual" types can potentially live in the TArgs
 * of a class (which lets us model things like callback protocols, where
 * a class might capture the full callable structure of an input function).
 *
 * A residual always has a "fallback" behavior where it flattens to an approximate
 * type, but if it appears in a Callable type then we will try to "explode" the
 * generic and/or overload structure implied by residual types.
 *
 * This module is for tests that touch this residual-type and exploding-callable
 * behavior.
 */

use crate::testcase;

// Make sure no residual type leaks into user output, when a residual
// winds up directly in a return type
testcase!(
    test_no_projection_leak_in_reveal_type,
    r#"
from typing import Callable, reveal_type
def identity[**P, R](x: Callable[P, R]) -> tuple[Callable[P, R], R]:
    ...
def foo[T](x: T) -> T:
    return x
f_out, r_out = identity(foo)
reveal_type(f_out)  # E: revealed type: [R](x: R) -> R
reveal_type(r_out)  # E: revealed type: Unknown
"#,
);

testcase!(
    test_callback_protocol_generic_call,
    r#"
from typing import Callable, Protocol, reveal_type

class GenericCallback(Protocol):
    def __call__[T](self, x: T) -> T: ...

def identity[A, R](f: Callable[[A], R]) -> Callable[[A], R]:
    return f

def use_it(cb: GenericCallback) -> None:
    result = identity(cb)
    reveal_type(result)  # E: revealed type: [R](R) -> R
"#,
);

testcase!(
    test_simple_generic_residual,
    r#"
from typing import Callable, reveal_type
def identity[S](x: Callable[[S], S]) -> Callable[[S], S]:
    return x
def generic_fn[T](x: T) -> T:
    return x
result = identity(generic_fn)
reveal_type(result)  # E: revealed type: [T](T) -> T
"#,
);

testcase!(
    test_two_tparam_generic_residual,
    r#"
from typing import Callable, reveal_type
def simple_identity[A, R](f: Callable[[A], R]) -> Callable[[A], R]:
    return f
def generic_fn[T](x: T) -> T: ...
result = simple_identity(generic_fn)
reveal_type(result)  # E: revealed type: [R](R) -> R
"#,
);

testcase!(
    test_add_prefix_generic,
    r#"
from typing import Callable, reveal_type
def add_prefix[A, R](f: Callable[[A], R]) -> Callable[[int, A], R]: ...
def identity_fn[T](x: T) -> T: ...
result = add_prefix(identity_fn)
reveal_type(result)  # E: revealed type: [R](int, R) -> R
"#,
);

testcase!(
    test_generic_residual_concrete_return,
    r#"
from typing import Callable, reveal_type
def higher_order[A, B](x: Callable[[A, B], int]) -> Callable[[A, B], int]:
    return x
def generic_fn[T](x: T, y: T) -> int:
    return 0
result = higher_order(generic_fn)
reveal_type(result)  # E: revealed type: [T](T, T) -> int
"#,
);

testcase!(
    test_generic_residual_distinct_positions,
    r#"
from typing import Callable, reveal_type
def higher_order[A, B](x: Callable[[A, B], B]) -> Callable[[A, B], B]:
    return x
def generic_fn[T, S](x: S, y: T) -> T:
    return y
result = higher_order(generic_fn)
reveal_type(result)  # E: revealed type: [T, S](S, T) -> T
"#,
);

testcase!(
    test_generic_residual_nested_pattern_inner_var,
    r#"
from typing import Callable, reveal_type
def higher_order[A](x: Callable[[list[A]], list[A]]) -> Callable[[list[A]], list[A]]:
    return x
def generic_fn[T](x: T) -> T:
    return x
result = higher_order(generic_fn)
reveal_type(result)  # E: revealed type: [A](list[A]) -> list[A]
"#,
);

testcase!(
    test_generic_residual_nested_source_inner_var,
    r#"
from typing import Callable, reveal_type
def higher_order[A](x: Callable[[A], A]) -> Callable[[A], A]:
    return x
def generic_fn[T](x: list[T]) -> list[T]:
    return x
result = higher_order(generic_fn)
reveal_type(result)  # E: revealed type: [T](list[T]) -> list[T]
"#,
);

testcase!(
    bug = "Generic callback protocols with extra type params degrade callable precision to Any",
    test_callback_protocol_phantom_target_var,
    r#"
from typing import Protocol, Callable, reveal_type

class Callback[In, Out, Phantom](Protocol):
    def __call__(self, x: In) -> Out: ...

def lift[In, Out, Phantom](f: Callback[In, Out, Phantom]) -> tuple[Callable[[In], Out], Phantom]:
    ...

def id_fn[T](x: T) -> T: ...

out_f, out_p = lift(id_fn)
reveal_type(out_f)  # E: revealed type: (Any) -> Any
reveal_type(out_p)  # E: revealed type: Unknown
"#,
);

testcase!(
    test_polarity_canary_protocol_in_negative_slot,
    r#"
from typing import Callable, Protocol, reveal_type

class PolyCb(Protocol):
    def __call__[T](self, x: T) -> T: ...

def choose[A](f: Callable[[PolyCb], A]) -> A:
    ...

def id_cb[X](cb: Callable[[X], X]) -> Callable[[X], X]:
    return cb

out = choose(id_cb)
reveal_type(out)  # E: revealed type: (Unknown) -> Unknown

def bad(cb: Callable[[int], str]) -> int:
    return 0

out2 = choose(bad)  # E: Argument `(cb: (int) -> str) -> int` is not assignable to parameter `f` with type `(PolyCb) -> @_` in function `choose`
"#,
);

testcase!(
    test_type_var_tuple_hof_against_concrete_tuple_with_generic_param,
    r#"
from typing import Callable, reveal_type
def higher_order[*Ts](x: Callable[[tuple[*Ts]], tuple[*Ts]]) -> Callable[[tuple[*Ts]], tuple[*Ts]]:
    return x
def generic_fn[T](x: tuple[int, T]) -> tuple[int, T]:
    return x
result = higher_order(generic_fn)
reveal_type(result)  # E: revealed type: [T](tuple[int, T]) -> tuple[int, T]
"#,
);

testcase!(
    test_type_var_tuple_generic_argument_against_concrete_tuple_hof,
    r#"
from typing import Callable, reveal_type
def higher_order[A, B](x: Callable[[tuple[A, B]], tuple[A, B]]) -> Callable[[tuple[A, B]], tuple[A, B]]:
    return x
def generic_fn[*Ts](x: tuple[*Ts]) -> tuple[*Ts]:
    return x
result = higher_order(generic_fn)
reveal_type(result)  # E: revealed type: [A, B](tuple[A, B]) -> tuple[A, B]
"#,
);

testcase!(
    test_type_var_tuple_identity_of_identity,
    r#"
from typing import Callable, reveal_type
def identity_tuple[*Ts, R](x: Callable[[*Ts], R]) -> Callable[[*Ts], R]:
    return x
result = identity_tuple(identity_tuple)
reveal_type(result)  # E: revealed type: [Ts, R](**tuple[(**tuple[*Ts]) -> R]) -> (**tuple[*Ts]) -> R
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
reveal_type(foo2)  # E: revealed type: [R](x: R, y: R) -> R
"#,
);

testcase!(
    test_param_spec_identity_of_identity,
    r#"
from typing import Callable, reveal_type
def identity[**P, T](x: Callable[P, T]) -> Callable[P, T]:
    return x
result = identity(identity)
reveal_type(result)  # E: revealed type: [P, T](x: (ParamSpec(P)) -> T) -> (ParamSpec(P)) -> T
"#,
);

testcase!(
    test_param_spec_identity_of_identity_behavior,
    r#"
from typing import Callable, reveal_type
def identity[**P, T](x: Callable[P, T]) -> Callable[P, T]:
    return x
def f(x: int, y: str) -> str:
    return y
result = identity(identity)
lifted = result(f)
reveal_type(lifted)  # E: revealed type: (x: int, y: str) -> str
out = lifted(1, "ok")
reveal_type(out)  # E: revealed type: str
"#,
);

testcase!(
    test_paramspec_wrap_generic_return,
    r#"
from typing import Callable, Awaitable, reveal_type
def wrap[**P, T](f: Callable[P, T]) -> Callable[P, Awaitable[T]]: ...
def identity_fn[X](x: X) -> X: ...

result = wrap(identity_fn)
reveal_type(result)  # E: revealed type: [T](x: T) -> Awaitable[T]
"#,
);

testcase!(
    test_concatenate_strip_first,
    r#"
from typing import Callable, Concatenate, Any, reveal_type
def strip_first[**P, T](
    f: Callable[Concatenate[Any, P], T]
) -> Callable[P, T]: ...
def two_arg[S](x: int, y: S) -> S: ...
result = strip_first(two_arg)
reveal_type(result)  # E: revealed type: [T](y: T) -> T
"#,
);

testcase!(
    test_typevar_class_field_projection_parity,
    r#"
from typing import Callable, reveal_type

class Box[T]:
    fn: Callable[[T], T]
    def __init__(self, fn: Callable[[T], T]) -> None:
        self.fn = fn

def f[S](x: S) -> S: ...
b = Box(f)
reveal_type(b.fn)  # E: revealed type: [S](S) -> S
called = b.fn(1)
reveal_type(called)  # E: revealed type: int
"#,
);

testcase!(
    test_callable_class_wrapper,
    r#"
from typing import Callable, reveal_type

class Wrapper[**P, R]:
    fn: Callable[P, R]
    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

def f[S](x: S) -> S: ...
wrapper = Wrapper(f)
reveal_type(wrapper.fn)  # E: revealed type: [R](x: R) -> R
reveal_type(wrapper.__call__)  # E: [R](self: Wrapper[[x: R], R], /, x: R) -> R
result = wrapper(1)
reveal_type(result)  # E: revealed type: int
"#,
);

testcase!(
    test_callable_class_wrapper_with_helper,
    r#"
from typing import Callable, reveal_type

class Wrapper[**P, R]:
    fn: Callable[P, R]
    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

def wrap[**P, R](f: Callable[P, R]) -> Wrapper[P, R]:
    return Wrapper(f)

def f[S](x: S) -> S: ...
wrapper = wrap(f)
reveal_type(wrapper.fn)  # E: revealed type: [R](x: R) -> R
reveal_type(wrapper.__call__)  # E: [R](self: Wrapper[[x: R], R], /, x: R) -> R
result = wrapper(1)
reveal_type(result)  # E: revealed type: int
"#,
);

testcase!(
    bug = "Need better display for callback protocol residuals in class targs",
    test_callable_class_wrapper_display_without_field,
    r#"
from typing import Callable, reveal_type

class Wrapper[**P, R]:
    def __init__(self, fn: Callable[P, R]) -> None: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

def f[S](x: S) -> S: ...
wrapper = Wrapper(f)
reveal_type(wrapper)  # E: revealed type: Wrapper[[x: Unknown], Unknown]
reveal_type(wrapper.__call__)  # E: [R](self: Wrapper[[x: R], R], /, x: R) -> R
"#,
);

testcase!(
    test_class_field_with_bare_residual,
    r#"
from typing import Callable, reveal_type

class Container[**P, R]:
    fn: Callable[P, R]
    x: R
    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn

def f[S](x: S) -> S: ...
c = Container(f)
reveal_type(c.fn)  # E: revealed type: [R](x: R) -> R
# This is expected - a bare residual targ in a class field should flatten on read
reveal_type(c.x)  # E: revealed type: Unknown
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
    bug = "Constructor identity still erases ParamSpec/return generics to Ellipsis/Unknown (and/or partial types)",
    test_callable_class_constructor_identity,
    r#"
from typing import Callable, reveal_type

def identity[**P, R](x: Callable[P, R]) -> Callable[P, R]:
    return x

class Wrapper[**P, R]:
    fn: Callable[P, R]
    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

ctor = identity(Wrapper)
reveal_type(ctor)  # E: revealed type: (fn: (...) -> Unknown) -> Wrapper[Ellipsis, Unknown]
identity2 = ctor(identity)
reveal_type(identity2.__call__)  # E: revealed type: (Wrapper[Ellipsis, Unknown], ...) -> Unknown
"#,
);

testcase!(
    test_paramspec_transform_overloaded,
    r#"
from typing import Callable, overload, reveal_type
def transform[**P, T](f: Callable[P, T]) -> Callable[P, T]: ...

@overload
def multi(x: int, y: str) -> bool: ...  # E: Overload return type `bool` is not assignable to implementation return type `None`
@overload
def multi(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def multi(*args, **kwargs): ...

result = transform(multi)
reveal_type(result)  # E: revealed type: Overload[
out_a = result(1, "ok")
reveal_type(out_a)  # E: revealed type: bool
out_b = result("ok")
"#,
);

testcase!(
    test_paramspec_identity_overloaded,
    r#"
from typing import Callable, overload, reveal_type
def identity[**P, R](x: Callable[P, R]) -> Callable[P, R]:
    return x

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def f(x): ...

result = identity(f)
reveal_type(result)  # E: revealed type: Overload[
out_a = result(1)
reveal_type(out_a)  # E: revealed type: str
out_b = result("ok")
"#,
);

testcase!(
    test_typevar_identity_overloaded,
    r#"
from typing import Callable, overload, reveal_type
def identity[A, R](x: Callable[[A], R]) -> Callable[[A], R]:
    return x

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def f(x): ...

result = identity(f)
reveal_type(result)  # E: revealed type: Overload[
out_a = result(1)
reveal_type(out_a)  # E: revealed type: str
out_b = result("ok")
"#,
);

testcase!(
    test_typevar_identity_overloaded_two_arg,
    r#"
from typing import Callable, overload, reveal_type
def identity[A, B, R](x: Callable[[A, B], R]) -> Callable[[A, B], R]:
    return x

@overload
def f(x: int, y: str) -> bool: ...  # E: Overload return type `bool` is not assignable to implementation return type `None`
@overload
def f(x: str, y: int) -> bytes: ...  # E: Overload return type `bytes` is not assignable to implementation return type `None`
def f(x, y): ...

result = identity(f)
reveal_type(result)  # E: revealed type: Overload[
out_a = result(1, "ok")
reveal_type(out_a)  # E: revealed type: bool
result("x", "ok")  # E: No matching overload found for function `typing.overload` called with arguments: (Literal['x'], Literal['ok'])
result(1, 1)  # E: No matching overload found for function `typing.overload` called with arguments: (Literal[1], Literal[1])
"#,
);

testcase!(
    test_typevar_overloaded_return_wraps_argument,
    r#"
from typing import Callable, overload, reveal_type
def higher_order[A, R](x: Callable[[A], R]) -> Callable[[list[A]], R]: ...

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def f(x): ...

result = higher_order(f)
reveal_type(result)  # E: revealed type: Overload[
out_a = result([1])
reveal_type(out_a)  # E: revealed type: str
out_b = result(["ok"])
reveal_type(out_b)  # E: revealed type: int
"#,
);

testcase!(
    test_typevar_overloaded_return_wraps_return,
    r#"
from typing import Callable, overload, reveal_type
def higher_order[A, R](x: Callable[[A], R]) -> Callable[[A], list[R]]: ...

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def f(x): ...

result = higher_order(f)
reveal_type(result)  # E: revealed type: Overload[
out_a = result(1)
reveal_type(out_a)  # E: revealed type: list[str]
out_b = result("ok")
reveal_type(out_b)  # E: revealed type: list[int]
"#,
);

testcase!(
    test_overload_pruning_bool_projection_baseline,
    r#"
from typing import Callable, overload, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[T], S]: ...

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
@overload
def f(x: bytes) -> bytes: ...  # E: Overload return type `bytes` is not assignable to implementation return type `None`
def f(x): ...

result = project(f, object())
reveal_type(result)  # E: revealed type: Overload[
"#,
);

testcase!(
    test_overload_pruning_eliminates_all_branches_float_str_vs_int,
    r#"
from typing import Callable, overload, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[T], S]: ...

@overload
def f(x: int) -> float: ...  # E: Overload return type `float` is not assignable to implementation return type `None`
@overload
def f(x: str) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
def f(x): ...

result = project(f, 1)  # E: Overload type was not compatible with the solved type `int` of type variable `S`
reveal_type(result)  # E: revealed type: (Never) -> int
"#,
);

testcase!(
    test_overload_pruning_collapses_to_single_branch,
    r#"
from typing import Callable, overload, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[T], S]: ...

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
@overload
def f(x: bytes) -> bytes: ...  # E: Overload return type `bytes` is not assignable to implementation return type `None`
def f(x): ...

result = project(f, "ok")
reveal_type(result)  # E: revealed type: (int) -> str
out = result(1)
reveal_type(out)  # E: revealed type: str
"#,
);

testcase!(
    test_overload_pruning_three_way_to_two_way,
    r#"
from typing import Callable, overload, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[T], S]: ...

@overload
def f(x: int) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
@overload
def f(x: bytes) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
def f(x): ...

result = project(f, 1)
reveal_type(result)  # E: revealed type: Overload[
out_a = result(1)
reveal_type(out_a)  # E: revealed type: int
out_b = result("ok")
reveal_type(out_b)  # E: revealed type: int
"#,
);

testcase!(
    test_overload_pruning_no_pruning_baseline,
    r#"
from typing import Callable, overload, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[T], S]: ...

@overload
def f(x: int) -> str: ...
@overload
def f(x: bytes) -> str: ...
def f(x) -> str: ...

# Both branches return str, so S=str is compatible with all branches.
# No pruning occurs; the result should be a full overload.
result = project(f, "ok")
reveal_type(result)  # E: revealed type: Overload[
out_a = result(1)
reveal_type(out_a)  # E: revealed type: str
out_b = result(b"ok")
reveal_type(out_b)  # E: revealed type: str
"#,
);

testcase!(
    test_overload_residual_equivalent_branch_collapse,
    r#"
from typing import Callable, overload, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[int], S]: ...

@overload
def f(x: int) -> str: ...
@overload
def f(x: bytes) -> str: ...
def f(x) -> str: ...

result = project(f, "ok")
reveal_type(result)  # E: revealed type: (int) -> str
out = result(1)
reveal_type(out)  # E: revealed type: str
"#,
);

testcase!(
    test_nested_higher_order_overload,
    r#"
from typing import Callable, overload, reveal_type

def identity[A, R](x: Callable[[A], R]) -> Callable[[A], R]:
    return x

@overload
def f(x: int) -> str: ...
@overload
def f(x: str) -> int: ...
def f(x) -> str | int: ...

result = identity(identity)(f)
reveal_type(result)  # E: revealed type: Overload[
out_a = result(1)
reveal_type(out_a)  # E: revealed type: str
out_b = result("ok")
reveal_type(out_b)  # E: revealed type: int
"#,
);

testcase!(
    test_overload_through_class_tparam,
    r#"
from typing import Callable, overload, reveal_type

class Wrapper[A, R]:
    fn: Callable[[A], R]
    def __init__(self, fn: Callable[[A], R]) -> None:
        self.fn = fn
    def __call__(self, x: A) -> R:
        return self.fn(x)

@overload
def f(x: int) -> str: ...
@overload
def f(x: str) -> int: ...
def f(x) -> str | int: ...

wrapper = Wrapper(f)
reveal_type(wrapper.fn)  # E: revealed type: Overload[
out_a = wrapper(1)
reveal_type(out_a)  # E: revealed type: str
out_b = wrapper("ok")
reveal_type(out_b)  # E: revealed type: int
"#,
);

testcase!(
    bug = "The semantics here work, but the flattening in reveal_type display output is confusing",
    test_overload_residual_into_callback_protocol,
    r#"
from typing import Callable, Protocol, overload, assert_type, reveal_type

class Callback[A, R](Protocol):
    def __call__(self, x: A) -> R: ...

def lift[A, R](f: Callable[[A], R]) -> Callback[A, R]: ...

@overload
def f(x: int) -> str: ...
@overload
def f(x: str) -> int: ...
def f(x) -> str | int: ...

result = lift(f)
# Should be Overload[(int) -> str, (str) -> int], but overload residual
# is flattened into a union of protocol instances instead.
reveal_type(result)  # E: revealed type: Callback[int, str] | Callback[str, int]
assert_type(result(1), str)
assert_type(result("ok"), int)
"#,
);

// Regression tests for https://github.com/facebook/pyrefly/issues/2105
// Overloaded callable protocol passed to higher-order function with ParamSpec.
// The solver commits to one overload branch too early and rejects valid calls.

testcase!(
    test_issue_2105_minimal,
    r#"
from typing import Protocol, overload, Callable

class Foo(Protocol):
    @overload
    def __call__(
        self,
        x: bool,
        y: int | None
    ) -> None: ...
    @overload
    def __call__(
        self,
        x: bool = False,
    ) -> None: ...

def higher_order[**P, T](callback: Callable[P, T], /, *args: P.args, **kwds: P.kwargs) -> Callable[P, T]: ...

def test(rmtree: Foo) -> None:
    higher_order(rmtree, y=True)
"#,
);

testcase!(
    test_two_overloaded_callables_cross_product,
    r#"
from typing import Callable, overload, reveal_type

def compose[A, B, C](f: Callable[[A], B], g: Callable[[B], C]) -> Callable[[A], C]: ...

@overload
def parse(x: str) -> int: ...
@overload
def parse(x: bytes) -> float: ...
def parse(x) -> int | float: ...

@overload
def fmt(x: int) -> str: ...
@overload
def fmt(x: float) -> bytes: ...
def fmt(x) -> str | bytes: ...

# Two independent overloaded callables produce distinct overload residual
# witnesses - in this case we flatten the types (which currently means
# we produce the first-branch match, for backward compatibility).
result = compose(parse, fmt)
reveal_type(result)  # E: revealed type: (str) -> str
"#,
);

testcase!(
    test_issue_2105_original,
    r#"
import shutil
from contextlib import ExitStack

def foo(tmpdir):
    with ExitStack() as resources:
        resources.callback(shutil.rmtree, tmpdir, ignore_errors=True)

def bar(tmpdir):
    shutil.rmtree(tmpdir, ignore_errors=True)
"#,
);
