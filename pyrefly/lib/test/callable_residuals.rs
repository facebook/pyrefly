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

use crate::test::util::TestEnv;
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
reveal_type(result)  # E: revealed type: [*Ts, R](**tuple[(**tuple[*Ts]) -> R]) -> (**tuple[*Ts]) -> R
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
reveal_type(result)  # E: revealed type: [**P, T](x: (ParamSpec(P)) -> T) -> (ParamSpec(P)) -> T
"#,
);

testcase!(
    test_param_spec_identity_of_identity_behavior,
    r#"
from typing import Callable, assert_type, reveal_type
def identity[**P, T](x: Callable[P, T]) -> Callable[P, T]:
    return x
def f(x: int, y: str) -> str:
    return y
result = identity(identity)
lifted = result(f)
reveal_type(lifted)  # E: revealed type: (x: int, y: str) -> str
assert_type(lifted(1, "ok"), str)
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
from typing import Callable, assert_type, reveal_type

class Box[T]:
    fn: Callable[[T], T]
    def __init__(self, fn: Callable[[T], T]) -> None:
        self.fn = fn

def f[S](x: S) -> S: ...
b = Box(f)
reveal_type(b.fn)  # E: revealed type: [S](S) -> S
assert_type(b.fn(1), int)
"#,
);

testcase!(
    test_callable_class_wrapper,
    r#"
from typing import Callable, assert_type, reveal_type

class Wrapper[**P, R]:
    fn: Callable[P, R]
    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

def f[S](x: S) -> S: ...
wrapper = Wrapper(f)
reveal_type(wrapper.fn)  # E: revealed type: [R](x: R) -> R
reveal_type(wrapper.__call__)  # E: [R](x: R) -> R
assert_type(wrapper(1), int)
"#,
);

testcase!(
    test_callable_class_wrapper_with_helper,
    r#"
from typing import Callable, assert_type, reveal_type

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
reveal_type(wrapper.__call__)  # E: [R](x: R) -> R
assert_type(wrapper(1), int)
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
reveal_type(wrapper)  # E: revealed type: Wrapper[[x: GenericResidual@R], GenericResidual@R]
reveal_type(wrapper.__call__)  # E: [R](x: R) -> R
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
reveal_type(c2)  # E: revealed type: [T](x: T) -> C[T]
x: C[int] = c2(1)
"#,
);

testcase!(
    test_overloaded_generic_new_constructor_callable,
    TestEnv::one_with_path(
        "constructor_stub",
        "constructor_stub.pyi",
        r#"
from typing import overload

class C[T = str]:
    @overload
    def __new__(cls) -> C[T]: ...
    @overload
    def __new__(cls, value: T, /) -> C[T]: ...
"#,
    ),
    r#"
from collections.abc import Callable
from typing import assert_type
from constructor_stub import C

def call0[R](ctor: Callable[[], R]) -> R:
    return ctor()

def identity[**P, R](ctor: Callable[P, R]) -> Callable[P, R]:
    return ctor

assert_type(call0(C), C[str])
assert_type(identity(C)(1), C[int])
"#,
);

testcase!(
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
reveal_type(ctor)  # E: revealed type: [**P, R](fn: (ParamSpec(P)) -> R) -> Wrapper[P, R]
identity2 = ctor(identity)
reveal_type(identity2.__call__)  # E: revealed type: [**P, R](x: (ParamSpec(P)) -> R) -> (ParamSpec(P)) -> R
"#,
);

testcase!(
    test_overloaded_constructor_return_only_class_tparam,
    r#"
from collections import defaultdict

def consume(metrics: dict[str, list[float]]) -> None: ...

metrics = defaultdict(list)
metrics["runtime"].append(1.0)
consume(metrics)
"#,
);

testcase!(
    test_constructor_overload_with_specialized_self,
    r#"
import contextlib
from typing import Any, Callable

null_context = contextlib.nullcontext

def consume(
    factory: Callable[[], contextlib.AbstractContextManager[Any]] = null_context,
) -> None: ...
"#,
);

testcase!(
    test_paramspec_transform_overloaded,
    r#"
from typing import Callable, overload, assert_type, reveal_type
def transform[**P, T](f: Callable[P, T]) -> Callable[P, T]: ...

@overload
def multi(x: int, y: str) -> bool: ...  # E: Overload return type `bool` is not assignable to implementation return type `None`
@overload
def multi(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def multi(*args, **kwargs): ...

result = transform(multi)
reveal_type(result)  # E: revealed type: Overload[ (x: int, y: str) -> bool (x: str) -> int ]
assert_type(result(1, "ok"), bool)
result("ok")
"#,
);

testcase!(
    test_paramspec_identity_overloaded,
    r#"
from typing import Callable, overload, assert_type, reveal_type
def identity[**P, R](x: Callable[P, R]) -> Callable[P, R]:
    return x

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def f(x): ...

result = identity(f)
reveal_type(result)  # E: revealed type: Overload[ (x: int) -> str (x: str) -> int ]
assert_type(result(1), str)
result("ok")
"#,
);

testcase!(
    test_typevar_identity_overloaded,
    r#"
from typing import Callable, overload, assert_type, reveal_type
def identity[A, R](x: Callable[[A], R]) -> Callable[[A], R]:
    return x

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def f(x): ...

result = identity(f)
reveal_type(result)  # E: revealed type: Overload[ (int) -> str (str) -> int ]
assert_type(result(1), str)
result("ok")
"#,
);

testcase!(
    test_typevar_identity_overloaded_two_arg,
    r#"
from typing import Callable, overload, assert_type, reveal_type
def identity[A, B, R](x: Callable[[A, B], R]) -> Callable[[A, B], R]:
    return x

@overload
def f(x: int, y: str) -> bool: ...  # E: Overload return type `bool` is not assignable to implementation return type `None`
@overload
def f(x: str, y: int) -> bytes: ...  # E: Overload return type `bytes` is not assignable to implementation return type `None`
def f(x, y): ...

result = identity(f)
reveal_type(result)  # E: revealed type: Overload[ (int, str) -> bool (str, int) -> bytes ]
assert_type(result(1, "ok"), bool)
result("x", "ok")  # E: No matching overload found for function `typing.overload` called with arguments: (Literal['x'], Literal['ok'])
result(1, 1)  # E: No matching overload found for function `typing.overload` called with arguments: (Literal[1], Literal[1])
"#,
);

testcase!(
    test_typevar_overloaded_return_wraps_argument,
    r#"
from typing import Callable, overload, assert_type, reveal_type
def higher_order[A, R](x: Callable[[A], R]) -> Callable[[list[A]], R]: ...

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def f(x): ...

result = higher_order(f)
reveal_type(result)  # E: revealed type: Overload[ (list[int]) -> str (list[str]) -> int ]
assert_type(result([1]), str)
assert_type(result(["ok"]), int)
"#,
);

testcase!(
    test_typevar_overloaded_return_wraps_return,
    r#"
from typing import Callable, overload, assert_type, reveal_type
def higher_order[A, R](x: Callable[[A], R]) -> Callable[[A], list[R]]: ...

@overload
def f(x: int) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
def f(x): ...

result = higher_order(f)
reveal_type(result)  # E: revealed type: Overload[ (int) -> list[str] (str) -> list[int] ]
assert_type(result(1), list[str])
assert_type(result("ok"), list[int])
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
reveal_type(result)  # E: revealed type: Overload[ (int) -> object (str) -> object (bytes) -> object ]
"#,
);

testcase!(
    test_overload_pruning_commits_nested_generic_constraints,
    r#"
from typing import Any, Iterable, assert_type, overload

@overload
def collect() -> list[Any]: ...
@overload
def collect[T](xs: Iterable[T]) -> list[T]: ...
def collect(xs: Any = ()) -> Any: ...

rows: list[list[int]] = [[1], [2]]
assert_type(list(map(collect, rows)), list[list[int]])
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

result = project(f, 1)  # E: Overload type was not compatible with solved type variables: S = int
reveal_type(result)  # E: revealed type: (Never) -> int
"#,
);

testcase!(
    test_overload_pruning_ignored_when_solved_before_materialization,
    r#"
from typing import Callable, overload, reveal_type

def project[T, S](f: Callable[[T], tuple[T, S]], y: T, z: S) -> Callable[[], tuple[T, S]]: ...

@overload
def f(x: int) -> tuple[int, str]: ...  # E: Overload return type `tuple[int, str]` is not assignable to implementation return type `None`
@overload
def f(x: str) -> tuple[str, int]: ...  # E: Overload return type `tuple[str, int]` is not assignable to implementation return type `None`
def f(x): ...

result = project(f, 1, 1)  # E: Overload type was not compatible with solved type variables: S = int, T = int
# We keep solved type-variable substitutions in the result even when overload pruning
# later rejects all captured branches.
reveal_type(result)  # E: revealed type: () -> tuple[int, int]
"#,
);

testcase!(
    test_overload_pruning_ignored_for_constrained_tvar_solved_early,
    r#"
from typing import Callable, overload, reveal_type

def project[T: (int, str)](f: Callable[[T], T], y: T) -> Callable[[T], T]: ...

@overload
def f(x: float) -> float: ...  # E: Overload return type `float` is not assignable to implementation return type `None`
@overload
def f(x: bytes) -> bytes: ...  # E: Overload return type `bytes` is not assignable to implementation return type `None`
def f(x): ...

result = project(f, 1)  # E: Overload type was not compatible with solved type variables: unknown = int
reveal_type(result)  # E: revealed type: (int) -> int
"#,
);

testcase!(
    test_overload_pruning_collapses_to_single_branch,
    r#"
from typing import Callable, overload, assert_type, reveal_type

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
assert_type(result(1), str)
"#,
);

testcase!(
    test_overload_pruning_three_way_to_two_way,
    r#"
from typing import Callable, overload, assert_type, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[T], S]: ...

@overload
def f(x: int) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
@overload
def f(x: str) -> int: ...  # E: Overload return type `int` is not assignable to implementation return type `None`
@overload
def f(x: bytes) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `None`
def f(x): ...

result = project(f, 1)
reveal_type(result)  # E: revealed type: Overload[ (int) -> int (str) -> int ]
assert_type(result(1), int)
assert_type(result("ok"), int)
"#,
);

testcase!(
    test_overload_pruning_no_pruning_baseline,
    r#"
from typing import Callable, overload, assert_type, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[T], S]: ...

@overload
def f(x: int) -> str: ...
@overload
def f(x: bytes) -> str: ...
def f(x) -> str: ...

# Both branches return str, so S=str is compatible with all branches.
# No pruning occurs; the result should be a full overload.
result = project(f, "ok")
reveal_type(result)  # E: revealed type: Overload[ (int) -> str (bytes) -> str ]
assert_type(result(1), str)
assert_type(result(b"ok"), str)
"#,
);

testcase!(
    test_overload_residual_equivalent_branch_collapse,
    r#"
from typing import Callable, overload, assert_type, reveal_type

def project[T, S](f: Callable[[T], S], y: S) -> Callable[[int], S]: ...

@overload
def f(x: int) -> str: ...
@overload
def f(x: bytes) -> str: ...
def f(x) -> str: ...

result = project(f, "ok")
reveal_type(result)  # E: revealed type: (int) -> str
assert_type(result(1), str)
"#,
);

testcase!(
    test_nested_higher_order_overload,
    r#"
from typing import Callable, overload, assert_type, reveal_type

def identity[A, R](x: Callable[[A], R]) -> Callable[[A], R]:
    return x

@overload
def f(x: int) -> str: ...
@overload
def f(x: str) -> int: ...
def f(x) -> str | int: ...

result = identity(identity)(f)
reveal_type(result)  # E: revealed type: Overload[ (int) -> str (str) -> int ]
assert_type(result(1), str)
assert_type(result("ok"), int)
"#,
);

testcase!(
    test_overload_through_class_tparam,
    r#"
from typing import Callable, overload, assert_type, reveal_type

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
reveal_type(wrapper.fn)  # E: revealed type: Overload[ (int) -> str (str) -> int ]
assert_type(wrapper(1), str)
assert_type(wrapper("ok"), int)
"#,
);

testcase!(
    test_overload_residual_nested_inline_union_fallback,
    r#"
from typing import Callable, overload, reveal_type

def project[A, R](f: Callable[[A], R]) -> list[tuple[A, R]]: ...

@overload
def f(x: int) -> str: ...
@overload
def f(x: str) -> int: ...
def f(x) -> str | int: ...

result = project(f)
reveal_type(result)  # E: revealed type: list[tuple[OverloadResidual@[int, str], OverloadResidual@[str, int]]]
"#,
);

testcase!(
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
# Overload residual fallback should stay inline for non-callable roots.
reveal_type(result)  # E: revealed type: Callback[OverloadResidual@[int, str], OverloadResidual@[str, int]]
assert_type(result(1), str)
assert_type(result("ok"), int)
"#,
);

testcase!(
    test_await_preserves_overload_residual_in_callback_protocol,
    r#"
import asyncio
from functools import partial
from typing import Callable, Protocol, assert_type, overload

class Callback[A, R](Protocol):
    def __call__(self, x: A) -> R: ...

def lift[A, R](f: Callable[[A], R]) -> Callback[A, R]: ...

@overload
def f(x: int) -> str: ...
@overload
def f(x: str) -> int: ...
def f(x: int | str) -> int | str: ...

async def test() -> None:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, partial(lift, f))
    assert_type(result(1), str)
    assert_type(result("ok"), int)
"#,
);

testcase!(
    test_await_resolves_overload_forwarded_through_paramspec,
    r#"
import asyncio
import subprocess
from typing import assert_type

async def test() -> None:
    result = await asyncio.to_thread(
        subprocess.run,
        ["printf", "ok"],
        capture_output=True,
        text=True,
    )
    assert_type(result, subprocess.CompletedProcess[str])
    assert_type(result.stdout + result.stderr, str)
"#,
);

testcase!(
    test_await_resolves_overload_forwarded_through_typevartuple,
    r#"
from collections.abc import Callable, Iterator
from typing import TypeVar, TypeVarTuple, Unpack, assert_type

Args = TypeVarTuple("Args")
R = TypeVar("R")

async def run_sync(
    func: Callable[[Unpack[Args]], R],
    *args: Unpack[Args],
) -> R: ...

async def test(iterator: Iterator[str]) -> None:
    value = await run_sync(next, iterator, None)
    assert_type(value, str | None)
    if value is not None:
        assert_type(value, str)
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
# we produce the branch union fallback).
result = compose(parse, fmt)
reveal_type(result)  # E: revealed type: (bytes | str) -> bytes | str
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

// Regression test for a panic when pruning against a residual Variable
// in the case where overload analysis merged the Quantified with a partial
// type (behavior for Recursive / Unwrap is the same).
testcase!(
    test_overload_residual_with_partial_quantified_var,
    r#"
from typing import overload, Callable, assert_type

class C[T]:
    @overload
    def method(self, x: T) -> T: ...
    @overload
    def method(self, x: str) -> str: ...
    def method(self, x): return x

def apply[U](fn: Callable[[U], U], default: U) -> U: ...

c = C()
result = apply(c.method, 42)
assert_type(result, int)
    "#,
);

// Regression test for a panic when converting a residual Variable to a Type
// in the case where overload analysis merged the Quantified with a partial
// type (behavior for Recursive / Unwrap is the same).
testcase!(
    test_overload_residual_with_partial_contained_var,
    r#"
from typing import overload, Any, Callable, assert_type, reveal_type

class C[T]:
    def __init__(self, items: list[T]) -> None: ...
    @overload
    def method(self, x: T) -> T: ...
    @overload
    def method(self, x: str) -> str: ...
    def method(self, x): return x

def apply[U](fn: Callable[[U], U]) -> U: ...

c = C([])
result = apply(c.method)
# The partial type for `c` does not get pinned, so it resolves to Unknown
assert_type(result, str | Any)
assert_type(c, C[Any])
    "#,
);

testcase!(
    test_overload_residual_in_param_default,
    r#"
from typing import Callable, assert_type
class A(int): ...
def f[T](x: int, y: Callable[[int], T] = A) -> T:
    return y(x)
assert_type(f(0), A)
    "#,
);

// Regression test for a false `incompatible-overload-residual` error when
// passing a generic overloaded function (like `operator.add`) to a
// higher-order function (like `functools.reduce`). The overload's first
// branch (`SupportsAdd`) is applicable, but a self-referential probe var
// leaking into the captured residual bound used to prune every branch.
testcase!(
    test_overload_residual_generic_protocol_arg_not_pruned,
    r#"
from typing import Callable, Iterable, TypeVar, Protocol, assert_type, overload

# Protocol type vars (distinct identities, mirroring _typeshed).
_PTc = TypeVar("_PTc", contravariant=True)
_PTco = TypeVar("_PTco", covariant=True)
class SupportsAdd(Protocol[_PTc, _PTco]):
    def __add__(self, x: _PTc, /) -> _PTco: ...
class SupportsRAdd(Protocol[_PTc, _PTco]):
    def __radd__(self, x: _PTc, /) -> _PTco: ...

# add's own type vars (distinct identities, mirroring _operator).
_Tcontra = TypeVar("_Tcontra", contravariant=True)
_Tco = TypeVar("_Tco", covariant=True)
@overload
def add(a: SupportsAdd[_Tcontra, _Tco], b: _Tcontra, /) -> _Tco: ...
@overload
def add(a: _Tcontra, b: SupportsRAdd[_Tcontra, _Tco], /) -> _Tco: ...
def add(a, b, /) -> object: ...

_T = TypeVar("_T")
def reduce(function: Callable[[_T, _T], _T], iterable: Iterable[_T], /) -> _T: ...

lists: list[list[str]] = [["a"], ["b"]]
y = reduce(add, lists)
assert_type(y, list[str])
    "#,
);

testcase!(
    test_overload_residual_generic_protocol_rejects_mixed_union_arg,
    r#"
from functools import reduce
from operator import add
xs: list[int | str] = [1, "x"]
reduce(add, xs)  # E: Overload type was not compatible with solved type variables: _T = int | str
    "#,
);
