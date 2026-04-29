/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
    bug = "TypeVarTuple loses *Ts and degrades to Unknown tuple in revealed type",
    test_type_var_tuple_identity_of_identity,
    r#"
from typing import Callable, reveal_type
def identity_tuple[*Ts, R](x: Callable[[*Ts], R]) -> Callable[[*Ts], R]:
    return x
result = identity_tuple(identity_tuple)
reveal_type(result)  # E: revealed type: [R](**tuple[(**tuple[Unknown, ...]) -> R]) -> (**tuple[Unknown, ...]) -> R
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
