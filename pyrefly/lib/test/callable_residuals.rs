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
reveal_type(f_out)  # E: revealed type: [R](x: Unknown) -> R
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
    bug = "Generic functions don't work with ParamSpec",
    test_param_spec_generic_function,
    r#"
from typing import Callable, reveal_type
def identity[**P, R](x: Callable[P, R]) -> Callable[P, R]:
    return x
def foo[T](x: T, y: T) -> T:
    return x
foo2 = identity(foo)
reveal_type(foo2)  # E: revealed type: [R](x: Unknown, y: Unknown) -> R
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
