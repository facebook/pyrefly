/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests documenting current Pyrefly behavior for degenerate generic callable
//! types — cases where a type variable appears only in the return type and
//! cannot be inferred from arguments.
//!
//! Each test uses two variables:
//! - `out_a`: assigned then immediately revealed (degenerate pin → Unknown)
//! - `out_b`: flows into a real expression that pins the partial type, then revealed
//!
//! See paramspec_overload_design.md, "Forall degeneracy" section.

use crate::testcase;

// Unbounded return-only type var: produces a partial type.
// Degenerate pin gives Unknown; typed assignment pins to int.
testcase!(
    test_unsolved_typevar_unbounded,
    r#"
from typing import reveal_type
def f[T]() -> T: ...
reveal_type(f())  # E: revealed type: @_
out_a = f()
reveal_type(out_a)  # E: revealed type: Unknown
out_b: int = f()
reveal_type(out_b)  # E: revealed type: int
"#,
);

// Bounded return-only type var: same partial behavior as unbounded.
// The bound does not serve as a fallback for the degenerate pin.
testcase!(
    test_unsolved_typevar_bounded,
    r#"
from typing import reveal_type
def f[T: int]() -> T: ...
reveal_type(f())  # E: revealed type: @_
out_a = f()
reveal_type(out_a)  # E: revealed type: Unknown
out_b: int = f()
reveal_type(out_b)  # E: revealed type: int
"#,
);

// Defaulted return-only type var: default IS used, no partial type.
testcase!(
    test_unsolved_typevar_with_default,
    r#"
from typing import reveal_type
def f[T = int]() -> T: ...
reveal_type(f())  # E: revealed type: int
out_a = f()
reveal_type(out_a)  # E: revealed type: int
out_b: int = f()
reveal_type(out_b)  # E: revealed type: int
"#,
);

// Unsolved type var nested in a container: partial type propagates into list.
// Calling .append(42) pins T to int.
testcase!(
    test_unsolved_typevar_in_container,
    r#"
from typing import reveal_type
def f[T]() -> list[T]: ...
reveal_type(f())  # E: revealed type: list[@_]
out_a = f()
reveal_type(out_a)  # E: revealed type: list[Unknown]
out_b = f()
out_b.append(42)
reveal_type(out_b)  # E: revealed type: list[int]
"#,
);

// Type var present in return but absent from all param types.
testcase!(
    test_unsolved_typevar_not_in_params,
    r#"
from typing import reveal_type
def f[T](x: int) -> T: ...
reveal_type(f(42))  # E: revealed type: @_
out_a = f(42)
reveal_type(out_a)  # E: revealed type: Unknown
out_b: str = f(42)
reveal_type(out_b)  # E: revealed type: str
"#,
);

// Passing a generic function through a ParamSpec wrapper loses generic structure.
// The result is a partial callable; calling it with a concrete arg pins the types.
testcase!(
    test_paramspec_wrap_generic,
    r#"
from typing import Callable, Awaitable, reveal_type
def wrap[**P, T](f: Callable[P, T]) -> Callable[P, Awaitable[T]]: ...
def identity[X](x: X) -> X: ...
reveal_type(wrap(identity))  # E: revealed type: (x: @_) -> Awaitable[@_]
out_a = wrap(identity)
reveal_type(out_a)  # E: revealed type: (x: Unknown) -> Awaitable[Unknown]
out_b = wrap(identity)
called = out_b(42)
reveal_type(out_b)  # E: revealed type: (x: int) -> Awaitable[int]
reveal_type(called)  # E: revealed type: Awaitable[int]
"#,
);

// Concatenate strips the only binding site of X, producing a degenerate callable.
// Unlike the other tests, this does NOT produce a partial type — X collapses to
// Any, and downstream usage cannot recover type information.
testcase!(
    test_concatenate_strips_generic_param,
    r#"
from typing import Callable, Concatenate, Any, reveal_type

def strip_first[**P, T](f: Callable[Concatenate[Any, P], T]) -> Callable[P, T]: ...
def identity[X](x: X) -> X: ...

# Stripping the first param of (X) -> X gives () -> X, but X has lost its
# binding site and collapses to Any. The result is () -> Any.
reveal_type(strip_first(identity))  # E: revealed type: () -> Any

# out_a: degenerate pin. The callable is () -> Any, not a partial type.
out_a = strip_first(identity)
reveal_type(out_a)  # E: revealed type: () -> Any

# out_b: calling it produces Any — no partial type to pin.
out_b = strip_first(identity)
called_b = out_b()
reveal_type(out_b)  # E: revealed type: () -> Any
reveal_type(called_b)  # E: revealed type: Any

# out_c: calling it produces Any — no partial type to pin.
out_c = strip_first(identity)
def takes_callback(callback: Callable[[], int]) -> None: ...
takes_callback(out_c)
reveal_type(out_c)  # E: revealed type: () -> Any
"#,
);

// Concatenate preserves X in params (X survives stripping), but the generic
// forall structure is still lost — X becomes a partial type.
// Calling the result with a concrete arg pins X.
testcase!(
    bug = "Generic structure lost even though X survives in params after stripping",
    test_concatenate_preserves_generic_param,
    r#"
from typing import Callable, Concatenate, Any, reveal_type
def strip_first[**P, T](f: Callable[Concatenate[Any, P], T]) -> Callable[P, T]: ...
def swap[X](ignored: int, x: X) -> X: ...
reveal_type(strip_first(swap))  # E: revealed type: (x: @_) -> @_
out_a = strip_first(swap)
reveal_type(out_a)  # E: revealed type: (x: Unknown) -> Unknown
out_b = strip_first(swap)
called = out_b("hello")
reveal_type(out_b)  # E: revealed type: (x: str) -> str
reveal_type(called)  # E: revealed type: str
"#,
);

// ParamSpec pins to first overload instead of preserving overload structure.
// This is the core bug from GitHub issue #2105.
testcase!(
    bug = "ParamSpec pins to first overload (issue #2105)",
    test_overload_paramspec_pins_first,
    r#"
from typing import Callable, Protocol, overload

class Foo(Protocol):
    @overload
    def __call__(self, ignore_errors: bool, onerror: int | None) -> None: ...
    @overload
    def __call__(self, ignore_errors: bool = False) -> None: ...

def callback[**P, T](
    callback: Callable[P, T], /, *args: P.args, **kwds: P.kwargs
) -> Callable[P, T]: ...

def test(rmtree: Foo) -> None:
    callback(rmtree, ignore_errors=True)  # E: Missing argument `onerror` in function `callback`
"#,
);
