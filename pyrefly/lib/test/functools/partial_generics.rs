/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `functools.partial` over generic / overloaded targets — the area where pyrefly's solver
//! leaks `GenericResidual@_T` (returning `Unknown`) or emits a false-positive `bad-specialization`.
//! Covers generic/overloaded scenarios and pyrefly issue regressions (`# Regression: ...`).
//! Divergences are `bug=`-marked; `# WANT:` records the correct behavior.

use crate::functools_testcase;

// ===== Generic functions =====

// The residual of a same-typevar partial stays generic, so a later call re-solves `T` from the
// remaining argument exactly as a direct call would: `partial(foo, 1)` accepts `p1("a")` (T=str),
// matching `foo(1, "a")`. This is precision-only under-approximation, consistent with the runtime.
functools_testcase!(
    test_partial_generic_same_typevar,
    r#"
from typing import TypeVar, reveal_type
import functools
T = TypeVar("T")
def foo(a: T, b: T) -> T: ...
p1 = functools.partial(foo, 1)
reveal_type(p1(2))  # E: revealed type: int
reveal_type(p1("a"))  # E: revealed type: str
p2 = functools.partial(foo, "a")
reveal_type(p2(1))  # E: revealed type: int
reveal_type(p2("a"))  # E: revealed type: str
"#,
);

functools_testcase!(
    test_partial_generic_two_typevars,
    r#"
from typing import TypeVar, reveal_type
import functools
T = TypeVar("T")
U = TypeVar("U")
def bar(a: T, b: U) -> U: ...
p3 = functools.partial(bar, 1)
reveal_type(p3(2))  # E: revealed type: int
reveal_type(p3("a"))  # E: revealed type: str
"#,
);

functools_testcase!(
    test_partial_of_generic_function,
    r#"
from functools import partial
from typing import TypeVar, List, reveal_type
T = TypeVar("T")
def get(n: int, args: List[T]) -> T: ...
first = partial(get, 0)
x: List[str] = []
reveal_type(first(x))  # E: revealed type: str
reveal_type(first([1]))  # E: revealed type: int
first_kw = partial(get, n=0)
reveal_type(first_kw(args=[1]))  # E: revealed type: int
first_kw([1])  # E: Expected argument `args` to be passed by name
"#,
);

// ===== Constrained TypeVar values =====

functools_testcase!(
    test_partial_type_var_values_f,
    r#"
from functools import partial
from typing import TypeVar, reveal_type
T = TypeVar("T", int, str)
def f(x: int, y: T) -> T:
    return y
fp = partial(f, 1)
reveal_type(fp(1))  # E: revealed type: int
reveal_type(fp("a"))  # E: revealed type: str
fp(1)
fp("a")
fp(object())  # E: `object` is not assignable to any of constraints `int`, `str` of type variable `T`
"#,
);

functools_testcase!(
    test_partial_type_var_values_g,
    r#"
from functools import partial
from typing import TypeVar, reveal_type
T = TypeVar("T", int, str)
def g(x: T, y: int) -> T:
    return x
gp = partial(g, 1)
reveal_type(gp(1))  # E: revealed type: int
gp(1)
gp("a")  # E: Argument `Literal['a']` is not assignable to parameter `y` with type `int`
"#,
);

// Same-typevar shared across the bound and residual positions: the residual stays generic and the
// call re-solves `T` from `y`, matching a direct `h(1, y)` call. `hp("a")` re-solves `T=str`.
functools_testcase!(
    test_partial_type_var_values_h,
    r#"
from functools import partial
from typing import TypeVar, reveal_type
T = TypeVar("T", int, str)
def h(x: T, y: T) -> T:
    return x
hp = partial(h, 1)
reveal_type(hp(1))  # E: revealed type: int
hp(1)
hp("a")
"#,
);

functools_testcase!(
    test_partial_bounded_type_var_target,
    r#"
from typing import Callable, TypeVar, Type
import functools
T = TypeVar("T", bound=Callable[[str, int], str])
S = TypeVar("S", bound=Type[int])
def foo(f: T) -> T:
    g = functools.partial(f, "foo")
    return f
def bar(f: S) -> S:
    g = functools.partial(f, "foo")
    return f
"#,
);

// ===== TypeVar erasure / scope =====

// A plain-TypeVar target (`func_b`, `func_c`) is now re-scoped into a `Forall` over the residual, so
// its genericity survives and the downstream incompatible use is flagged at the residual param. A
// `ParamSpec`/`TypeVarTuple` target still defers to the stub and leaks a `GenericResidual` placeholder
// (out of scope); `# WANT` records the eventual erasure-to-`Any` behavior.
functools_testcase!(
    bug = "ParamSpec/TypeVarTuple partial targets defer to the stub and leak GenericResidual instead of erasing to Any",
    test_partial_type_var_erasure_no_leak,
    r#"
from typing import reveal_type
from typing import Callable, TypeVar, Union
from typing_extensions import ParamSpec, TypeVarTuple, Unpack
from functools import partial
def use_int_callable(x: Callable[[int], int]) -> None:
    pass
def use_func_callable(
    x: Callable[
        [Callable[[int], None]],
        Callable[[int], None],
    ],
) -> None:
    pass
Tc = TypeVar("Tc", int, str)
Tb = TypeVar("Tb", bound=Union[int, str])
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")
def func_b(a: Tb, b: str) -> Tb:
    return a
def func_c(a: Tc, b: str) -> Tc:
    return a
def func_fn(fn: Callable[P, Tc], b: str) -> Callable[P, Tc]:
    return fn
def func_fn_unpack(fn: Callable[[Unpack[Ts]], Tc], b: str) -> Callable[[Unpack[Ts]], Tc]:
    return fn
reveal_type(partial(func_b, b=""))  # E: revealed type: [Tb: int | str](a: Tb, *, b: str = ...) -> Tb
reveal_type(partial(func_c, b=""))  # E: revealed type: [Tc: (int, str)](a: Tc, *, b: str = ...) -> Tc
# WANT: revealed type: partial[(*Any, **Any) -> Any]
reveal_type(partial(func_fn, b=""))  # E: revealed type: partial[(ParamSpec(GenericResidual@P)) -> GenericResidual@Tc]
# WANT: revealed type: partial[(*Any) -> Any]
reveal_type(partial(func_fn_unpack, b=""))  # E: revealed type: partial[(**tuple[*GenericResidual@Ts]) -> GenericResidual@Tc]
use_int_callable(partial(func_b, b=""))
use_func_callable(partial(func_b, b=""))
use_int_callable(partial(func_c, b=""))
use_func_callable(partial(func_c, b=""))
# WANT: error: partial[(*Any, **Any) -> Any] not assignable to Callable[[int], int]
use_int_callable(partial(func_fn, b=""))  # E: Argument `partial[(ParamSpec(GenericResidual@P)) -> GenericResidual@Tc]` is not assignable to parameter `x` with type `(int) -> int` in function `use_int_callable`
use_func_callable(partial(func_fn, b=""))
# WANT: error: partial[(*Any) -> Any] not assignable to Callable[[int], int]
use_int_callable(partial(func_fn_unpack, b=""))  # E: Argument `partial[(**tuple[*GenericResidual@Ts]) -> GenericResidual@Tc]` is not assignable to parameter `x` with type `(int) -> int` in function `use_int_callable`
use_func_callable(partial(func_fn_unpack, b=""))
"#,
);

// A TypeVar bound by the enclosing function is preserved in the residual signature (not erased); the
// downstream incompatible use is flagged against the residual param.
functools_testcase!(
    test_partial_type_var_erasure_in_scope_bounded,
    r#"
from typing import reveal_type
from typing import Callable, TypeVar, Union
from functools import partial
Tb = TypeVar("Tb", bound=Union[int, str])
def use_int_callable(x: Callable[[int], int]) -> None:
    pass
def outer_b(arg: Tb) -> None:
    def inner(a: Tb, b: str) -> Tb:
        return a
    reveal_type(partial(inner, b=""))  # E: revealed type: (a: Tb, *, b: str = ...) -> Tb
    use_int_callable(partial(inner, b=""))  # E: Argument `(a: Tb, *, b: str = ...) -> Tb` is not assignable to parameter `x` with type `(int) -> int` in function `use_int_callable`
"#,
);

functools_testcase!(
    bug = "an in-scope constrained TypeVar stays symbolic in the residual rather than being expanded to partial[int]/partial[str]",
    test_partial_type_var_erasure_in_scope_constrained,
    r#"
from typing import reveal_type
from typing import Callable, TypeVar
from functools import partial
Tc = TypeVar("Tc", int, str)
def use_int_callable(x: Callable[[int], int]) -> None:
    pass
def outer_c(arg: Tc) -> None:
    def inner(a: Tc, b: str) -> Tc:
        return a
    # WANT: revealed type: partial[int] / partial[str] (constrained typevar expanded)
    reveal_type(partial(inner, b=""))  # E: revealed type: (a: Tc, *, b: str = ...) -> Tc
    use_int_callable(partial(inner, b=""))  # E: Argument `(a: Tc, *, b: str = ...) -> Tc` is not assignable to parameter `x` with type `(int) -> int` in function `use_int_callable`
"#,
);

// ===== Overloaded targets =====

// `partial(foo)` with nothing bound forwards the overload unchanged, so overload resolution still
// happens at the call site: each shape resolves to its own overload and mismatches are flagged.
functools_testcase!(
    test_partial_over_overloaded_function,
    r#"
from typing import reveal_type, overload, Any
import functools
@overload
def foo(a: int, b: str) -> int: ...
@overload
def foo(a: str, b: int) -> str: ...
def foo(*a: Any, **k: Any) -> Any: ...
p1 = functools.partial(foo)
reveal_type(p1(1, "a"))  # E: revealed type: int
reveal_type(p1("a", 1))  # E: revealed type: str
p1(1, 2)  # E: No matching overload found for function `foo`
p1("a", "b")  # E: No matching overload found for function `foo`
"#,
);

// Binding an argument compatible with exactly one overload branch drops the others, so the residual
// is that branch's remaining parameters and calls are checked against them.
functools_testcase!(
    test_partial_bound_over_overloaded_selects_branch,
    r#"
from typing import reveal_type, overload, Any
import functools
@overload
def foo(a: int, b: str) -> int: ...
@overload
def foo(a: str, b: int) -> str: ...
def foo(*a: Any, **k: Any) -> Any: ...
p = functools.partial(foo, 1)
reveal_type(p)  # E: revealed type: (b: str) -> int
reveal_type(p("ok"))  # E: revealed type: int
p(2)  # E: Argument `Literal[2]` is not assignable to parameter `b` with type `str`
q = functools.partial(foo, "s")
reveal_type(q)  # E: revealed type: (b: int) -> str
reveal_type(q(2))  # E: revealed type: str
"#,
);

// Binding an argument compatible with several branches keeps all of them, recombined into an overload
// so per-call resolution still selects a branch and a call matching none is rejected.
functools_testcase!(
    test_partial_bound_over_overloaded_multiple_survivors,
    r#"
from typing import reveal_type, overload, Any
import functools
@overload
def h(a: int, b: str) -> int: ...
@overload
def h(a: int, b: bytes) -> bool: ...
def h(*a: Any, **k: Any) -> Any: ...
p = functools.partial(h, 1)
reveal_type(p("x"))  # E: revealed type: int
reveal_type(p(b"x"))  # E: revealed type: bool
p(5)  # E: No matching overload found for function `h`
"#,
);

// A keyword-bound argument filters branches the same way; the branch whose keyword type matches wins.
functools_testcase!(
    test_partial_kwbound_over_overloaded,
    r#"
from typing import reveal_type, overload, Any
import functools
@overload
def foo(a: int, b: str) -> int: ...
@overload
def foo(a: str, b: int) -> str: ...
def foo(*a: Any, **k: Any) -> Any: ...
p = functools.partial(foo, b="s")
reveal_type(p(1))  # E: revealed type: int
p("x")  # E: Argument `Literal['x']` is not assignable to parameter `a` with type `int`
"#,
);

functools_testcase!(
    bug = "partial of an overloaded __call__ protocol always resolves the first overload: partial(x, \"a\")() should be str but is int",
    test_partial_over_overloaded_protocol,
    r#"
from typing import reveal_type
from functools import partial
from typing import Protocol, overload
class P(Protocol):
    @overload
    def __call__(self, x: int) -> int: ...
    @overload
    def __call__(self, x: str) -> str: ...
def f(x: P) -> None:
    reveal_type(partial(x, 1)())  # E: revealed type: int
    # WANT: revealed type: str
    reveal_type(partial(x, "a")())  # E: revealed type: int
"#,
);

// ===== Issue regressions (generic / decorator) =====

// Regression: https://github.com/facebook/pyrefly/issues/3330
functools_testcase!(
    test_partial_decorator_erases_signature,
    r#"
import functools
from typing import TypeVar, reveal_type
C = TypeVar("C")
def decorator(fn: C, s: str) -> C: return fn
@functools.partial(decorator, s="foo")
def f(x: int) -> int: return x
reveal_type(f)  # E: revealed type: (x: int) -> int
f(None)  # E: Argument `None` is not assignable to parameter `x` with type `int` in function `f`
"#,
);

// Regression: https://github.com/facebook/pyrefly/issues/3329
// Binding only the keyword-only `s` leaves `fun` (and thus `C`) free, bound at decoration time, so
// there is no spurious bad-specialization against `C`'s upper bound.
functools_testcase!(
    test_partial_generic_decorator_kwonly_false_positive,
    r#"
import functools
from typing import TypeVar, Callable
C = TypeVar("C", bound=Callable)
def api_boundary2(fun: C, *, s: str | None = None) -> C: return fun
@functools.partial(api_boundary2, s="foo")
def test() -> None: ...
"#,
);

// Regression: https://github.com/facebook/pyrefly/issues/3638
// `partial(f)` binds nothing, so the overloaded `f` is forwarded unchanged and resolves at the
// decoration site, preserving `g`'s signature.
functools_testcase!(
    test_partial_overloaded_decorator_erases_signature,
    r#"
from typing import Callable, Any, overload, reveal_type
from functools import partial
@overload
def f[C: Callable[..., Any]](x: C) -> C: ...
@overload
def f[C: Callable[..., Any]]() -> Callable[[C], C]: ...
def f[C: Callable[..., Any]](x: C | None = None) -> C | Callable[[C], C]: ...
@partial(f)
def g(x: int) -> str: ...
reveal_type(g)  # E: revealed type: (x: int) -> str
g(5)
"#,
);

// Regression: https://github.com/facebook/pyrefly/issues/3546
// Binding the enclosing-scope `factory` unifies its `_S` with `build`'s `_S`, so the residual keeps a
// single `_S` and `run(partial_fn)` is `Box[_S]`, matching the declared return with no leaked residual.
functools_testcase!(
    test_partial_generic_factory_residual_leak,
    r#"
import functools
from typing import Callable, Generic, TypeVar, reveal_type
_S = TypeVar('_S')
class Box(Generic[_S]): pass
def build(x: int, factory: Callable[[], _S]) -> _S: return factory()
def run(f: Callable[[int], _S]) -> Box[_S]: return Box()
def test(factory: Callable[[], _S]) -> Box[_S]:
    partial_fn = functools.partial(build, factory=factory)
    reveal_type(partial_fn)  # E: revealed type: (x: int, *, factory: () -> _S = ...) -> _S
    reveal_type(run(partial_fn))  # E: revealed type: Box[_S]
    return run(partial_fn)
"#,
);
