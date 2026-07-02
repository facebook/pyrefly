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

functools_testcase!(
    bug = "partial does not propagate a single shared TypeVar: the call reveals Unknown instead of int, and the bad-arg call is not flagged",
    test_partial_generic_same_typevar,
    r#"
from typing import TypeVar, reveal_type
import functools
T = TypeVar("T")
def foo(a: T, b: T) -> T: ...
p1 = functools.partial(foo, 1)
# WANT: revealed type: int
reveal_type(p1(2))  # E: revealed type: Unknown
p1("a")  # WANT: Argument 1 to "foo" has incompatible type "str"; expected "int"
p2 = functools.partial(foo, "a")
p2(1)  # WANT: Argument 1 to "foo" has incompatible type "int"; expected "str"
# WANT: revealed type: str
reveal_type(p2("a"))  # E: revealed type: Unknown
"#,
);

functools_testcase!(
    bug = "partial does not propagate distinct TypeVars: both calls reveal Unknown instead of the solved int/str",
    test_partial_generic_two_typevars,
    r#"
from typing import TypeVar, reveal_type
import functools
T = TypeVar("T")
U = TypeVar("U")
def bar(a: T, b: U) -> U: ...
p3 = functools.partial(bar, 1)
# WANT: revealed type: int
reveal_type(p3(2))  # E: revealed type: Unknown
# WANT: revealed type: str
reveal_type(p3("a"))  # E: revealed type: Unknown
"#,
);

functools_testcase!(
    bug = "partial over a generic function loses the type variable: calls return Unknown instead of the solved element type, and an invalid call is not flagged",
    test_partial_of_generic_function,
    r#"
from functools import partial
from typing import TypeVar, List, reveal_type
T = TypeVar("T")
def get(n: int, args: List[T]) -> T: ...
first = partial(get, 0)
x: List[str] = []
# WANT: revealed type: str
reveal_type(first(x))  # E: revealed type: Unknown
# WANT: revealed type: int
reveal_type(first([1]))  # E: revealed type: Unknown
first_kw = partial(get, n=0)
# WANT: revealed type: int
reveal_type(first_kw(args=[1]))  # E: revealed type: Unknown
first_kw([1])  # WANT: Too many positional arguments / Too few arguments / Argument 1 incompatible
"#,
);

// ===== Constrained TypeVar values =====

functools_testcase!(
    bug = "partial with a constrained-TypeVar function is broken: every call errors with bad-specialization and reveals Unknown instead of resolving the constraint",
    test_partial_type_var_values_f,
    r#"
from functools import partial
from typing import TypeVar, reveal_type
T = TypeVar("T", int, str)
def f(x: int, y: T) -> T:
    return y
fp = partial(f, 1)
# WANT: revealed type: int
reveal_type(fp(1))  # E: revealed type: Unknown # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
# WANT: revealed type: str
reveal_type(fp("a"))  # E: revealed type: Unknown # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
# WANT: no error (y=1 satisfies T=int)
fp(1)  # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
# WANT: no error (y="a" satisfies T=str)
fp("a")  # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
# WANT: Value of type variable "T" of "f" cannot be "object"
fp(object())  # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
"#,
);

functools_testcase!(
    bug = "partial with a constrained TypeVar in the bound position errors with bad-specialization instead of solving T=int",
    test_partial_type_var_values_g,
    r#"
from functools import partial
from typing import TypeVar, reveal_type
T = TypeVar("T", int, str)
def g(x: T, y: int) -> T:
    return x
gp = partial(g, 1)
# WANT: revealed type: int
reveal_type(gp(1))  # E: revealed type: Unknown # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
# WANT: no error
gp(1)  # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
# WANT: Argument 1 to "g" has incompatible type "str"; expected "int"
gp("a")  # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
"#,
);

functools_testcase!(
    bug = "partial with a constrained TypeVar shared across params errors with bad-specialization instead of solving T=int",
    test_partial_type_var_values_h,
    r#"
from functools import partial
from typing import TypeVar, reveal_type
T = TypeVar("T", int, str)
def h(x: T, y: T) -> T:
    return x
hp = partial(h, 1)
# WANT: revealed type: int
reveal_type(hp(1))  # E: revealed type: Unknown # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
# WANT: no error
hp(1)  # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
# WANT: Argument 1 to "h" has incompatible type "str"; expected "int"
hp("a")  # E: `GenericResidual@T` is not assignable to any of constraints `int`, `str` of type variable `T`
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

functools_testcase!(
    bug = "partial should erase out-of-scope typevars to Any, but pyrefly leaks GenericResidual@ typevars (so the downstream Callable assignability is judged against the residual)",
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
# WANT: revealed type: partial[Any]
reveal_type(partial(func_b, b=""))  # E: revealed type: partial[GenericResidual@Tb]
# WANT: revealed type: partial[Any]
reveal_type(partial(func_c, b=""))  # E: revealed type: partial[GenericResidual@Tc]
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

functools_testcase!(
    bug = "a TypeVar bound by the enclosing function (not by the partial target) is correctly NOT erased; the downstream incompatible use is flagged",
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
    reveal_type(partial(inner, b=""))  # E: revealed type: partial[Tb]
    use_int_callable(partial(inner, b=""))  # E: Argument `partial[Tb]` is not assignable to parameter `x` with type `(int) -> int` in function `use_int_callable`
"#,
);

functools_testcase!(
    bug = "an in-scope constrained TypeVar is left as partial[Tc] rather than being expanded to partial[int]/partial[str]",
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
    reveal_type(partial(inner, b=""))  # E: revealed type: partial[Tc]
    use_int_callable(partial(inner, b=""))  # E: Argument `partial[Tc]` is not assignable to parameter `x` with type `(int) -> int` in function `use_int_callable`
"#,
);

// ===== Overloaded targets =====

functools_testcase!(
    bug = "partial over an overloaded function resolves both call shapes to the first overload's return (int) and does not flag invalid arg combinations",
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
# WANT: revealed type: str (matches the second overload)
reveal_type(p1("a", 1))  # E: revealed type: int
p1(1, 2)  # WANT: error - no overload matches (a: int, b: int)
p1("a", "b")  # WANT: error - no overload matches (a: str, b: str)
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
    bug = "functools.partial used as a decorator erases the decorated function to Unknown, so its signature is lost and the bad call is missed",
    test_partial_decorator_erases_signature,
    r#"
import functools
from typing import TypeVar, reveal_type
C = TypeVar("C")
def decorator(fn: C, s: str) -> C: return fn
@functools.partial(decorator, s="foo")
def f(x: int) -> int: return x
# WANT: revealed type: (x: int) -> int
reveal_type(f)  # E: revealed type: Unknown
f(None)  # WANT: Argument `None` is not assignable to parameter `x` with type `int`
"#,
);

// Regression: https://github.com/facebook/pyrefly/issues/3329
functools_testcase!(
    bug = "False positive: partial binding the keyword-only `s` of a generic callable spuriously reports GenericResidual@C not assignable to C's upper bound",
    test_partial_generic_decorator_kwonly_false_positive,
    r#"
import functools
from typing import TypeVar, Callable
C = TypeVar("C", bound=Callable)
def api_boundary2(fun: C, *, s: str | None = None) -> C: return fun
# WANT: no error (partial only binds keyword-only `s`; `fun` stays free, bound at decoration time)
@functools.partial(api_boundary2, s="foo")  # E: `GenericResidual@C` is not assignable to upper bound `(...) -> Unknown` of type variable `C`
def test() -> None: ...
"#,
);

// Regression: https://github.com/facebook/pyrefly/issues/3638
functools_testcase!(
    bug = "partial(f) over an overloaded generic decorator erases g's signature to Unknown instead of preserving (x: int) -> str",
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
# WANT: revealed type: (x: int) -> str
reveal_type(g)  # E: revealed type: Unknown
g(5)  # WANT: no error (signature unchanged, g(5) is valid)
"#,
);

// Regression: https://github.com/facebook/pyrefly/issues/3546
functools_testcase!(
    bug = "partial leaks GenericResidual@_T instead of binding _S, causing a false-positive bad-return",
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
    # WANT: revealed type: partial[_S]
    reveal_type(partial_fn)  # E: revealed type: partial[GenericResidual@_T]
    # WANT: revealed type: Box[_S]
    reveal_type(run(partial_fn))  # E: revealed type: Box[GenericResidual@_T]
    # WANT: no error (run(partial_fn) is Box[_S], matches the declared return)
    return run(partial_fn)  # E: Returned type `Box[GenericResidual@_T]` is not assignable to declared return type `Box[_S]`
"#,
);
