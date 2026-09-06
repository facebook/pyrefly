/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `functools.partial` edge cases beyond the core `partial.rs` bank, plus the closed
//! issue #149 regression. Divergences are `bug=`-marked with the correct behavior recorded inline
//! as `# WANT: ...`.

use crate::functools_testcase;
use crate::test::util::TestEnv;
use crate::testcase;

// Regression: https://github.com/facebook/pyrefly/issues/149 (closed — pyrefly already passes)
functools_testcase!(
    test_partial_keyword_bind_callable_arg,
    r#"
from __future__ import annotations
from functools import partial
from typing import Callable, Match
def bar(a: Match[str], b: int) -> str: return f'{a}{b}'
def zoo(a: Callable[[Match[str]], str]) -> None: return None
zoo(partial(bar, b=99))
"#,
);

functools_testcase!(
    test_partial_edge_construct_too_many_bound,
    r#"
from typing import reveal_type
import functools
def target(a: int, b: str, c: float) -> bytes: return b""
p = functools.partial(target, 1, "x", 2.0, 99)  # E: Expected 3 positional arguments, got 4 in function `target`
reveal_type(p)  # E: revealed type: partial[bytes]
r = p()
reveal_type(r)  # E: revealed type: bytes
"#,
);

functools_testcase!(
    test_partial_edge_construct_duplicate_bound_kw,
    r#"
from typing import reveal_type
import functools
def target(a: int, b: int) -> int: return 0
p = functools.partial(target, 1, a=2)  # E: Multiple values for argument `a` in function `target`
reveal_type(p)  # E: revealed type: partial[int]
"#,
);

functools_testcase!(
    test_partial_edge_construct_bound_kwonly_wrong_type,
    r#"
from typing import reveal_type
import functools
def kwonly(a: int, *, b: str) -> int: return 0
p = functools.partial(kwonly, 1, b=5)  # E: Argument `Literal[5]` is not assignable to parameter `b` with type `str` in function `kwonly`
reveal_type(p)  # E: revealed type: (*, b: str = ...) -> int
"#,
);

functools_testcase!(
    test_partial_edge_call_missing_remaining,
    r#"
from typing import reveal_type
import functools
def f(a: int, b: str) -> bytes: return b""
p = functools.partial(f, 1)
reveal_type(p)  # E: revealed type: (b: str) -> bytes
p()  # E: Missing argument `b`
"#,
);

functools_testcase!(
    test_partial_edge_target_bound_method_badcall,
    r#"
from typing import reveal_type
import functools
class C:
    def m(self, a: int, b: str) -> float: return 0.0
p = functools.partial(C().m, 1)
reveal_type(p)  # E: revealed type: (b: str) -> float
p(2)  # E: Argument `Literal[2]` is not assignable to parameter `b` with type `str`
"#,
);

functools_testcase!(
    test_partial_edge_target_typed_lambda_badcall,
    r#"
import functools
from typing import Callable, reveal_type
g: Callable[[int, str], bytes] = lambda a, b: b""
p = functools.partial(g, 1)
reveal_type(p)  # E: revealed type: (str) -> bytes
p(2)  # E: Argument `Literal[2]` is not assignable to parameter with type `str`
"#,
);

functools_testcase!(
    test_partial_edge_positional_only_marker,
    r#"
from typing import reveal_type
import functools
def g(a: int, b: int, /) -> bytes: return b""
p = functools.partial(g, 1)
reveal_type(p)  # E: revealed type: (b: int, /) -> bytes
p(b=2)  # E: Expected argument `b` to be positional
g(1, b=2)  # E: Expected argument `b` to be positional in function `g`
"#,
);

functools_testcase!(
    test_partial_edge_keyword_only_marker_positional,
    r#"
import functools
from typing import assert_type
def k(a: int, *, b: str) -> bytes: return b""
p = functools.partial(k)
assert_type(p(1, b="x"), bytes)
p(1, "x")  # E: Expected argument `b` to be passed by name
"#,
);

functools_testcase!(
    bug = "nested partial loses precision: partial(partial, foo) returns Unknown, so the inner call's return type and arg-type checking are both lost",
    test_partial_edge_nested_partial_wrongtype,
    r#"
from typing import reveal_type
import functools
def foo(x: int) -> int: return x
p = functools.partial(functools.partial, foo)
# WANT: revealed type: int
reveal_type(p()(1))  # E: revealed type: Unknown
p()("no")  # WANT: Argument "no" to "foo" has incompatible type "str"; expected "int"
"#,
);

// ===== Inheritance dimension =====

// The residual of a partial over a function with a base-class parameter must accept a subclass
// instance (Liskov) and reject an unrelated type. Binding one positional leaves the base-class
// parameter in the residual; subtype substitution must still hold there.
functools_testcase!(
    test_partial_edge_base_param_accepts_subclass,
    r#"
from typing import reveal_type
import functools
class Base: ...
class Sub(Base): ...
class Other: ...
def sink(tag: int, node: Base) -> bytes: return b""
p = functools.partial(sink, 1)
reveal_type(p)  # E: revealed type: (node: Base) -> bytes
p(Sub())
p(Base())
p(Other())  # E: Argument `Other` is not assignable to parameter `node` with type `Base`
"#,
);

// Binding the base-class positional with a subclass instance is accepted at construction. The
// same-typevar residual stays generic, so a later `Base()` re-solves `T` to `Base` (matching the
// direct call `pick(Sub(), Base())`) rather than being frozen to `Sub`.
functools_testcase!(
    test_partial_edge_bind_subclass_to_generic_base,
    r#"
from typing import TypeVar, reveal_type
import functools
class Base: ...
class Sub(Base): ...
T = TypeVar("T", bound=Base)
def pick(first: T, second: T) -> T: ...
p = functools.partial(pick, Sub())
reveal_type(p(Sub()))  # E: revealed type: Sub
reveal_type(p(Base()))  # E: revealed type: Base
"#,
);

// A partial over a bound method reached through a subclass that overrides it currently defers to
// the stub (bound-method targets are deferred), so no residual arg-checking happens.
functools_testcase!(
    test_partial_edge_overridden_bound_method,
    r#"
from typing import reveal_type
import functools
class Base:
    def act(self, a: int, b: str) -> float: return 0.0
class Sub(Base):
    def act(self, a: int, b: str) -> float: return 1.0
p = functools.partial(Sub().act, 1)
reveal_type(p)  # E: revealed type: (b: str) -> float
p(2)  # E: Argument `Literal[2]` is not assignable to parameter `b` with type `str`
"#,
);

// ===== Cross-module dimension: generic target imported across the module boundary =====

testcase!(
    test_partial_cross_module_generic,
    TestEnv::one(
        "ghelper",
        "from typing import TypeVar, List\n_T = TypeVar('_T')\ndef nth(n: int, xs: List[_T]) -> _T: ...",
    ).enable_strict_partial_subtyping(),
    r#"
from typing import reveal_type
import functools
from ghelper import nth
first = functools.partial(nth, 0)
reveal_type(first([1]))  # E: revealed type: int
reveal_type(first(["a"]))  # E: revealed type: str
"#,
);

// A class object whose `__init__` is inherited from a base in another module is reduced to that
// constructor, so the residual is checked against the inherited signature.
testcase!(
    test_partial_xmod_class_object_inherited_init,
    TestEnv::one(
        "basemod",
        "class Base:\n    def __init__(self, a: int, b: str) -> None: ...\n",
    )
    .enable_strict_partial_subtyping(),
    r#"
from typing import reveal_type
import functools
from basemod import Base
class Sub(Base):
    pass
p = functools.partial(Sub, 1)
reveal_type(p)  # E: revealed type: (b: str) -> Sub
p("ok")
p(2)  # E: Argument `Literal[2]` is not assignable to parameter `b` with type `str`
"#,
);

// An enclosing-scope TypeVar that appears in BOTH a keyword-bound parameter (`make`, unified with
// the outer `factory`'s `_S`) AND a required residual positional (`x`) must not freeze `x` to a rigid
// `_S`: the residual call `pf(5)` re-solves it exactly as the direct call `pair(5, factory)` does.
functools_testcase!(
    test_partial_kwbind_and_required_positional_same_typevar,
    r#"
import functools
from typing import Callable, TypeVar, reveal_type
_S = TypeVar('_S')
def pair(x: _S, make: Callable[[], _S]) -> _S: return x
def test(factory: Callable[[], _S]) -> None:
    reveal_type(pair(5, factory))  # E: revealed type: int | _S
    pf = functools.partial(pair, make=factory)
    reveal_type(pf(5))  # E: revealed type: int
"#,
);

// Cross-module variant of the kw-bind + required-positional enclosing-scope-TypeVar case.
testcase!(
    test_partial_xmod_kwbind_and_required_positional,
    TestEnv::one(
        "pmod",
        "from typing import Callable, TypeVar\n_S = TypeVar('_S')\ndef pair(x: _S, make: Callable[[], _S]) -> _S: return x\n",
    ).enable_strict_partial_subtyping(),
    r#"
import functools
from typing import Callable, TypeVar, reveal_type
from pmod import pair
_S = TypeVar('_S')
def test(factory: Callable[[], _S]) -> None:
    reveal_type(pair(5, factory))  # E: revealed type: int | _S
    pf = functools.partial(pair, make=factory)
    reveal_type(pf(5))  # E: revealed type: int
"#,
);

// Cross-module #3546 enclosing-scope unification: `build` (own `_S`) imported; caller's `_S` is a
// DIFFERENT locally-defined TypeVar. Binding enclosing-scope `factory` must unify.
testcase!(
    test_partial_xmod_factory_unify,
    TestEnv::one(
        "bhelper",
        "from typing import Callable, TypeVar\n_S = TypeVar('_S')\ndef build(x: int, factory: Callable[[], _S]) -> _S: return factory()\n",
    ).enable_strict_partial_subtyping(),
    r#"
import functools
from typing import Callable, Generic, TypeVar, reveal_type
from bhelper import build
_S = TypeVar('_S')
class Box(Generic[_S]): pass
def run(f: Callable[[int], _S]) -> Box[_S]: return Box()
def test(factory: Callable[[], _S]) -> Box[_S]:
    partial_fn = functools.partial(build, factory=factory)
    reveal_type(partial_fn)  # E: revealed type: (x: int, *, factory: () -> _S = ...) -> _S
    reveal_type(run(partial_fn))  # E: revealed type: Box[_S]
    return run(partial_fn)
"#,
);

// Cross-module #3330 decorator, imported.
testcase!(
    test_partial_xmod_decorator,
    TestEnv::one(
        "dhelper",
        "from typing import TypeVar\nC = TypeVar('C')\ndef decorator(fn: C, s: str) -> C: return fn\n",
    ).enable_strict_partial_subtyping(),
    r#"
import functools
from typing import reveal_type
from dhelper import decorator
@functools.partial(decorator, s="foo")
def f(x: int) -> int: return x
reveal_type(f)  # E: revealed type: (x: int) -> int
f(None)  # E: Argument `None` is not assignable to parameter `x` with type `int` in function `f`
"#,
);

// Cross-module #3329 bounded-TypeVar decorator, imported. Must be zero errors.
testcase!(
    test_partial_xmod_bounded_decorator,
    TestEnv::one(
        "ahelper",
        "from typing import TypeVar, Callable\nC = TypeVar('C', bound=Callable)\ndef api_boundary2(fun: C, *, s: str | None = None) -> C: return fun\n",
    ).enable_strict_partial_subtyping(),
    r#"
import functools
from ahelper import api_boundary2
@functools.partial(api_boundary2, s="foo")
def test() -> None: ...
"#,
);

// Partial over a local generic whose return uses an IMPORTED generic class `Box[_S]`.
testcase!(
    test_partial_xmod_generic_box_return,
    TestEnv::one(
        "boxmod2",
        "from typing import Generic, TypeVar\n_S = TypeVar('_S')\nclass Box(Generic[_S]):\n    def __init__(self, v: _S) -> None: self.v = v\n",
    ).enable_strict_partial_subtyping(),
    r#"
from typing import TypeVar, reveal_type
import functools
from boxmod2 import Box
_S = TypeVar('_S')
def wrap(tag: int, v: _S) -> Box[_S]: ...
p = functools.partial(wrap, 1)
reveal_type(p("x"))  # E: revealed type: Box[str]
reveal_type(p(5))  # E: revealed type: Box[int]
"#,
);

// Decorator with a TypeVar bound to an IMPORTED base class, applied cross-module.
testcase!(
    test_partial_xmod_decorator_imported_bound,
    TestEnv::one("basemod3", "class MyBase: ...\n",).enable_strict_partial_subtyping(),
    r#"
import functools
from typing import TypeVar, Callable, reveal_type
from basemod3 import MyBase
C = TypeVar("C", bound=Callable[..., MyBase])
def deco(fn: C, s: str) -> C: return fn
class Impl(MyBase): ...
@functools.partial(deco, s="foo")
def make(x: int) -> Impl: ...
reveal_type(make)  # E: revealed type: (x: int) -> Impl
make("bad")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int` in function `make`
"#,
);

// A `functools.partial` conforms to a partial-shaped `Protocol`; its `__class_getitem__`, a class
// hook rather than an instance member, must not block conformance.
functools_testcase!(
    test_partial_conforms_to_partial_protocol,
    r#"
from typing import Protocol, Any, Callable, TypeVar, runtime_checkable
import functools
import types
T_co = TypeVar("T_co", covariant=True)
@runtime_checkable
class Partial(Protocol[T_co]):
    __call__: Callable[..., T_co]
    @property
    def func(self) -> Callable[..., T_co]: ...
    @property
    def args(self) -> tuple[Any, ...]: ...
    @property
    def keywords(self) -> dict[str, Any]: ...
    def __class_getitem__(cls, item: Any) -> types.GenericAlias: ...
def f() -> int: ...
def use(p: "functools.partial[int]") -> None:
    x: Partial[int] = p
q: Partial[int] = functools.partial(f)
"#,
);
