/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

// At some point in the past, this test took many minutes and consumed 50Gb of RAM.
testcase!(
    test_quadratic,
    r#"
from typing import TypeVar
_T = TypeVar("_T")
def table() -> _T: ...
class Configs:
    def __init__(self):
        self.value = {
            "1": {
                "a": table(),
                "b": table(),
                "c": table(),
                "d": table(),
                "e": table(),
                "f": table(),
                "g": table(),
                "h": table(),
                "i": table(),
                "j": table(),
            },
            "2": {
                "a": table(),
                "b": table(),
                "c": table(),
                "d": table(),
                "e": table(),
                "f": table(),
                "g": table(),
                "h": table(),
                "i": table(),
            },
            "3": {},
        }
"#,
);

// Repro for issue #2595: Protocol-vs-Protocol structural subtyping cycle.
// Without memoization, checking `Impl <: Array1` fans out into 10 member checks,
// each recursively requiring `Impl <: Array2`, each of those requiring `Impl <: Array3`,
// etc. This creates O(M^D) work (M=methods, D=protocol chain depth). With the
// subset_cache, each (class, protocol) pair is checked at most once, collapsing to O(M*D).
testcase!(
    test_protocol_cycle_exponential,
    r#"
from typing import Protocol

class Array1(Protocol):
    def op_0(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_1(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_2(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_3(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_4(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_5(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_6(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_7(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_8(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_9(self, other: "Array1 | complex", /) -> "Array2": ...

class Array2(Protocol):
    def op_0(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_1(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_2(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_3(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_4(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_5(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_6(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_7(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_8(self, other: "Array2 | complex", /) -> "Array3": ...
    def op_9(self, other: "Array2 | complex", /) -> "Array3": ...

class Array3(Protocol):
    def op_0(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_1(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_2(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_3(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_4(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_5(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_6(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_7(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_8(self, other: "Array3 | complex", /) -> "Array1": ...
    def op_9(self, other: "Array3 | complex", /) -> "Array1": ...

class Impl:
    def op_0(self, other: object, /) -> "Impl": ...
    def op_1(self, other: object, /) -> "Impl": ...
    def op_2(self, other: object, /) -> "Impl": ...
    def op_3(self, other: object, /) -> "Impl": ...
    def op_4(self, other: object, /) -> "Impl": ...
    def op_5(self, other: object, /) -> "Impl": ...
    def op_6(self, other: object, /) -> "Impl": ...
    def op_7(self, other: object, /) -> "Impl": ...
    def op_8(self, other: object, /) -> "Impl": ...
    def op_9(self, other: object, /) -> "Impl": ...

def f1(x: Array1 | complex) -> Array1: ...
def f2(x: Array2 | complex) -> Array2: ...
def f3(x: Array3 | complex) -> Array3: ...

def test() -> None:
    val = Impl()
    a1 = f1(val)
    a2 = f2(val)
    a3 = f3(val)
    c1 = f1(a2)
    c2 = f2(a3)
    c3 = f3(a1)
"#,
);

// Minimal test for cross-protocol structural subtyping.
testcase!(
    test_protocol_cross_check_minimal,
    r#"
from typing import Protocol

class P1(Protocol):
    def foo(self, other: "P1 | complex", /) -> "P2": ...

class P2(Protocol):
    def foo(self, other: "P2 | complex", /) -> "P1": ...

def make_p2() -> P2: ...
def f(x: P1) -> None: ...

def test() -> None:
    p = make_p2()
    f(p)
"#,
);

// Soundness test for coinductive cache invalidation in subset_cache.
//
// Without rollback-on-failure, a naive persistent cache produces a false
// positive here. The scenario:
//   1. Check `A <: P1 | P2` — union handler tries `A <: P1` first.
//   2. `A <: P1` inserts InProgress. Method `foo` return type requires `A <: P2`.
//   3. `A <: P2` inserts InProgress. Method `foo` return type requires `A <: P1`.
//   4. `A <: P1` is InProgress → coinductive Ok. `A <: P2` succeeds, cached as Ok.
//   5. Back in `A <: P1`: method `bar` fails (A lacks `bar`). `A <: P1` fails.
//   6. Union handler tries `A <: P2` — stale cached Ok → false positive!
//
// The rollback mechanism rolls back the cached Ok for `A <: P2` (step 4) when
// `A <: P1` fails (step 5), because `A <: P2` was computed during `A <: P1`'s
// computation and may have depended on the coinductive assumption that `A <: P1`
// holds. Re-checking `A <: P2` then correctly fails.
testcase!(
    test_protocol_coinductive_cache_soundness,
    r#"
from typing import Protocol

class P1(Protocol):
    def foo(self) -> "P2": ...
    def bar(self) -> int: ...

class P2(Protocol):
    def foo(self) -> "P1": ...

class A:
    def foo(self) -> "A": ...

def f(x: P1 | P2) -> None: ...

def test() -> None:
    f(A())  # E: Argument `A` is not assignable to parameter `x` with type `P1 | P2`
"#,
);
