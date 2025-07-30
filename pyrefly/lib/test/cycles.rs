/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

/*
Leaky loop tests: Some of these are genuinely nondeterministic in cargo (they
normally pass, but might not when run in the full test suite with threading
because the check of `main` can race the check of `leaky_loop`) and so they are
commented out.

They demonstrate that we get nondeterminism from loop recursion, even with
no placeholder types involved. The loop creates a cycle in the definition of
`x`, and depending where we start the cycle we can get different answers.

The one I've left uncommented is the one where there's no race condition.
*/

fn env_leaky_loop() -> TestEnv {
    TestEnv::one(
        "leaky_loop",
        r#"
x = None
def f(_: str | None) -> tuple[str, str]: ...
def g(_: int | None) -> tuple[int, int]: ...
while True: # E: `int | None` is not assignable to `str | None` (caused by inconsistent types when breaking cycles)
    y, x = f(x)
    z, x = g(x) # E: Argument `str` is not assignable to parameter `_` with type `int | None` in function `g`
"#,
    )
}

testcase!(
    bug = "If we don't force anything, x will come back as `int`.",
    try_leaky_loop_and_import_x,
    env_leaky_loop(),
    r#"
from typing import assert_type, Any
from leaky_loop import x
assert_type(x, int | None)
"#,
);

testcase!(
    bug = "Forcing `y` first gives us `int` for `x`",
    try_leaky_loop_and_import_y,
    env_leaky_loop(),
    r#"
from typing import assert_type, Any
from leaky_loop import y
assert_type(y, str)
from leaky_loop import x
assert_type(x, int | None)
"#,
);

/*
The variant of this test that exercises an actual race condition can potentially
give nondeterministic output so it is commented for CI stability.

testcase!(
    bug = "Forcing `z` first gives us `Any` for `x`",
    try_leaky_loop_and_import_z,
    env_leaky_loop(),
    r#"
from typing import assert_type, Any
from leaky_loop import z
assert_type(z, int)
from leaky_loop import x
assert_type(x, Any | None)
"#,
);
*/

/*
Import cycle tests: We can create a cycle of imports pretty easily. If we never
do anything with imported names except forward them, we won't be able to exhibit
nondeterminism because the answer to everything is just `Any` regardless of orders.

But if anything in the cycle is able to actually compute a result (for example,
because it makes a function call that takes a cyclic argument, but the function
itself has a well-defined return type), we will see nondeterminism, because
- If we break the cycle on exactly that element, it will spit out a recursive
  `Var` from the point of view of its dependents, which when forced is typically
  `Any`.
- If we break the cycle anywhere else, the function call will be evaluated and
  we'll spit out a concrete answer (the same concrete answer we'll eventually
  get in the other case when we unwind the cycle back to ourselves), and our
  dependents will see that.
- Note that the nondeterminism *originates* from the place where we break
  recursion, but the *visible effects* occur in the dependents of that element,
  not the element itself.

Unlike the leaky loop tests, these have no variations that aren't potentially
subject to race conditions, so they are all commented out for CI stability.

fn env_import_cycle() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "xx",
        r#"
from yy import y

def f[T](arg: T) -> T: ...
def g(_: object) -> int: ...
x0 = f(y)
x1 = g(x0)
"#,
    );
    env.add(
        "yy",
        r#"
from xx import x1

def f[T](arg: T) -> T: ...
def g(_: object) -> int: ...
y = f(x1)
"#,
    );
    env
}

testcase!(
    import_cycle_a,
    env_import_cycle(),
    r#"
from typing import assert_type, Any
from xx import x0
assert_type(x0, int)
from yy import y
assert_type(y, int)
from xx import x1
assert_type(y, int)
"#,
);

testcase!(
    import_cycle_b,
    env_import_cycle(),
    r#"
from typing import assert_type, Any
from xx import x1
assert_type(y, Any)
from yy import y
assert_type(y, Any)
from xx import x0
assert_type(x0, Any)
"#,
);

testcase!(
    import_cycle_c,
    env_import_cycle(),
    r#"
from typing import assert_type, Any
from yy import y
assert_type(y, int)
from xx import x1
assert_type(y, int)
from xx import x0
assert_type(x0, Any)
"#,
);

testcase!(
    import_cycle_d,
    env_import_cycle(),
    r#"
from typing import assert_type, Any
from yy import y
assert_type(y, int)
from xx import x0
assert_type(x0, object)
from xx import x1
assert_type(y, Any)
"#,
);
*/

// This pair of tests shows that fully annotating modules eliminates
// nondeterminism from import cycles of globals defined with assignment.
//
// The determinism we get relies on lazy evaluation of the flow type
// for annotated exports, so it's worth having regression tests.

fn env_import_cycle_annotated() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "xx",
        r#"
from yy import yyy
def fx(arg: int) -> int: ...
xxx: bytes = fx(yyy) # E: `int` is not assignable to `bytes` # E: Argument `bytes` is not assignable to parameter `arg` with type `int`
"#,
    );
    env.add(
        "yy",
        r#"
from xx import xxx
def fy(arg: str) -> str: ...
yyy: bytes = fy(xxx) # E: `str` is not assignable to `bytes` # E: Argument `bytes` is not assignable to parameter `arg` with type `str`
"#,
    );
    env
}

testcase!(
    import_cycle_annotated_a,
    env_import_cycle_annotated(),
    r#"
from typing import assert_type, Any
from yy import yyy
assert_type(yyy, bytes)
from xx import xxx
assert_type(xxx, bytes)
"#,
);

testcase!(
    import_cycle_annotated_b,
    env_import_cycle_annotated(),
    r#"
from typing import assert_type, Any
from xx import xxx
assert_type(xxx, bytes)
from yy import yyy
assert_type(yyy, bytes)
"#,
);

// The following tests demonstrate that decorator cycles exhibit nondeterminism.
//
// The unit tests themselves are deterministic, because the `main` module (which
// I need outside the cycle for my own sanity) isn't participating directly in
// the cycle, and everything here has a concrete type once it's fully resolved.
//
// But if you run with `--nocapture`, you'll see that the type errors for
// the `xx` and `yy` modules are not consistent between the two tests:
// - In version (a) we get no type errors in xx and a type error in yy
// - In version (b) we get no type errors in yy and a type error in xx
//
// The root cause of the error is that whichever of `fx` / `fy` *doesn't* break
// the cycle winds up with type `int` prior to `@dec` being applied, but
// whichever one *does* break it has type `Any` (until the cycle completes).

/*
fn env_import_cycle_decorators() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "xx",
        r#"
from typing import Callable, Any
from yy import fy
def dec(
    arg: Callable[[Callable[..., int]], Callable[..., int]]
) -> Callable[..., int]: ...
@dec  # Sometimes an error, depends on the cycle resolution order
@fy
def fx(arg: Callable[..., Any]) -> Callable[..., Any]: ...
"#,
    );
    env.add(
        "yy",
        r#"
from typing import Callable, Any
from xx import fx
def dec(
    arg: Callable[[Callable[..., int]], Callable[..., int]]
) -> Callable[..., int]: ...
@dec  # Sometimes an error, depends on the cycle resolution order
@fx
def fy(arg: Callable[..., Any]) -> Callable[..., Any]: ...
"#,
    );
    env
}

testcase!(
    bug = "Type errors reported in xx / yy differ between versions (a) and (b) (run with --nocapture)",
    import_cycle_decorators_a,
    env_import_cycle_decorators(),
    r#"
from typing import assert_type, Callable
from yy import fy
assert_type(fy, Callable[..., int])
from xx import fx
assert_type(fx, Callable[..., int])
"#,
);

testcase!(
    bug = "Type errors reported in xx / yy differ between versions (a) and (b) (run with --nocapture)",
    import_cycle_decorators_b,
    env_import_cycle_decorators(),
    r#"
from typing import assert_type, Callable
from xx import fx
assert_type(fx, Callable[..., int])
from yy import fy
assert_type(fy, Callable[..., int])
"#,
);
*/

testcase!(
    bug = "This cycle is deterministic but ill-behaved. Both speculative Phi and narrowing pinning Var are contributing",
    test_inconsistent_types_from_cycle_in_loop,
    r#"
from typing import Iterable, Iterator, cast

def iterate[T](*items: T | Iterable[T]) -> Iterator[T]:
    for item in items:  # E: `Iterable[T] | str | T` is not assignable to `Iterable[T] | T` (caused by inconsistent types when breaking cycles)
        if isinstance(item, str):
            yield cast(T, item)
        elif isinstance(item, Iterable):
            yield from item
        else:
            yield item
"#,
);

// This pair of tests failed until we separated Mro out from ClassMetadata - parsing base
// types depends on the metadata but not the Mro, which was leading to patterns where a base
// class in the cycle is generic over a class in the cycle to incorrectly fail to resolve
// Mro (nondeterministically, it depended on where we entered the cycle).

testcase!(
    potential_cycle_through_generic_bases_a,
    r#"
from typing import assert_type
class Node[T]:
    @property
    def x(self) -> T: ...
class A(Node['B']):
    pass
class B(A):
    pass
assert_type(B().x, B)
assert_type(A().x, B)
"#,
);

testcase!(
    potential_cycle_through_generic_bases_b,
    r#"
from typing import assert_type
class Node[T]:
    @property
    def x(self) -> T: ...
class A(Node['B']):
    pass
class B(A):
    pass
assert_type(A().x, B)
assert_type(B().x, B)
"#,
);

testcase!(
    test_init_cycle,
    r#"
from typing import reveal_type
class A:
    def __init__(self):
        self.x = 42
        self.f()
    def f(self):
        pass
reveal_type(A.__init__)  # E: revealed type: (self: Self@A) -> None
    "#,
);
