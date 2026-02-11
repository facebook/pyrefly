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

/* Improving loop handling in D85922045 made these tests once again
 * nondeterministic in cargo tests, because some percentage of the time
 * the type check for `y, x = f(x)` passes.
 *
 * This is occurring because it is entrypoint-dependent whether we type
 * check that binding once or twice, and if we type check it twice then
 * the second type check can produce a different answer due to
 * non-idempotence

 fn env_leaky_loop() -> TestEnv {
    TestEnv::one(
        "leaky_loop",
        r#"
x = None
def f(_: str | None) -> tuple[str, str]: ...
def g(_: int | None) -> tuple[int, int]: ...
while True: # E: Pyrefly detected conflicting types while breaking a dependency cycle: `int | None` is not assignable to `str | None`.
    y, x = f(x)  # E: Argument `int | None` is not assignable to parameter `_` with type `str | None` in function `f`
    z, x = g(x)  # E: Argument `str` is not assignable to parameter `_` with type `int | None` in function `g`
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
*/

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
    potential_cycle_through_generic_base_union,
    r#"
type A = Child | int
class Base[T]:
    def __init__(self, value: T) -> None:
        ...
class Child(Base[A]):  # Note how the Base targ is a union of `Child` and another class
    pass

Child("abc")  # E: not assignable to parameter `value` with type `Child | int`
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
reveal_type(A.__init__)  # E: revealed type: (self: A) -> None
    "#,
);

// Regression test for issue #1791, which was a stack overflow due to recursion
// with an infinite loop of Type::Vars.
testcase!(
    force_for_narrowing_cycle_detection,
    r#"
def f(  # E: Expected `)`, found newline
    if n:  # E: Type narrowing encountered a cycle in Type::Var # E: Expected an indented block after `if` statement
)n = min(n, size)  # E: Expected a statement # E: `n` is uninitialized # E: Could not find name `size`
"#,
);

// Regression test for issue #2175, which was infinite recursion in is_subset_eq
// when checking recursive type patterns. The cycle detection in is_subset_eq
// should prevent stack overflow.
testcase!(
    recursive_type_subset_check_no_overflow,
    r#"
from typing import Protocol, TypeVar, Generic

T = TypeVar("T", covariant=True)

# A recursive protocol that references itself
class Readable(Protocol[T]):
    def read(self) -> T: ...

# A class that implements the recursive protocol
class Stream(Generic[T]):
    def read(self) -> T: ...

def consume(x: Readable[int]) -> int:
    return x.read()

s: Stream[int] = Stream()
consume(s)  # Should not cause stack overflow
"#,
);

// Test that mutually recursive classes don't cause infinite recursion
testcase!(
    mutually_recursive_classes_subset,
    r#"
from typing import Generic, TypeVar

T = TypeVar("T")

class A(Generic[T]):
    def get_b(self) -> "B[T]": ...

class B(Generic[T]):
    def get_a(self) -> "A[T]": ...

def f(x: A[int]) -> B[int]:
    return x.get_b()

def g(x: B[int]) -> A[int]:
    return x.get_a()
"#,
);

// Regression test for forward references in class bodies causing stack overflow.
// When a class has many fields where field[i] references field[i+1] (forward reference),
// binding_to_type_class_body_unknown_name calls get_class_field_map just to check if
// the name is a class field. This triggers computing ALL field types, and when a field's
// type computation involves resolving another forward reference, it re-enters
// get_class_field_map quadratically.

const NUM_FORWARD_REF_FIELDS: usize = 600;

fn env_forward_reference_class_fields() -> TestEnv {
    let mut code = String::from(
        r#"
class Wrapper:
    def __init__(self, value: "Wrapper | None") -> None:
        self.value = value

class Container:
"#,
    );

    // Generate 600 fields where each references the next (forward reference)
    for i in 0..NUM_FORWARD_REF_FIELDS {
        if i < NUM_FORWARD_REF_FIELDS - 1 {
            // Forward reference to the next field - currently produces unknown-name error
            code.push_str(&format!(
                "    FIELD_{} = Wrapper(FIELD_{})  # E: Could not find name `FIELD_{}`\n",
                i,
                i + 1,
                i + 1
            ));
        } else {
            // Last field has no forward reference
            code.push_str(&format!("    FIELD_{} = Wrapper(None)\n", i));
        }
    }

    TestEnv::one("forward_refs", &code)
}

testcase!(
    bug = "Forward references in class bodies cause O(n^2) recursion depth, stack overflow at ~570 fields",
    forward_reference_class_fields,
    env_forward_reference_class_fields(),
    r#"
from forward_refs import Container
"#,
);

// A small reproduction from parso/python/tokenize.py of a stack overflow
// that we hit when making changes to SCC resolution; useful to have in the
// unit tests to ensure we don't repeat the same bug. We spent several hours
// minimizing the repro to ~85 lines, it may be possible to further minimize
// but this seems good, the goal is just to have the unit test suite ensure
// we don't crash.
//
// The actual errors are not of any particular interest, we care about the
// binding graph traversal here.
testcase!(
    tokenize_minimal_scc_stack_overflow,
    r#"
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, Pattern, Set

class Token(NamedTuple):
    type: int
    string: str
    start_pos: Tuple[int, int]
    prefix: str

class PythonToken(Token):
    pass

class FStringNode:
    quote: str
    def is_in_expr(self) -> bool: ...

def tokenize_lines(
    lines: Iterable[str],
    *,
    pseudo_token: Pattern,
    triple_quoted: Set[str],
    endpats: Dict[str, Pattern],
) -> Iterator[PythonToken]:
    contstr = ''
    contstr_start: Tuple[int, int]
    endprog: Pattern
    prefix = ''
    additional_prefix = ''
    lnum = 0
    fstring_stack: List[FStringNode] = []

    for line in lines:
        lnum += 1
        pos = 0
        max_ = len(line)

        if contstr:
            endmatch = endprog.match(line)  # E:
            if endmatch:
                pos = endmatch.end(0)
                yield PythonToken(0, contstr + line[:pos], contstr_start, prefix)  # E:
                contstr = ''
            else:
                contstr = contstr + line
                continue

        while pos < max_:
            if fstring_stack:
                tos = fstring_stack[-1]
                if not tos.is_in_expr():
                    if pos == max_:
                        break

            if fstring_stack:
                string_line = line
                for fstring_stack_node in fstring_stack:
                    quote = fstring_stack_node.quote
                    end_match = endpats[quote].match(line, pos)
                    if end_match is not None:
                        end_match_string = end_match.group(0)
                        string_line = line[:pos] + end_match_string
                pseudomatch = pseudo_token.match(string_line, pos)
            else:
                pseudomatch = pseudo_token.match(line, pos)

            if pseudomatch:
                prefix = additional_prefix + pseudomatch.group(1)
                additional_prefix = ''
                start, pos = pseudomatch.span(2)
                spos = (lnum, start)
                token = pseudomatch.group(2)
            else:
                break

            if token in triple_quoted:
                endprog = endpats[token]
                endmatch = endprog.match(line, pos)
                if endmatch:
                    pos = endmatch.end(0)
                    yield PythonToken(0, token, spos, prefix)
                else:
                    contstr_start = spos
                    contstr = line[start:]
"#,
);
