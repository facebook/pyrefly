/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Write;

use crate::test::util::TestEnv;
use crate::test::util::testcase_for_macro;
use crate::testcase;

#[test]
fn annotated_recursive_signatures_do_not_recurse_through_bodies() -> anyhow::Result<()> {
    const N: usize = 64;

    let mut code = String::new();
    for i in 0..N {
        let (return_annotation, expected_error) = if i == 0 {
            (
                "int",
                "  # E: Function declared to return `int` but is missing an explicit `return`",
            )
        } else {
            ("None", "")
        };
        writeln!(
            &mut code,
            "def f{i}() -> {return_annotation}:{expected_error}\n    f{}()",
            (i + 1) % N,
        )
        .unwrap();
    }
    code.push_str("f0()\n");
    testcase_for_macro(
        TestEnv::new().with_recursion_depth_limit(32),
        &code,
        file!(),
        line!(),
    )
}

// Leaky loop tests: These demonstrate that loop recursion can create cycles
// in the definition of variables. The loop creates a cycle in `x`, and with
// iterative fixpoint solving we get deterministic answers regardless of which
// variable we force first or where we enter the cycle.

fn env_leaky_loop() -> TestEnv {
    TestEnv::one(
        "leaky_loop",
        r#"
x = None
def f(_: str | None) -> tuple[str, str]: ...
def g(_: int | None) -> tuple[int, int]: ...
while True:
    y, x = f(x)  # E: Argument `int | None` is not assignable to parameter `_` with type `str | None` in function `f`
    z, x = g(x)  # E: Argument `str` is not assignable to parameter `_` with type `int | None` in function `g`
"#,
    )
}

testcase!(
    try_leaky_loop_and_import_x,
    env_leaky_loop(),
    r#"
from typing import assert_type, Any
from leaky_loop import x
assert_type(x, int | None)
"#,
);

testcase!(
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

testcase!(
    try_leaky_loop_and_import_z,
    env_leaky_loop(),
    r#"
from typing import assert_type, Any
from leaky_loop import z
assert_type(z, int)
from leaky_loop import x
assert_type(x, int | None)
"#,
);

// Regression test against lambda parameters not being inferred in a way that reflects
// fixpoint semantics.
testcase!(
    class_field_lambda_param_scc_consistent_errors,
    r#"
from typing import Callable

class A:
    def __init__(self):
        self.a: Callable[[Callable[[int], int]], int] = lambda f: self.b
        self.b = self.a(lambda x: x + "foo")  # E: `+` is not supported between `int` and `Literal['foo']`
        self.c = self.z(lambda x: x + "foo")  # E: `+` is not supported between `int` and `Literal['foo']`
        self.z: Callable[[Callable[[int], int]], int] = lambda f: self.c
"#,
);

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
assert_type(x1, int)
"#,
);

testcase!(
    import_cycle_b,
    env_import_cycle(),
    r#"
from typing import assert_type, Any
from xx import x1
assert_type(x1, int)
from yy import y
assert_type(y, int)
from xx import x0
assert_type(x0, int)
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
assert_type(x1, int)
from xx import x0
assert_type(x0, int)
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
assert_type(x0, int)
from xx import x1
assert_type(x1, int)
"#,
);

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

// Decorator cycle tests: `fx` and `fy` each import and use the other as a
// decorator, creating a cross-module cycle through decorator application.
// With iterative fixpoint solving, these should converge deterministically.

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
@dec  # E: Argument `int` is not assignable to parameter `arg` with type `((...) -> int) -> (...) -> int` in function `dec`
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
@dec  # E: Argument `int` is not assignable to parameter `arg` with type `((...) -> int) -> (...) -> int` in function `dec`
@fx
def fy(arg: Callable[..., Any]) -> Callable[..., Any]: ...
"#,
    );
    env
}

testcase!(
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
    if n:  # E: Expected an indented block after `if` statement
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

// Regression test for stack overflow in generated code with many fields.
//
// Generated Python classes have a `__repr__` method with sequential
// `if self.fieldN is not None:` blocks that reassign `value` and call
// `L.append(...)`, creating a phi chain in the SSA binding graph.
//
// Each if-block includes a stmt_expr (L.append) as the last statement,
// which creates a termination_key for the branch. The phi's filter_map
// termination check calls get_idx(term_key) to resolve it, triggering
// the full expression chain within the block:
//   L.append('...%s' % (value)) → value → repr(self.fN) → phi_(N-1)
fn env_deep_phi_chain_term_expr() -> TestEnv {
    const NUM_FIELDS: usize = 1000;

    let mut code = String::new();
    code.push_str("class Struct:\n");

    code.push_str("  def __init__(self):\n");
    for i in 0..NUM_FIELDS {
        writeln!(code, "    self.f{i} = None").unwrap();
    }

    code.push_str("  def serialize(self):\n");
    code.push_str("    L: list[str] = []\n");
    for i in 0..NUM_FIELDS {
        writeln!(code, "    if self.f{i} is not None:").unwrap();
        writeln!(code, "          value = repr(self.f{i})").unwrap();
        writeln!(code, "          L.append('    f{i}=%s' % (value,))").unwrap();
    }
    code.push_str("    return value  # E: may be uninitialized\n");

    TestEnv::one("gen", &code)
}

testcase!(
    deep_phi_chain_term_expr,
    env_deep_phi_chain_term_expr(),
    r#"
from gen import Struct
x = Struct().serialize()
"#,
);

// Verify that a simple loop variable whose type is stable across iterations
// is correctly inferred. The LoopPhi cold-start bypass resolves the prior
// value (x = 0, type int) during iteration 1, and the warm-start iteration
// confirms convergence.
testcase!(
    iterative_loop_phi_simple,
    r#"
from typing import assert_type

def f(cond: bool):
    x = 0
    while cond:
        x = x + 1
    assert_type(x, int)
"#,
);

// Verify that multiple loop variables modified in the same loop are all
// correctly inferred. Each variable produces its own LoopPhi node in the
// SCC; the cold-start bypass must handle all of them independently, and
// the warm-start iteration must confirm convergence for every variable.
testcase!(
    iterative_loop_phi_multi_var,
    r#"
from typing import assert_type

def f(cond: bool):
    x = 0
    y = 1.5
    z: list[int] = []
    while cond:
        x = x + 1
        y = y * 2.0
        z = z + [x]
    assert_type(x, int)
    assert_type(y, float)
    assert_type(z, list[int])
"#,
);

// Verify that a loop variable whose type widens across iterations converges
// correctly under iterative fixpoint solving. The variable `x` starts as
// `int` but may be reassigned to `None` inside the loop, so the LoopPhi
// must converge to `int | None`.
testcase!(
    iterative_loop_phi_increment,
    r#"
from typing import assert_type

def f(cond: bool):
    x: int | None = 0
    while cond:
        assert_type(x, int | None)
        if cond:
            x = None
        else:
            x = 1
    assert_type(x, int | None)
"#,
);

// Verify that mutually recursive inferred returns stabilize under iterative
// fixpoint solving. Each function's return answer depends on the other's, so
// solving either answer enters the same SCC. The `Unknown` in the revealed
// fixpoint witnesses the placeholder used to break the recursive backedge.
testcase!(
    iterative_warm_start_mutual_recursion,
    r#"
from typing import reveal_type

def f(x: int):
    if x <= 0:
        return "done"
    return g(x - 1)

def g(x: int):
    return f(x)

reveal_type(f(1))  # E: revealed type: Literal['done'] | Unknown
reveal_type(g(1))  # E: revealed type: Literal['done'] | Unknown
"#,
);

// Diagnostics produced from cold-start placeholders must not leak after
// mutually recursive inferred returns stabilize. The inner reveal sees
// `Unknown` during cold start and `int | Unknown` during warm iterations, so
// its expectation only passes when the cold diagnostic is discarded.
testcase!(
    iterative_error_suppression_no_spurious_errors,
    r#"
from typing import reveal_type

def f(x: int):
    if x <= 0:
        return 0
    return reveal_type(g(x - 1)) + 1  # E: revealed type: int | Unknown

def g(x: int):
    return f(x)

reveal_type(f(1))  # E: revealed type: int | Unknown
reveal_type(g(1))  # E: revealed type: int | Unknown
"#,
);

// Real errors discovered after mutually recursive inferred returns stabilize
// must still be reported. Both answers contain `Literal['done'] | Unknown`,
// making the inner call to `f` an invalid argument to the outer call.
testcase!(
    iterative_error_suppression_real_errors_reported,
    r#"
from typing import reveal_type

def f(x: int):
    if x <= 0:
        return "done"
    return g(x - 1)

def g(x: int):
    return f(f(x))  # E: Argument `Literal['done'] | Unknown` is not assignable to parameter `x` with type `int` in function `f`

reveal_type(f(1))  # E: revealed type: Literal['done'] | Unknown
reveal_type(g(1))  # E: revealed type: Literal['done'] | Unknown
"#,
);

// Verify that mutually recursive inferred returns spanning modules form an
// SCC and converge through the cross-module `solve_idx_erased` path.

fn env_iterative_cross_module_cycle() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "a",
        r#"
from b import g

def f(x: int):
    if x <= 0:
        return 0
    return g(x - 1)
"#,
    );
    env.add(
        "b",
        r#"
from a import f

def g(x: int):
    return f(x)
"#,
    );
    env
}

testcase!(
    iterative_cross_module_scc_cycle,
    env_iterative_cross_module_cycle(),
    r#"
from typing import reveal_type
from a import f
from b import g
reveal_type(f(1))  # E: revealed type: Literal[0] | Unknown
reveal_type(g(1))  # E: revealed type: Literal[0] | Unknown
"#,
);

// Verify that two disjoint inferred-return SCCs solve independently. Each
// pair has its own concrete base case and no dependencies on the other pair.
testcase!(
    iterative_disjoint_scc_independence,
    r#"
from typing import reveal_type

# SCC 1: f and g call each other
def f(x: int):
    if x <= 0:
        return 0
    return g(x - 1)

def g(x: int):
    return f(x)

# SCC 2: h and k call each other (completely independent of f/g)
def h(x: str):
    if not x:
        return "done"
    return k(x[1:])

def k(x: str):
    return h(x)

reveal_type(f(1))  # E: revealed type: Literal[0] | Unknown
reveal_type(g(1))  # E: revealed type: Literal[0] | Unknown
reveal_type(h("a"))  # E: revealed type: Literal['done'] | Unknown
reveal_type(k("a"))  # E: revealed type: Literal['done'] | Unknown
"#,
);
