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
    bug = "over-binding 4 positionals to a 3-param target should be a 'Too many arguments' error at construction; pyrefly accepts it",
    test_partial_edge_construct_too_many_bound,
    r#"
from typing import reveal_type
import functools
def target(a: int, b: str, c: float) -> bytes: return b""
p = functools.partial(target, 1, "x", 2.0, 99)  # WANT: Too many arguments for "target"
reveal_type(p)  # E: revealed type: partial[bytes]
r = p()
reveal_type(r)  # E: revealed type: bytes
"#,
);

functools_testcase!(
    bug = "partial binds `a` both positionally (1) and by keyword (a=2); runtime raises 'multiple values for a', but pyrefly accepts it",
    test_partial_edge_construct_duplicate_bound_kw,
    r#"
from typing import reveal_type
import functools
def target(a: int, b: int) -> int: return 0
p = functools.partial(target, 1, a=2)  # WANT: "target" gets multiple values for keyword argument "a"
reveal_type(p)  # E: revealed type: partial[int]
"#,
);

functools_testcase!(
    bug = "partial does not type-check a bound keyword-only argument value against the wrapped callable's parameter type",
    test_partial_edge_construct_bound_kwonly_wrong_type,
    r#"
from typing import reveal_type
import functools
def kwonly(a: int, *, b: str) -> int: return 0
p = functools.partial(kwonly, 1, b=5)  # WANT: Argument "b" has incompatible type "int"; expected "str"
reveal_type(p)  # E: revealed type: partial[int]
"#,
);

functools_testcase!(
    bug = "calling a partial that leaves a required arg unfilled should error, but pyrefly accepts it",
    test_partial_edge_call_missing_remaining,
    r#"
from typing import reveal_type
import functools
def f(a: int, b: str) -> bytes: return b""
p = functools.partial(f, 1)
reveal_type(p)  # E: revealed type: partial[bytes]
p()  # WANT: Missing positional argument "b" in call
"#,
);

functools_testcase!(
    bug = "partial over a bound method drops remaining-arg type checking: p(2) passes int where b: str is expected, but pyrefly emits no error",
    test_partial_edge_target_bound_method_badcall,
    r#"
from typing import reveal_type
import functools
class C:
    def m(self, a: int, b: str) -> float: return 0.0
p = functools.partial(C().m, 1)
reveal_type(p)  # E: revealed type: partial[float]
p(2)  # WANT: Argument 1 to "m" has incompatible type "int"; expected "str"
"#,
);

functools_testcase!(
    bug = "partial over a target-typed lambda (Callable[[int, str], bytes]) loses the remaining parameter types: p(2) should be an arg-type error",
    test_partial_edge_target_typed_lambda_badcall,
    r#"
import functools
from typing import Callable, reveal_type
g: Callable[[int, str], bytes] = lambda a, b: b""
p = functools.partial(g, 1)
reveal_type(p)  # E: revealed type: partial[bytes]
p(2)  # WANT: Argument 1 has incompatible type "int"; expected "str"
"#,
);

functools_testcase!(
    bug = "partial drops the positional-only marker: p(b=2) binds to a positional-only param without error, though the direct call g(1, b=2) is correctly flagged",
    test_partial_edge_positional_only_marker,
    r#"
from typing import reveal_type
import functools
def g(a: int, b: int, /) -> bytes: return b""
p = functools.partial(g, 1)
reveal_type(p)  # E: revealed type: partial[bytes]
p(b=2)  # WANT: Unexpected keyword argument "b" (b is positional-only in g)
g(1, b=2)  # E: Expected argument `b` to be positional in function `g`
"#,
);

functools_testcase!(
    bug = "partial does not enforce the keyword-only marker: passing a keyword-only param positionally should be an error",
    test_partial_edge_keyword_only_marker_positional,
    r#"
import functools
from typing import assert_type
def k(a: int, *, b: str) -> bytes: return b""
p = functools.partial(k)
assert_type(p(1, b="x"), bytes)
p(1, "x")  # WANT: Too many positional arguments / takes 1 positional argument but 2 were given (b is keyword-only)
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
