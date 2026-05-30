/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn shape_dsl_env() -> TestEnv {
    let path = std::env::var("SHAPE_DSL_TEST_PATH").expect("SHAPE_DSL_TEST_PATH must be set");
    let mut env = TestEnv::new_with_site_package_paths(&[&path]);
    env.add_with_path(
        "my_shapes",
        "my_shapes.pyi",
        r#"
from typing import Any
from shape_extensions.dsl import shape_dsl_function

class symint: ...
Unknown: Any = ...

@shape_dsl_function
def identity_ir(x: int) -> int:
    return x

@shape_dsl_function
def times_two(x: int) -> int:
    return x + x

@shape_dsl_function
def double_ir(x: int) -> int:
    return times_two(x)

def not_a_dsl_fn(x: int) -> int: ...

@shape_dsl_function
def bad_syntax_ir(x: int) -> int:
    while x > 0:  # E: @shape_dsl_function: unexpected statement in DSL body
        x = x - 1
    return x

@shape_dsl_function
def kwargs_ir(x: int, **kwargs) -> int:  # E: @shape_dsl_function: **kwargs parameters are not supported
    return x

@shape_dsl_function
def calls_undefined(x: int) -> int:  # E: @shape_dsl_function type error: undefined function: nonexistent
    return nonexistent(x)  # E: Could not find name `nonexistent`

@shape_dsl_function
def bad_no_ret(x: int):  # E: @shape_dsl_function type error: DSL function bad_no_ret must have a return type
    return x

@shape_dsl_function
def returns_wrong_type_ir(x: int) -> bool:  # E: @shape_dsl_function type error: return expression type int is not compatible with declared return type bool
    return x  # E: Returned type `int` is not assignable to declared return type `bool`

@shape_dsl_function
def dims_as_scalar_union_ir(x: list[int | symint]) -> int | symint:
    return [d for d in x]  # E: Returned type `list[int | symint]` is not assignable to declared return type `int | symint`

@shape_dsl_function
def unknown_fallback_ir(x: int) -> int:
    return Unknown

@shape_dsl_function
def helper_exact_one_ir(x: int) -> int:
    return x

@shape_dsl_function
def too_few_args_ir() -> int:  # E: @shape_dsl_function type error: 'helper_exact_one_ir' takes exactly 1 argument(s), got 0
    return helper_exact_one_ir()

@shape_dsl_function
def too_many_args_ir(x: int) -> int:  # E: @shape_dsl_function type error: 'helper_exact_one_ir' takes at most 1 argument(s), got 2
    return helper_exact_one_ir(x, x)

@shape_dsl_function
def two_errors_ir(x: int) -> int:  # E: @shape_dsl_function type error: undefined function: missing_one  # E: @shape_dsl_function type error: undefined function: missing_two
    return missing_one(x) + missing_two(x)  # E: Could not find name `missing_one`  # E: Could not find name `missing_two`
"#,
    );
    env.add_with_path(
        "my_lib",
        "my_lib.pyi",
        r#"
from typing import Any, overload
from shape_extensions import uses_shape_dsl
from my_shapes import identity_ir, double_ir, not_a_dsl_fn, bad_syntax_ir, kwargs_ir, calls_undefined, bad_no_ret, two_errors_ir, returns_wrong_type_ir, dims_as_scalar_union_ir, unknown_fallback_ir, helper_exact_one_ir, too_few_args_ir, too_many_args_ir
import my_shapes

non_literal: Any

@uses_shape_dsl(identity_ir)
def plain_fn(x: int) -> int: ...

@overload
def overloaded_with_impl(x: int) -> int: ...
@overload
def overloaded_with_impl(x: str) -> str: ...
@uses_shape_dsl(identity_ir)
def overloaded_with_impl(x: int | str) -> int | str: ...

@uses_shape_dsl(identity_ir)
@overload
def overloaded_no_impl(x: int) -> int: ...
@overload
def overloaded_no_impl(x: str) -> str: ...

@uses_shape_dsl(double_ir)
def double_fn(x: int) -> int: ...

@uses_shape_dsl(not_a_dsl_fn)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def bad_fn(x: int) -> int: ...

@uses_shape_dsl(bad_syntax_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def bad_syntax_fn(x: int) -> int: ...

@uses_shape_dsl(kwargs_ir)
def kwargs_fn(x: int) -> int: ...

@uses_shape_dsl(calls_undefined)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def calls_undefined_fn(x: int) -> int: ...

@uses_shape_dsl(bad_no_ret)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def no_ret_fn(x: int) -> int: ...

@uses_shape_dsl(two_errors_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def two_errors_fn(x: int) -> int: ...

@uses_shape_dsl(returns_wrong_type_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def returns_wrong_type_fn(x: int) -> bool: ...

@uses_shape_dsl(dims_as_scalar_union_ir)
def dims_as_scalar_union_fn(x: tuple[int, int]) -> tuple[int, int]: ...

@uses_shape_dsl(unknown_fallback_ir)
def unknown_fallback_fn(x: int) -> int: ...

@uses_shape_dsl(helper_exact_one_ir)
def helper_exact_one_fn(x: int) -> int: ...

@uses_shape_dsl(too_few_args_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def too_few_args_fn() -> int: ...

@uses_shape_dsl(too_many_args_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def too_many_args_fn(x: int) -> int: ...

class BadCaptureInit:
    @uses_shape_dsl(identity_ir, capture_init=["x", non_literal])  # E: `capture_init` entries must be string literals
    def forward(self, x: int) -> int: ...

@uses_shape_dsl(my_shapes.identity_ir)
def dotted_fn(x: int) -> int: ...

"#,
    );
    env
}

testcase!(
    test_uses_shape_dsl_preserves_type,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import plain_fn

# identity_ir returns its input unchanged. Because val_to_type synthesizes
# Literal[n] from the DSL's traced integer value (not the declared return
# type), the result is Literal[1], not int.
assert_type(plain_fn(1), Literal[1])
"#,
);

testcase!(
    test_uses_shape_dsl_overload_with_implementation,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import overloaded_with_impl

assert_type(overloaded_with_impl(1), Literal[1])
assert_type(overloaded_with_impl("a"), str)
"#,
);

testcase!(
    test_uses_shape_dsl_overload_no_implementation,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import overloaded_no_impl

assert_type(overloaded_no_impl(1), Literal[1])
assert_type(overloaded_no_impl("a"), str)
"#,
);

testcase!(
    test_uses_shape_dsl_cross_function_call,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import double_fn

assert_type(double_fn(3), Literal[6])
"#,
);

testcase!(
    test_uses_shape_dsl_not_a_dsl_function,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import bad_fn

# The @uses_shape_dsl argument is not a @shape_dsl_function, so no shape
# transform is applied and the declared return type (int) is used instead.
assert_type(bad_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_unsupported_syntax,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import bad_syntax_fn

# bad_syntax_ir uses a while loop which is unsupported DSL syntax, so
# bad_syntax_fn falls back to the declared return type.
assert_type(bad_syntax_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_kwargs_warning,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import kwargs_fn

# kwargs_ir has **kwargs which triggers a warning but the DSL conversion
# still succeeds (kwargs are silently dropped), so shape inference works.
assert_type(kwargs_fn(1), Literal[1])
"#,
);

testcase!(
    test_shape_dsl_uses_failing_function,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import calls_undefined_fn

# calls_undefined is rejected because its body calls an undefined helper. The
# consumer also gets rejected as a DSL use-site and falls back to its declared
# return type.
assert_type(calls_undefined_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_function_requires_return_annotation,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import no_ret_fn

# bad_no_ret is not accepted as a DSL function without a return annotation, so
# no_ret_fn falls back to its declared return type.
assert_type(no_ret_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_reports_multiple_errors,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import two_errors_fn

# two_errors_ir reports both undefined helper names from the same DSL body, and
# the consumer falls back to the declared return type.
assert_type(two_errors_fn(1), int)
"#,
);

testcase!(
    bug =
        "dotted-name arguments to @uses_shape_dsl currently silent-noop; should emit a diagnostic",
    test_shape_dsl_dotted_name_silent_noop,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import dotted_fn

# Dotted-name arguments are currently ignored without a diagnostic, so no shape
# transform is applied and the declared return type is used.
assert_type(dotted_fn(1), int)
"#,
);

// ── Recursion-safety tests ────────────────────────────────────────────────────

fn shape_dsl_recursion_env() -> TestEnv {
    let path = std::env::var("SHAPE_DSL_TEST_PATH").expect("SHAPE_DSL_TEST_PATH must be set");
    let mut env = TestEnv::new_with_site_package_paths(&[&path]);
    env.add_with_path(
        "recursive_shapes",
        "recursive_shapes.pyi",
        r#"
from shape_extensions.dsl import shape_dsl_function

# Direct self-recursion: should be rejected with a cycle diagnostic.
@shape_dsl_function
def self_recursive_ir(x: int) -> int:  # E: @shape_dsl_function type error: DSL function 'self_recursive_ir' is recursive
    return self_recursive_ir(x)

# Mutual recursion A → B → A: both should be rejected individually.
@shape_dsl_function
def mutual_a_ir(x: int) -> int:  # E: @shape_dsl_function type error: DSL function 'mutual_a_ir' is recursive
    return mutual_b_ir(x)

@shape_dsl_function
def mutual_b_ir(x: int) -> int:  # E: @shape_dsl_function type error: DSL function 'mutual_b_ir' is recursive
    return mutual_a_ir(x)

# Non-recursive depth-3 chain: triple_ir → triple_mid → triple_leaf.
# For input n, triple_leaf(n) = n+n+n = 3n, so triple_ir(4) = 12.
@shape_dsl_function
def triple_leaf(x: int) -> int:
    return x + x + x

@shape_dsl_function
def triple_mid(x: int) -> int:
    return triple_leaf(x)

@shape_dsl_function
def triple_ir(x: int) -> int:
    return triple_mid(x)
"#,
    );
    env.add_with_path(
        "recursive_lib",
        "recursive_lib.pyi",
        r#"
from shape_extensions import uses_shape_dsl
from recursive_shapes import self_recursive_ir, mutual_a_ir, triple_ir

@uses_shape_dsl(self_recursive_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def self_recursive_fn(x: int) -> int: ...

@uses_shape_dsl(mutual_a_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def mutual_fn(x: int) -> int: ...

@uses_shape_dsl(triple_ir)
def triple_fn(x: int) -> int: ...
"#,
    );
    env
}

testcase!(
    test_shape_dsl_self_recursive_rejected,
    shape_dsl_recursion_env(),
    r#"
from typing import assert_type
from recursive_lib import self_recursive_fn

# self_recursive_ir is rejected as recursive, so self_recursive_fn falls
# back to its declared return type rather than crashing the evaluator.
assert_type(self_recursive_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_mutual_recursive_rejected,
    shape_dsl_recursion_env(),
    r#"
from typing import assert_type
from recursive_lib import mutual_fn

# mutual_a_ir / mutual_b_ir form a cycle; mutual_fn falls back to int.
assert_type(mutual_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_non_recursive_chain,
    shape_dsl_recursion_env(),
    r#"
from typing import Literal, assert_type
from recursive_lib import triple_fn

# triple_ir → triple_mid → triple_leaf is a valid depth-3 chain with no
# cycles.  triple_leaf(x) = x+x+x, so triple_fn(4) evaluates to Literal[12].
assert_type(triple_fn(4), Literal[12])
"#,
);

testcase!(
    test_shape_dsl_wrong_return_type,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import returns_wrong_type_fn

# returns_wrong_type_ir is declared `-> bool` but its body returns an `int`
# expression, so it fails the compile-time return-type check and
# returns_wrong_type_fn falls back to its declared bool return type.
assert_type(returns_wrong_type_fn(1), bool)
"#,
);

testcase!(
    test_shape_dsl_list_return_for_scalar_union,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import dims_as_scalar_union_fn

# Tensor.size() uses this shape: the DSL annotation is the scalar dimension
# type `int | symint`, but returning a list of dimensions means "produce a
# concrete tuple of dimensions".
assert_type(dims_as_scalar_union_fn((1, 2)), tuple[Literal[1], Literal[2]])
"#,
);

testcase!(
    test_shape_dsl_unknown_return_fallback,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import unknown_fallback_fn

# Unknown is the DSL's explicit fixture fallback sentinel. It should not make
# the DSL function invalid just because it evaluates to Val::None internally.
assert_type(unknown_fallback_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_arg_count_too_few,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import too_few_args_fn

# too_few_args_ir calls helper_exact_one_ir() with 0 args but it needs 1,
# so the DSL compile-time check fires and the consumer falls back to int.
assert_type(too_few_args_fn(), int)
"#,
);

testcase!(
    test_shape_dsl_arg_count_too_many,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import too_many_args_fn

# too_many_args_ir calls helper_exact_one_ir(x, x) with 2 args but it takes 1,
# so the DSL compile-time check fires and the consumer falls back to int.
assert_type(too_many_args_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_capture_init_requires_string_literals,
    shape_dsl_env(),
    r#"
from my_lib import BadCaptureInit

# capture_init is read during class binding. Non-literal entries are rejected
# instead of silently dropping them from the captured __init__ field list.
BadCaptureInit()
"#,
);
