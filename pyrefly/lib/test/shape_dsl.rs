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
from shape_extensions.dsl import shape_dsl_function

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

@shape_dsl_function  # E: @shape_dsl_function: unexpected statement in DSL body
def bad_syntax_ir(x: int) -> int:
    while x > 0:
        x = x - 1
    return x

@shape_dsl_function
def kwargs_ir(x: int, **kwargs) -> int:  # E: @shape_dsl_function: **kwargs parameters are not supported
    return x

@shape_dsl_function
def calls_undefined(x: int) -> int:  # E: @shape_dsl_function type error: undefined function: nonexistent
    return nonexistent(x)  # E: Could not find name `nonexistent`
"#,
    );
    env.add_with_path(
        "my_lib",
        "my_lib.pyi",
        r#"
from typing import overload
from shape_extensions import uses_shape_dsl
from my_shapes import identity_ir, double_ir, not_a_dsl_fn, bad_syntax_ir, kwargs_ir

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

// The `calls_undefined` function in my_shapes.pyi calls `nonexistent()`,
// which produces a type error diagnostic (tested by the `# E:` annotation
// on its definition). No separate test case is needed here — the error is
// validated by every test that uses `shape_dsl_env()`.
