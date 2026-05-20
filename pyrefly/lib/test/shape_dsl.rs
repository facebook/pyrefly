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
"#,
    );
    env.add_with_path(
        "my_lib",
        "my_lib.pyi",
        r#"
from typing import overload
from shape_extensions import uses_shape_dsl
from my_shapes import identity_ir

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
"#,
    );
    env
}

testcase!(
    test_uses_shape_dsl_preserves_type,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import plain_fn

assert_type(plain_fn(1), int)
"#,
);

testcase!(
    test_uses_shape_dsl_overload_with_implementation,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import overloaded_with_impl

assert_type(overloaded_with_impl(1), int)
assert_type(overloaded_with_impl("a"), str)
"#,
);

testcase!(
    test_uses_shape_dsl_overload_no_implementation,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import overloaded_no_impl

assert_type(overloaded_no_impl(1), int)
assert_type(overloaded_no_impl("a"), str)
"#,
);
