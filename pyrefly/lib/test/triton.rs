/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn env() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "triton.runtime.jit",
        "triton/runtime/jit.pyi",
        r#"
from typing import Callable, overload

class _JitDecorator:
    def __call__[**P, R](self, fn: Callable[P, R]) -> Callable[P, R]: ...

@overload
def jit[**P, R](fn: Callable[P, R]) -> Callable[P, R]: ...
@overload
def jit(*, debug: bool = False) -> _JitDecorator: ...
"#,
    );
    env.add("triton", "from triton.runtime.jit import jit as jit");
    env.add("triton.language.core", "class constexpr: ...");
    env.add(
        "triton.language",
        "from triton.language.core import constexpr as constexpr",
    );
    env.add(
        "kernel_helpers",
        r#"
from triton import jit as compile_kernel
from triton.language import constexpr as constant
"#,
    );
    env
}

testcase!(
    test_triton_constexpr_parameter,
    env(),
    r#"
from typing import Any, assert_type
import triton
import triton.language as tl

@triton.jit
def kernel(block: tl.constexpr) -> int:
    assert_type(block, Any)
    return block + 1

assert_type(kernel(128), int)
"#,
);

testcase!(
    test_triton_constexpr_decorator_factory,
    env(),
    r#"
from typing import Any, assert_type
import triton
import triton.language as tl

@triton.jit(debug=True)
def kernel(flag: tl.constexpr = True) -> bool:
    assert_type(flag, Any)
    return flag

assert_type(kernel(), bool)
assert_type(kernel(False), bool)
"#,
);

testcase!(
    test_triton_constexpr_aliases,
    env(),
    r#"
from typing import Any, assert_type
from triton import jit as compile_kernel
from triton.language import constexpr as constant

@compile_kernel
def kernel(value: constant) -> str:
    assert_type(value, Any)
    return value

assert_type(kernel("mode"), str)
"#,
);

testcase!(
    test_triton_constexpr_definition_modules,
    env(),
    r#"
from typing import Any, assert_type
from triton.runtime.jit import jit
from triton.language.core import constexpr

@jit
def kernel(shape: constexpr) -> None:
    assert_type(shape, Any)

kernel((16, 32))
"#,
);

testcase!(
    test_triton_constexpr_parameter_kinds,
    env(),
    r#"
from typing import Any, assert_type
from triton import jit
from triton.language import constexpr

@jit
def kernel(block: "constexpr", /, *, mode: constexpr = "fast") -> None:
    assert_type(block, Any)
    assert_type(mode, Any)

kernel(128)
kernel(64, mode="slow")
"#,
);

testcase!(
    test_triton_constexpr_reexports,
    env(),
    r#"
from typing import Any, assert_type
from kernel_helpers import compile_kernel, constant

@compile_kernel(debug=True)
def kernel(value: constant) -> None:
    assert_type(value, Any)

kernel(128)
kernel()  # E: Missing argument `value`
"#,
);

testcase!(
    test_triton_constexpr_parameter_shadows_import,
    env(),
    r#"
from typing import Any, assert_type
from triton import jit
from triton.language import constexpr

@jit
def kernel(constexpr: int, block: constexpr) -> None:
    assert_type(constexpr, int)
    assert_type(block, Any)

kernel(1, 128)
"#,
);

testcase!(
    test_triton_constexpr_other_parameters_checked,
    env(),
    r#"
from triton import jit
from triton.language import constexpr

@jit
def kernel(block: constexpr, count: int) -> int:
    return count

kernel(128, "bad")  # E: not assignable to parameter `count`
"#,
);

testcase!(
    test_triton_constexpr_body_checked,
    env(),
    r#"
from triton import jit
from triton.language import constexpr

@jit
def kernel(block: constexpr) -> int:
    count: int = "bad"  # E: not assignable to `int`
    return count
"#,
);

testcase!(
    test_triton_constexpr_without_jit,
    env(),
    r#"
from typing import assert_type
from triton.language import constexpr

def ordinary(value: constexpr) -> None:
    assert_type(value, constexpr)

ordinary(128)  # E: not assignable to parameter `value`
"#,
);

testcase!(
    test_triton_constexpr_unrelated_decorator,
    env(),
    r#"
from typing import assert_type
from triton.language import constexpr

def jit[T](fn: T) -> T:
    return fn

@jit
def ordinary(value: constexpr) -> None:
    assert_type(value, constexpr)

ordinary(128)  # E: not assignable to parameter `value`
"#,
);

testcase!(
    test_triton_constexpr_unrelated_annotation,
    env(),
    r#"
from typing import assert_type
from triton import jit

class constexpr: ...

@jit
def kernel(value: constexpr) -> None:
    assert_type(value, constexpr)

kernel(128)  # E: not assignable to parameter `value`
"#,
);

testcase!(
    test_triton_constexpr_nested_function,
    env(),
    r#"
from typing import Any, assert_type
from triton import jit
from triton.language import constexpr

@jit
def kernel(value: constexpr) -> None:
    assert_type(value, Any)
    def ordinary(inner: constexpr) -> None:
        assert_type(inner, constexpr)
    ordinary(128)  # E: not assignable to parameter `inner`
"#,
);
