/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::shape_dsl::legacy_shaped_array_env;
use crate::test::shaped::shaped_env;
use crate::test::util::TestEnv;
use crate::test::util::testcase_for_macro;
use crate::testcase;

#[test]
fn test_shaped_comments_expression_depth_limit() {
    let payload = std::iter::repeat_n("1", 1200)
        .collect::<Vec<_>>()
        .join(" + ");
    let source = format!(
        "# enable-shape-type-comments\nn: int  # shaped: {payload}  # E: too deeply nested\n"
    );
    testcase_for_macro(TestEnv::new(), &source, file!(), line!())
        .expect("deep comment payloads should diagnose rather than overflow the stack");
}

testcase!(
    test_shaped_comments_scalar_type_argument,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from shape_extensions import IntVar

class Sized[N: IntVar]: ...

# shapes: M
def f(
    x: Sized[int],  # shaped: .args[0] = M
) -> Sized:  # shaped: M + 1
    ...

def check(x: Sized[3]) -> None:
    assert_type(f(x), Sized[4])
"#,
);

testcase!(
    test_shaped_comments_transpose,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray, dtype

# shapes: M, N
def transpose(
    n: int,  # shaped: M
    x: ndarray,  # shaped: [M, N]
) -> ndarray[tuple[int, int]]:  # shaped: .args[0] = [N, M]
    ...

def check(x: ndarray[[3, 4]]) -> None:
    assert_type(transpose(3, x), ndarray[[4, 3]])
    transpose(2, x)  # E: not assignable
"#,
);

testcase!(
    test_shaped_comments_variadic_and_nested_targets,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray, dtype

# shapes: *Batch, M, N
def batched(
    x: ndarray[tuple[int, ...], dtype[float]],  # shaped: .args[0] = [*Batch, M, N]
) -> ndarray[tuple[int, ...], dtype[float]]:  # shaped: .args[0] = [*Batch, N, M]
    ...

# shapes: M, N
def nested(
    x: list[ndarray[tuple[int, int], dtype[float]]],  # shaped: .args[0].args[0] = [M, N]
) -> ndarray:  # shaped: [N, M]
    ...

def check(x: ndarray[[2, 7, 3, 4], dtype[float]], y: list[ndarray[[3, 4], dtype[float]]]) -> None:
    assert_type(batched(x), ndarray[[2, 7, 4, 3], dtype[float]])
    assert_type(nested(y), ndarray[[4, 3]])
"#,
);

testcase!(
    test_shaped_comments_whole_tuple_and_arithmetic,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray
from shape_extensions import Int

# shapes: M, N
def from_shape(
    shape: tuple[int, int],  # shaped: type = [M, N]
    size: int,  # shaped: M + 1
) -> ndarray:  # shaped: [M, N]
    ...

def check() -> None:
    assert_type(from_shape((3, 4), 4), ndarray[[3, 4]])
    from_shape((3, 4), 5)  # E: not assignable
"#,
);

testcase!(
    test_shaped_comments_dsl_payloads,
    shaped_env(),
    r##"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray
from shape_extensions import IntTuple, broadcast, type_shape_dsl_function

@type_shape_dsl_function
def tagged(shape: IntTuple, tag: str) -> IntTuple:
    return shape

# shapes: M, N
def f(
    x: ndarray,  # shaped: [M, N]
) -> ndarray:  # shaped: tagged(broadcast(IntTuple[M, 1], IntTuple[1, N]), "#")
    ...

def check(x: ndarray[[3, 4]]) -> None:
    assert_type(f(x), ndarray[[3, 4]])
"##,
);
testcase!(
    test_shaped_comments_quoted_annotations,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray

# shapes: M, N
def f(
    x: "ndarray[tuple[int, int]]",  # shaped: .args[0] = [M, N]
) -> "ndarray":  # shaped: [N, M]
    ...

def check(x: ndarray[[3, 4]]) -> None:
    assert_type(f(x), ndarray[[4, 3]])
"#,
);

testcase!(
    test_shaped_comments_generic_replacement,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type, TypeVar
from arrays import ndarray, dtype
T = TypeVar("T")

# shapes: M, N
def f(
    x: ndarray,  # shaped: type = ndarray[[M, N], dtype[T]]
) -> ndarray:  # shaped: [N, M], dtype[T]
    ...

def check(x: ndarray[[3, 4], dtype[float]]) -> None:
    assert_type(f(x), ndarray[[4, 3], dtype[float]])
"#,
);

testcase!(
    test_shaped_comments_no_header,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray

# shapes: M, *Batch
def f(
    x: ndarray,  # shaped: this is not an expression
) -> ndarray:  # shaped: .args[999] = [Missing]
    assert_type(x, ndarray)
    return x
"#,
);
testcase!(
    test_shaped_comments_registered_array,
    legacy_shaped_array_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType = object]:
    shape: Shape

# shapes: M, N
def f(
    x: Array[tuple[int, int], float],  # shaped: .args[0] = [M, N]
) -> Array:  # shaped: [N, M], float
    ...

def check(x: Array[[3, 4], float]) -> None:
    assert_type(f(x), Array[[4, 3], float])
"#,
);
testcase!(
    test_shaped_comments_registered_array_dsl_and_bare_carrier,
    legacy_shaped_array_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from shape_extensions import IntTuple, shaped_array, broadcast

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType = object]: ...

# shapes: *Left, *Right
def combine(
    left: Array[IntTuple, float],  # shaped: .args[0] = Left
    right: Array[IntTuple, float],  # shaped: .args[0] = Right
) -> Array:  # shaped: broadcast(Left, Right), float
    ...

def check(left: Array[[3, 1], float], right: Array[[1, 4], float]) -> None:
    assert_type(combine(left, right), Array[[3, 4], float])
"#,
);
