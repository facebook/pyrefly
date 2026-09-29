/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn index_shape_env() -> TestEnv {
    let mut env = TestEnv::one_with_path(
        "shape_extensions",
        "shape_extensions/__init__.pyi",
        r#"
from typing import Any

class Int[T]: ...
class IntTuple: ...
class IntVar: ...
class Index: ...
def index_shape(shape: IntTuple, index: Any) -> IntTuple: ...
"#,
    );
    env.add(
        "shape_helpers",
        "from shape_extensions import Index, index_shape as apply_index\n",
    );
    env
}

testcase!(
    test_index_shape_on_ordinary_generic_class,
    index_shape_env(),
    r#"
from typing import Any, assert_type, overload
from shape_extensions import Index, Int, IntTuple, IntVar, index_shape
from shape_helpers import Index as ReexportedIndex, apply_index

class Array[Shape: IntTuple]:
    shape: Shape

    @overload
    def __getitem__[I: Index](self, index: I) -> Array[index_shape(Shape, I)]: ...
    @overload
    def __getitem__(self, index: str) -> Array[IntTuple]: ...
    def __getitem__(self, index: object) -> Array[Any]: ...

def aliased[Shape: IntTuple, I: ReexportedIndex](
    value: Array[Shape], index: I
) -> Array[apply_index(Shape, I)]: ...

def symbolic_negative_start[N: IntVar](
    value: Array[IntTuple[10]], n: Int[N]
) -> None:
    assert_type(value[-(n + 1):], Array[IntTuple[N + 1]])
    assert_type(value[-(-n):], Array[IntTuple[10 - N]])
    assert_type(value[-(-(-(n + 1))):], Array[IntTuple[N + 1]])
    assert_type(value[:n + 1], Array[IntTuple[N + 1]])
    assert_type(value[::n + 1], Array[IntTuple[(10 + N) // (N + 1)]])

def check(value: Array[IntTuple[10, 20, 30]]) -> None:
    assert_type(value[0], Array[IntTuple[20, 30]])
    assert_type(value[1:5:2], Array[IntTuple[2, 20, 30]])
    assert_type(value[:, None, -1], Array[IntTuple[10, 1, 30]])
    assert_type(value[:, (0, 2)], Array[IntTuple[10, 2, 30]])
    assert_type(value[..., 0], Array[IntTuple[10, 20]])
    assert_type(value["fallback"], Array[IntTuple])
    assert_type(aliased(value, 0), Array[IntTuple[20, 30]])

def gradual(
    value: Array[IntTuple[10, 20]],
    dynamic: Any,
    indices: list[int],
) -> None:
    assert_type(value[dynamic], Array[IntTuple])
    assert_type(value[indices], Array[IntTuple[int, 20]])
"#,
);

testcase!(
    test_index_shape_diagnostics_and_call_validation,
    index_shape_env(),
    r#"
from shape_extensions import Index, IntTuple, index_shape

class Array[Shape: IntTuple]:
    def __getitem__[I: Index](self, index: I) -> Array[index_shape(Shape, I)]: ...

def check(value: Array[IntTuple[2, 3]], scalar: Array[IntTuple[()]]) -> None:
    value[0, 0, 0]  # E: Too many indices for tensor: got 3, expected at most 2
    value[..., ...]  # E: an index may contain at most one ellipsis
    scalar[0]  # E: Cannot index scalar tensor (rank 0)

def bad_shape[I: Index](index: I) -> Array[index_shape(int, I)]: ...  # E: Expected an `IntTuple` first argument to `index_shape`, got `int`
def bad_index[S: IntTuple](shape: S) -> Array[index_shape(S, str)]: ...  # E: Expected an `Index` second argument to `index_shape`, got `str`
def bad_both() -> Array[index_shape(int, str)]: ...  # E: Expected an `IntTuple` first argument to `index_shape`, got `int`  # E: Expected an `Index` second argument to `index_shape`, got `str`
def bad_arity[S: IntTuple](shape: S) -> Array[index_shape(S)]: ...  # E: Expected 2 arguments for `index_shape`, got 1
def bad_starred[Args]() -> Array[index_shape(*Args)]: ...  # E: `index_shape` does not accept starred arguments
def bad_keyword[S: IntTuple, I: Index](shape: S, index: I) -> Array[index_shape(shape=S, index=I)]: ...  # E: `index_shape` does not accept keyword arguments
"#,
);

testcase!(
    test_shape_index_speculation_preserves_ordinary_slice_inference,
    index_shape_env(),
    r#"
from typing import TypeVar, assert_type

T = TypeVar("T", bound=int)

def check(values: list[str], start: T) -> None:
    assert_type(values[start + 1:], list[str])
"#,
);
