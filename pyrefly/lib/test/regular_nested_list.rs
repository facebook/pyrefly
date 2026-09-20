/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::shape_extensions_env;
use crate::testcase;

testcase!(
    regular_nested_list_literals,
    shape_extensions_env(),
    r#"
from typing import Any, Callable, assert_type
from shape_extensions import Elements, IntTuple, RegularNestedList

class Array[Shape: IntTuple]: ...

def consume[Shape: IntTuple](x: RegularNestedList[Shape, int]) -> Array[Shape]: ...
def factory[Shape: IntTuple](
    x: RegularNestedList[Shape, int] | list[object],
) -> Array[Shape]: ...
def precise_only[Shape: IntTuple](
    x: RegularNestedList[Shape, int] | RegularNestedList[Shape, str],
) -> Array[Shape]: ...
def array_or_data[Shape: IntTuple](
    x: Array[Shape] | RegularNestedList[Shape, int],
) -> Array[Shape]: ...
def array_scalar_or_data[Shape: IntTuple = []](
    x: Array[Shape] | int | RegularNestedList[Shape, int],
) -> Array[Shape]: ...
def consume_callbacks[Shape: IntTuple](
    x: RegularNestedList[Shape, Callable[[int], int]],
) -> Array[Shape]: ...
def marker_or_ordinary_callbacks[Shape: IntTuple](
    x: RegularNestedList[Shape, Callable[[int], int]]
    | list[Callable[[str], str]],
) -> None: ...
def generic_fallback[Shape: IntTuple, T](
    x: RegularNestedList[Shape, T] | list[object], value: T,
) -> T: ...
def fixed_shape_or_list(
    x: RegularNestedList[[2], list[int]] | list[object],
) -> None: ...
def require_two(x: RegularNestedList[[2], int]) -> None: ...
def from_objects[Shape: IntTuple](
    x: RegularNestedList[Shape, object] | list[object],
) -> Array[Shape]: ...
type Data[Shape: IntTuple] = RegularNestedList[Shape, int]
type DataOrObjects[Shape: IntTuple] = RegularNestedList[Shape, int] | list[object]
def aliased[Shape: IntTuple](x: Data[Shape]) -> Array[Shape]: ...
def aliased_factory[Shape: IntTuple](x: DataOrObjects[Shape]) -> Array[Shape]: ...

consume(1)  # E: is not assignable to parameter `x`
assert_type(consume([]), Array[[0]])
assert_type(consume([1, 2, 3]), Array[[3]])
assert_type(consume([[1, 2], [3, 4]]), Array[[2, 2]])
assert_type(consume([[]]), Array[[1, 0]])
mismatched_context: Array[[2, 2]] = consume([1, 2])  # E: is not assignable

assert_type(aliased([[1], [2]]), Array[[2, 1]])
assert_type(aliased_factory([1, 2]), Array[[2]])
assert_type(factory([1, 2]), Array[[2]])
assert_type(factory(["x"]), Array[IntTuple])
assert_type(factory([["x"], ["y", "z"]]), Array[IntTuple])

assert_type(array_scalar_or_data([1, 2]), Array[[2]])
assert_type(array_scalar_or_data(1), Array[[]])

def require_matrix(x: Array[[2, 2]]) -> None: ...
require_matrix(array_scalar_or_data(1))  # E: is not assignable to parameter `x`

assert_type(consume_callbacks([lambda value: value + 1]), Array[[1]])
consume_callbacks([lambda value: value + "bad"])  # E: is not assignable
# A rejected marker arm must not leak errors into a valid ordinary-list arm.
marker_or_ordinary_callbacks([lambda value: value + "good"])

# A rejected marker arm must not constrain `T` before ordinary list matching.
assert_type(generic_fallback([[1], [2, 3]], "x"), str)

defaulted: RegularNestedList = [1, 2]
assert_type(defaulted, RegularNestedList[IntTuple, bool | int | float | complex])
one: RegularNestedList[[2], int] = [1, 2]
empty: RegularNestedList[[0], int] = []
nested: list[RegularNestedList[[2], int]] = [[1, 2], [3, 4]]
strings: RegularNestedList[[2], str] = ["a", "b"]

def pair() -> RegularNestedList[[2], int]:
    return [1, 2]

def preserve[Shape: IntTuple](
    x: RegularNestedList[Shape, int],
) -> RegularNestedList[Shape, object]:
    return x

raw = [1, 2]
bad_raw: RegularNestedList[[2], int] = raw  # E: is not assignable
bad_star: RegularNestedList[[2], int] = [*raw]  # E: is not assignable
bad_leaf: RegularNestedList[[1], int] = ["x"]  # E: is not assignable
bad_shape: RegularNestedList[[3], int] = [1, 2]  # E: is not assignable
one.append(3)  # E: Object of class `RegularNestedList` has no attribute `append`

consume([[1], [2, 3]])  # E: is not assignable to parameter `x`
consume([1, [2]])  # E: is not assignable to parameter `x`
precise_only([["x"], ["y", "z"]])  # E: is not assignable to parameter `x`

def use_array(x: Array[[2]]) -> None:
    assert_type(array_or_data(x), Array[[2]])

def gradual_leaf(x: Any) -> None:
    assert_type(consume([x]), Array[[1, *Elements[IntTuple]]])
    assert_type(consume([[x]]), Array[[1, 1, *Elements[IntTuple]]])
    assert_type(consume([[x], [1]]), Array[[2, 1, *Elements[IntTuple]]])
    consume([[x], [1, 2], [3, 4, 5]])  # E: is not assignable
    require_two([x])  # E: is not assignable

def check_container(xs: list[int]) -> None:
    assert_type(from_objects([xs]), Array[IntTuple])

xs = []
fixed_shape_or_list([xs])
xs.append("x")

def soft_hint(x: RegularNestedList[[1], int]) -> None:
    x or [[1], [2, 3]]
"#,
);
