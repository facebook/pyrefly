/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::shape_extensions_env;
use crate::testcase;

testcase!(
    array_coercible_literals,
    shape_extensions_env(),
    r#"
from collections.abc import Sequence
from typing import Any, Callable, assert_type
from shape_extensions import ArrayCoercible, Elements, IntTuple, Scalar

class Array[Shape: IntTuple]: ...

class IntList(list[int]): ...
class IntTupleSubclass(tuple[int, ...]): ...
class TextSubclass(str): ...
class BytesSubclass(bytes): ...

def consume[Shape: IntTuple](x: ArrayCoercible[Shape, int]) -> Array[Shape]: ...
def consume_objects[Shape: IntTuple](
    x: ArrayCoercible[Shape, object],
) -> Array[Shape]: ...
def consume_any[Shape: IntTuple](
    x: ArrayCoercible[Shape, Any],
) -> Array[Shape]: ...
def factory[Shape: IntTuple](
    x: ArrayCoercible[Shape, int] | list[object],
) -> Array[Shape]: ...
def precise_only[Shape: IntTuple](
    x: ArrayCoercible[Shape, int] | ArrayCoercible[Shape, str],
) -> Array[Shape]: ...
def array_or_data[Shape: IntTuple](
    x: Array[Shape] | ArrayCoercible[Shape, int],
) -> Array[Shape]: ...
def array_scalar_or_data[Shape: IntTuple](
    x: Array[Shape] | Scalar[Shape, int] | ArrayCoercible[Shape, int],
) -> Array[Shape]: ...
type Data[Shape: IntTuple] = ArrayCoercible[Shape, int]
type DataOrObjects[Shape: IntTuple] = ArrayCoercible[Shape, int] | list[object]
def aliased[Shape: IntTuple](x: Data[Shape]) -> Array[Shape]: ...
def aliased_factory[Shape: IntTuple](x: DataOrObjects[Shape]) -> Array[Shape]: ...
def require_one(x: ArrayCoercible[[1], int]) -> Array[[1]]: ...
def consume_callbacks[Shape: IntTuple](
    x: ArrayCoercible[Shape, Callable[[int], int]],
) -> Array[Shape]: ...
def consume_str_callbacks[Shape: IntTuple](
    x: ArrayCoercible[Shape, Callable[[str], str]],
) -> Array[Shape]: ...
def pick_int_or_str[Shape: IntTuple](
    x: ArrayCoercible[Shape, int] | ArrayCoercible[Shape, str],
) -> Array[Shape]: ...
def pick_int_or_str_callback[Shape: IntTuple](
    x: ArrayCoercible[Shape, Callable[[int], int]]
    | ArrayCoercible[Shape, Callable[[str], str]],
) -> Array[Shape]: ...
def pick_str_or_int_callback[Shape: IntTuple](
    x: ArrayCoercible[Shape, Callable[[str], str]]
    | ArrayCoercible[Shape, Callable[[int], int]],
) -> Array[Shape]: ...
def pick_callback_or_str[Shape: IntTuple](
    x: ArrayCoercible[Shape, Callable[[int], int]] | ArrayCoercible[Shape, str],
) -> Array[Shape]: ...
def marker_or_ordinary_callbacks[Shape: IntTuple](
    x: ArrayCoercible[Shape, Callable[[int], int]]
    | list[Callable[[str], str]],
) -> None: ...
def concrete_marker_or_list(x: ArrayCoercible[[3], int] | list[int]) -> None: ...
def ident[T](x: T) -> T: ...
def generic_domain[Shape: IntTuple, T](
    x: ArrayCoercible[Shape, T],
) -> tuple[Array[Shape], T]: ...

def scalar_family_is_a_leaf[Phantom: IntTuple](x: Scalar[Phantom, int]) -> None:
    assert_type(consume(x), Array[[]])
    assert_type(consume([x]), Array[[1]])
    require_one(x)  # E: is not assignable
    assert_type(require_one([x]), Array[[1]])

def concrete_scalar_family_is_a_leaf(x: Scalar[[], int]) -> None:
    assert_type(consume(x), Array[[]])
    assert_type(consume([x]), Array[[1]])

assert_type(consume(1), Array[[]])
assert_type(consume([]), Array[[0]])
assert_type(consume([1, 2, 3]), Array[[3]])
assert_type(consume([[1, 2], [3, 4]]), Array[[2, 2]])
assert_type(consume([[]]), Array[[1, 0]])
assert_type(aliased([[1], [2]]), Array[[2, 1]])
assert_type(aliased_factory([1, 2]), Array[[2]])
assert_type(array_scalar_or_data([1, 2]), Array[[2]])
assert_type(consume_callbacks([lambda value: value + 1]), Array[[1]])
consume_callbacks([lambda value: value + "bad"])  # E: is not assignable
assert_type(pick_int_or_str(["a", "b"]), Array[[2]])
assert_type(pick_int_or_str([1, 2]), Array[[2]])
assert_type(
    pick_int_or_str_callback([lambda value: value + "bad"]), Array[[1]]
)
assert_type(
    pick_str_or_int_callback([lambda value: value + "bad"]), Array[[1]]
)
assert_type(
    # No clean alternative exists, so ordinary list inference rejects the argument.
    pick_callback_or_str([lambda value: value + "bad"]),  # E: is not assignable
    Array[IntTuple],
)
# An errorful marker projection must not hide a valid ordinary union alternative.
marker_or_ordinary_callbacks([lambda value: value + "good"])
# A concrete marker with the wrong shape does not hide a compatible ordinary arm.
concrete_marker_or_list([1, 2])
# Lambda parameters follow the domain hint: the same body that errors under
# `Callable[[int], int]` above is clean under `Callable[[str], str]`.
assert_type(consume_str_callbacks([lambda value: value + "good"]), Array[[1]])
assert_type(consume_str_callbacks([lambda value: value + "bad"]), Array[[1]])
generic_result = generic_domain([1, missing_generic])  # E: Could not find name `missing_generic`
assert_type(generic_result, tuple[Array[[2]], int])

defaulted: ArrayCoercible = 1
assert_type(defaulted, ArrayCoercible[[], bool | int | float | complex])
typed_scalar: int = 1
typed_scalar_view: ArrayCoercible[[], int] = typed_scalar

one: ArrayCoercible[[2], int] = [1, 2]
empty: ArrayCoercible[[0], int] = []
nested: list[ArrayCoercible[[2], int]] = [[1, 2], [3, 4]]
strings: ArrayCoercible[[2], str] = ["a", "b"]
wide: ArrayCoercible[IntTuple, object] = one

def accepts_exact_strings(text: str, blob: bytes) -> None:
    text_scalar: ArrayCoercible[[], str] = text
    bytes_scalar: ArrayCoercible[[], bytes] = blob
    text_list: ArrayCoercible[[2], str] = [text, text]
    bytes_list: ArrayCoercible[[2], bytes] = [blob, blob]

def pair() -> ArrayCoercible[[2], int]:
    return [1, 2]

def preserve[Shape: IntTuple](
    x: ArrayCoercible[Shape, int],
) -> ArrayCoercible[Shape, object]:
    return x

raw = [1, 2]
raw_tuple = (1, 2)
bad_raw: ArrayCoercible[[2], int] = raw  # E: is not assignable
bad_tuple: ArrayCoercible[[2], int] = (1, 2)  # E: is not assignable
bad_star: ArrayCoercible[[2], int] = [*raw]  # E: is not assignable
bad_leaf: ArrayCoercible[[1], int] = ["x"]  # E: is not assignable
bad_shape: ArrayCoercible[[3], int] = [1, 2]  # E: is not assignable
bad_scalar_shape: ArrayCoercible[[1], int] = 1  # E: is not assignable
one.append(3)  # E: Object of class `ArrayCoercible` has no attribute `append`

consume([[1], [2, 3]])  # E: is not assignable to parameter `x`
consume([1, [2]])  # E: is not assignable to parameter `x`
assert_type(factory([["x"], ["y", "z"]]), Array[IntTuple])
precise_only([["x"], ["y", "z"]])  # E: is not assignable to parameter `x`
consume([[missing], [2, 3]])  # E: Could not find name `missing` # E: is not assignable to parameter `x`

assert_type(factory([1, 2]), Array[[2]])
assert_type(factory(["x"]), Array[IntTuple])

def use_array(x: Array[[2]]) -> None:
    assert_type(array_or_data(x), Array[[2]])

def rejects_array_leaf(x: Array[[2]]) -> None:
    consume([x])  # E: is not assignable
    consume_objects([x])  # E: is not assignable
    consume_objects(x)  # E: is not assignable

def rejects_deferred_containers(
    xs: IntList,
    ts: IntTupleSubclass,
    maybe_xs: int | list[int],
    coercible: ArrayCoercible[[2], int],
    sequence: Sequence[int],
) -> None:
    consume_objects(xs)  # E: is not assignable
    consume_objects([xs])  # E: is not assignable
    consume_objects(ts)  # E: is not assignable
    consume_objects([ts])  # E: is not assignable
    consume_objects([raw])  # E: is not assignable
    consume_objects([raw_tuple])  # E: is not assignable
    consume_objects([maybe_xs])  # E: is not assignable
    consume_any(xs)  # E: is not assignable
    consume_any([xs])  # E: is not assignable
    consume_any(ts)  # E: is not assignable
    consume_any([ts])  # E: is not assignable
    consume_objects([coercible])  # E: is not assignable
    consume_objects(sequence)  # E: is not assignable
    consume_objects([sequence])  # E: is not assignable

def rejects_bounded_container[T: list[int]](value: T) -> None:
    consume_objects(value)  # E: is not assignable
    consume_objects([value])  # E: is not assignable

def string_subclasses_are_leaves(text: TextSubclass, blob: BytesSubclass) -> None:
    assert_type(consume_objects(text), Array[[]])
    assert_type(consume_objects([text]), Array[[1]])
    assert_type(consume_objects(blob), Array[[]])
    assert_type(consume_objects([blob]), Array[[1]])

def rejects_unrestricted_type_var[T](value: T) -> None:
    consume_objects(value)
    consume_objects([value])  # E: is not assignable

def unresolved_generic_is_not_a_scalar_leaf[T](value: T) -> None:
    # A generic return value delegates to ordinary typing instead of pinning a
    # scalar shape, so a literal containing one is rejected exactly like the
    # unrestricted type variable above. Solved variables expand first and
    # unresolved solver variables take the same deferred path.
    consume_objects([ident(value)])  # E: is not assignable

def any_leaf(x: Any, y: Any) -> None:
    assert_type(consume([1, x]), Array[[2]])
    assert_type(consume([[], x]), Array[[2, 0]])
    assert_type(consume([x]), Array[[1, *Elements[IntTuple]]])
    assert_type(consume([x, y]), Array[[2, *Elements[IntTuple]]])
    assert_type(consume([[x]]), Array[[1, 1, *Elements[IntTuple]]])
    assert_type(consume([[x], [y]]), Array[[2, 1, *Elements[IntTuple]]])
    assert_type(consume([[x], [1]]), Array[[2, 1]])
    consume([[x], [[y]]])  # E: is not assignable
    consume([[x], [[y]], [1]])  # E: is not assignable
    consume([[1], [x], [[y]]])  # E: is not assignable

def any_argument(x: Any) -> None:
    assert_type(consume(x), Array[Any])

def soft_hint(x: ArrayCoercible[[1], int]) -> None:
    x or [[1], [2, 3]]
"#,
);
