/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::shape_extensions_env;
use crate::testcase;

testcase!(
    scalar_normalization_and_inference,
    shape_extensions_env(),
    r#"
from typing import Any, Never, assert_type
from shape_extensions import Elements, IntTuple, Scalar

class Array[Shape: IntTuple]: ...

type ArrayLike[Shape: IntTuple] = Array[Shape] | Scalar[Shape]

def array_like[Shape: IntTuple](x: ArrayLike[Shape]) -> Array[Shape]: ...
def suspended[Shape: IntTuple](x: Scalar[Shape, int]) -> Scalar[Shape, int]:
    y: int = x
    # Reading the scalar through its upper bound does not constrain Shape.
    assert_type(x, Scalar[Shape, int])
    assert_type(x.bit_length(), int)
    return x

def preserve_family_in_union[Shape: IntTuple](
    x: Scalar[Shape, int],
) -> Array[Shape] | Scalar[Shape, int]:
    return x

def cannot_manufacture[Shape: IntTuple](x: int) -> Scalar[Shape, int]:
    return x  # E: is not assignable to declared return type

def consume[Shape: IntTuple](x: Scalar[Shape, int]) -> Array[Shape]: ...
def consume_unpacked[Shape: IntTuple](
    x: Scalar[IntTuple[*Elements[Shape]], int],
) -> Array[Shape]: ...
def scalar_identity[Shape: IntTuple](x: Scalar[Shape, int]) -> Scalar[Shape, int]: ...
def identity_helper[T](x: T) -> T: ...

def normalize_through_solver_var(x: Scalar[[2], int]) -> None:
    assert_type(identity_helper(x), Never)

def any_through_solver_var(x: Any) -> None:
    consume_never(identity_helper(x))

def propagate[Actual: IntTuple](x: Scalar[Actual, int]) -> Array[Actual]:
    return consume(x)

def relay[Actual: IntTuple](x: Scalar[Actual, int]) -> Scalar[Actual, int]:
    return scalar_identity(x)

def relay_through_inference[Actual: IntTuple](
    x: Scalar[Actual, int],
) -> Scalar[Actual, int]:
    return scalar_identity(identity_helper(x))

assert_type(array_like(1), Array[[]])
assert_type(consume_unpacked(1), Array[[]])

def use_array(array: Array[[2, 3]]) -> None:
    assert_type(array_like(array), Array[[2, 3]])

def use_scalar_union(value: int | float) -> None:
    assert_type(array_like(value), Array[[]])

def use_mixed_union(value: int | Array[[2, 3]]) -> None:
    assert_type(array_like(value), Array[IntTuple])

def same_shape[Shape: IntTuple](
    left: ArrayLike[Shape], right: ArrayLike[Shape]
) -> Array[Shape]: ...

assert_type(same_shape(1, 2), Array[[]])

def shared_shape_rejects_widening(
    value: int | Array[[2]], array: Array[[2]]
) -> None:
    same_shape(
        value,  # E: is not assignable to parameter `left`
        array,
    )

default_scalar: Scalar = 1
empty_scalar: Scalar[[], int] = 1
gradual_scalar: Scalar[IntTuple, int] = 1
any_shape_scalar: Scalar[Any, int] = 1
positive_scalar: Scalar[[2], int] = 1  # E: is not assignable to `Never`
positive_gradual_scalar: Scalar[[Any], int] = 1  # E: is not assignable to `Never`
bad_shape: Scalar[int, int]  # E: is not assignable to upper bound `IntTuple`
too_many: Scalar[[], int, str]  # E: Expected 2 type arguments for `Scalar`, got 3

assert_type(default_scalar, bool | int | float | complex)
assert_type(empty_scalar, int)
assert_type(gradual_scalar, int)
assert_type(any_shape_scalar, int)
assert_type(consume(empty_scalar), Array[[]])

def consume_never(x: Scalar[[2], int]) -> Never:
    return x

def consume_tuple_empty(x: Scalar[tuple[()], int]) -> int:
    return x

def consume_tuple_never(x: Scalar[tuple[int], int]) -> Never:
    return x
"#,
);

testcase!(
    scalar_is_first_class,
    shape_extensions_env(),
    r#"
from typing import Callable, Literal, Never, assert_type, overload
from shape_extensions import Elements, IntTuple, Scalar

def identity[Shape: IntTuple](x: Scalar[Shape, int]) -> Scalar[Shape, int]:
    return x

def tuple_identity[Shape: IntTuple](
    x: Scalar[IntTuple[*Elements[Shape]], int],
) -> Scalar[IntTuple[*Elements[Shape]], int]: ...

def relay_tuple[Actual: IntTuple](
    x: Scalar[IntTuple[*Elements[Actual]], int],
) -> Scalar[IntTuple[*Elements[Actual]], int]:
    return tuple_identity(x)

@overload
def prefer_exact(x: int) -> Literal["exact"]: ...
@overload
def prefer_exact[Shape: IntTuple](x: Scalar[Shape, int]) -> Literal["scalar"]: ...
def prefer_exact(x: object) -> str: ...

@overload
def prefer_exact_reversed[Shape: IntTuple](x: Scalar[Shape, int]) -> Literal["scalar"]: ...
@overload
def prefer_exact_reversed(x: int) -> Literal["exact"]: ...
def prefer_exact_reversed(x: object) -> str: ...

def operators[Shape: IntTuple](x: Scalar[Shape, int]) -> int:
    assert_type(x + 1, int)
    assert_type(1 + x, int)
    assert_type(-x, int)
    return x.bit_length()

def scalar_result[Shape: IntTuple](x: Scalar[Shape, int]) -> Scalar[Shape, int]: ...

class Box[Shape: IntTuple]:
    value: Scalar[Shape, int]

def use_boxes(empty: Box[[]], positive: Box[[2]]) -> None:
    assert_type(empty.value, int)
    assert_type(empty.value.bit_length(), int)
    assert_type(positive.value, Never)

def return_scalar() -> Scalar[[], int]:
    return 1

values: list[Scalar[[], int]] = [1, 2]
type ScalarAlias[Shape: IntTuple] = Scalar[Shape, int]
type ScalarOrStr[Shape: IntTuple] = ScalarAlias[Shape] | str
type ImpossibleScalar = ScalarAlias[[1]]
type OrdinaryUnion = int | str
aliased: ScalarAlias[[]] = 1
impossible_alias: ScalarAlias[[1]] = 1  # E: is not assignable to `Never`
possible_union: ScalarOrStr[[1]] = "ok"
impossible_union: ScalarOrStr[[1]] = 1  # E: is not assignable to `str`
callback: Callable[[int], int] = identity

def ordinary_result[T](x: T) -> OrdinaryUnion: ...

def impossible_alias_is_never(x: ImpossibleScalar) -> Scalar[[2], int]:
    return x

def impossible_union_is_never(
    x: Scalar[[1], int] | Scalar[[2], int],
) -> Scalar[[3], int]:
    return x

def bottom_domain_is_never(x: Scalar[[], Never]) -> Scalar[[1], int]:
    return x

def suspended_bottom_domain_is_never[Shape: IntTuple](
    x: Scalar[Shape, Never],
) -> Scalar[[1], int]:
    return x

assert_type(return_scalar(), int)
assert_type(prefer_exact(1), Literal["exact"])
assert_type(prefer_exact_reversed(1), Literal["scalar"])
assert_type(scalar_result(1), int)
assert_type(scalar_result(1).bit_length(), int)
assert_type(values, list[int])
assert_type(aliased, int)
assert_type(possible_union, str)
assert_type(ordinary_result(1), OrdinaryUnion)
"#,
);
