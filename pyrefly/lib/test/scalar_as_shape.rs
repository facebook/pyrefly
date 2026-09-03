/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::shape_extensions_env;
use crate::testcase;

testcase!(
    scalar_as_shape_binds_empty_shape,
    shape_extensions_env(),
    r#"
from typing import Any, Callable, Never, Protocol, TypedDict, assert_type
from shape_extensions import IntTuple, ScalarAsShape, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple]: ...

type Scalar = bool | int | float | complex
type ArrayLike[Shape: IntTuple] = Array[Shape] | ScalarAsShape[Scalar, Shape]
type MultiScalar[Shape: IntTuple] = (
    ScalarAsShape[int, IntTuple[2]] | ScalarAsShape[int, Shape]
)
type MixedScalar[Shape: IntTuple] = (
    ScalarAsShape[float, IntTuple[2]] | ScalarAsShape[int, Shape]
)
type SplitScalar[Shape: IntTuple] = (
    ScalarAsShape[int, Shape] | ScalarAsShape[float, IntTuple[2]]
)
type GenericMultiScalar[T, Shape: IntTuple] = (
    ScalarAsShape[T, IntTuple[2]] | ScalarAsShape[T, Shape]
)
type OrdinaryConstrained = int | ScalarAsShape[float, IntTuple[2]]
type OrdinaryGeneric[Shape: IntTuple] = int | ScalarAsShape[float, Shape]

def unary[Shape: IntTuple](x: ArrayLike[Shape]) -> Array[Shape]: ...
def direct[Shape: IntTuple](x: ScalarAsShape[Scalar, Shape]) -> Array[Shape]: ...
def binary[Left: IntTuple, Right: IntTuple](
    left: ArrayLike[Left], right: ArrayLike[Right]
) -> tuple[Array[Left], Array[Right]]: ...
def ternary[A: IntTuple, B: IntTuple, C: IntTuple](
    a: ArrayLike[A], b: ArrayLike[B], c: ArrayLike[C]
) -> tuple[Array[A], Array[B], Array[C]]: ...
def same_shape[Shape: IntTuple](
    left: ArrayLike[Shape], right: ArrayLike[Shape]
) -> Array[Shape]: ...
def fixed_shape(x: ScalarAsShape[int, IntTuple[2]]) -> None: ...
def multi_scalar[Shape: IntTuple](x: MultiScalar[Shape]) -> Array[Shape]: ...
def variadic[Shape: IntTuple](*values: ScalarAsShape[int, Shape]) -> Array[Shape]: ...
def keyword_variadic[Shape: IntTuple](
    **values: ScalarAsShape[int, Shape],
) -> Array[Shape]: ...
def splat_scalars[Shape: IntTuple](
    *values: ScalarAsShape[Scalar, Shape],
) -> Array[Shape]: ...
def split_scalar[Shape: IntTuple](x: SplitScalar[Shape]) -> Array[Shape]: ...
def ordinary_constrained(x: OrdinaryConstrained) -> None: ...
def ordinary_generic[Shape: IntTuple](x: OrdinaryGeneric[Shape]) -> Array[Shape]: ...
def scalar_kwargs[Shape: IntTuple](
    **values: ScalarAsShape[int, Shape],
) -> Array[Shape]: ...
def generic_then_valid[T: str, Shape: IntTuple](
    x: ScalarAsShape[T, Shape] | ScalarAsShape[int, Shape],
) -> Array[Shape]: ...
def bounded_valid_target[Shape: IntTuple](
    x: ScalarAsShape[int, Shape] | ScalarAsShape[float, Shape],
) -> Array[Shape]: ...
def bounded_invalid_target[Shape: IntTuple](
    x: ScalarAsShape[int, Shape] | ScalarAsShape[str, IntTuple[2]],
) -> Array[Shape]: ...

def use_array(array: Array[[2, 3]]) -> None:
    assert_type(unary(array), Array[[2, 3]])
    assert_type(binary(array, 1), tuple[Array[[2, 3]], Array[[]]])
    assert_type(binary(1, array), tuple[Array[[]], Array[[2, 3]]])
    assert_type(
        ternary(array, 1.0, 1j),
        tuple[Array[[2, 3]], Array[[]], Array[[]]],
    )
    same_shape(1, array)  # E: Argument `Array[[2, 3]]` is not assignable to parameter `right`
    same_shape(array, 1)  # E: Scalar argument is treated as having shape `()`, which is not assignable to parameter `right` with shape `IntTuple[2, 3]`

assert_type(unary(True), Array[[]])
assert_type(unary(x=True), Array[[]])
assert_type(direct(True), Array[[]])
assert_type(unary(1), Array[[]])
assert_type(unary(1.0), Array[[]])
assert_type(unary(1j), Array[[]])
assert_type(multi_scalar(1), Array[[]])
assert_type(variadic(1, 2), Array[[]])
assert_type(keyword_variadic(left=1, right=2), Array[[]])
fixed_shape(1)  # E: Scalar argument is treated as having shape `()`, which is not assignable to parameter `x` with shape `IntTuple[2]`

def use_splats(
    valid: tuple[int, *tuple[float, ...], complex],
    invalid: tuple[int, *tuple[float, ...], str],
) -> None:
    assert_type(splat_scalars(*valid), Array[[]])
    splat_scalars(*invalid)  # E: Argument `str` is not assignable to parameter `*values`

def use_scalar_union(value: int | float) -> None:
    split_scalar(value)  # E: Scalar argument is treated as having shape `()`, which is not assignable to parameter `x` with shape `IntTuple[2]`
    ordinary_constrained(value)  # E: Scalar argument is treated as having shape `()`, which is not assignable to parameter `x` with shape `IntTuple[2]`
    assert_type(ordinary_generic(value), Array[[]])

def use_mixed_array_scalar_unions(
    array_first: Array[[2, 3]] | int,
    scalar_first: int | Array[[2, 3]],
) -> None:
    unary(array_first)  # E: is not assignable to parameter `x`
    unary(scalar_first)  # E: is not assignable to parameter `x`

def use_array_first_bound[T: Array[[2, 3]] | int](value: T) -> None:
    unary(value)  # E: is not assignable to parameter `x`

def use_scalar_first_bound[T: int | Array[[2, 3]]](value: T) -> None:
    unary(value)  # E: is not assignable to parameter `x`

class ScalarKeywordArguments(TypedDict):
    left: int
    right: int

class InvalidScalarKeywordArguments(TypedDict):
    left: int
    right: str

def use_typed_dict_kwargs(
    valid: ScalarKeywordArguments,
    invalid: InvalidScalarKeywordArguments,
) -> None:
    assert_type(scalar_kwargs(**valid), Array[[]])
    scalar_kwargs(**invalid)  # E: Argument `str` is not assignable to parameter `right`

assert_type(generic_then_valid(1), Array[[]])

def use_valid_bound[T: int | float](value: T) -> None:
    assert_type(bounded_valid_target(value), Array[[]])

def use_invalid_bound[T: int | str](value: T) -> None:
    bounded_invalid_target(value)  # E: Scalar argument is treated as having shape `()`, which is not assignable to parameter `x` with shape `IntTuple[2]`

unary("not a scalar")  # E: Argument `Literal['not a scalar']` is not assignable to parameter `x`

def use_any(any_value: Any) -> None:
    assert_type(unary(any_value), Array[IntTuple])

def use_unknown(value) -> None:
    assert_type(unary(value), Array[IntTuple])

def unreachable(never: Never) -> None:
    assert_type(unary(never), Array[IntTuple])

def body[Shape: IntTuple](x: ScalarAsShape[int, Shape]) -> int:
    assert_type(x, int)
    return x

type ScalarShape[Shape: IntTuple] = ScalarAsShape[Scalar, Shape]
type ScalarShapeAlias[Shape: IntTuple] = ScalarShape[Shape]

def aliased_body[Shape: IntTuple](x: ScalarShapeAlias[Shape]) -> Scalar:
    assert_type(x, Scalar)
    return x

def callback[Shape: IntTuple](x: ScalarAsShape[int, Shape]) -> int:
    return x

def arraylike_callback[Shape: IntTuple](x: ArrayLike[Shape]) -> Array[Shape]: ...

callable_value: Callable[[int], int] = callback
scalar_arraylike_callable: Callable[[int], Array[[]]] = arraylike_callback
array_arraylike_callable: Callable[[Array[[2, 3]]], Array[[2, 3]]] = arraylike_callback

class CallbackProtocol(Protocol):
    def __call__(self, x: int) -> int: ...

protocol_value: CallbackProtocol = callback

class MarkerCallbackProtocol(Protocol):
    def __call__(self, x: ScalarAsShape[int, IntTuple]) -> int: ...

marker_protocol_value: MarkerCallbackProtocol = callback

def union_marker_callback[Shape: IntTuple](
    x: int | ScalarAsShape[float, Shape],
) -> int:
    return 0

class UnionMarkerCallbackProtocol(Protocol):
    def __call__(self, x: ScalarAsShape[float, IntTuple]) -> int: ...

union_marker_protocol_value: UnionMarkerCallbackProtocol = union_marker_callback

def multi_scalar_callback[Shape: IntTuple](x: MultiScalar[Shape]) -> int:
    return 0

def mixed_scalar_callback[Shape: IntTuple](x: MixedScalar[Shape]) -> int:
    return 0

multi_scalar_protocol_value: CallbackProtocol = multi_scalar_callback
mixed_scalar_protocol_value: CallbackProtocol = mixed_scalar_callback

def generic_marker_callback[T, Shape: IntTuple](
    x: ScalarAsShape[T, Shape],
) -> T:
    return x

def generic_multi_scalar_callback[T, Shape: IntTuple](
    x: GenericMultiScalar[T, Shape],
) -> T:
    return x

class GenericMarkerProtocol[T](Protocol):
    def __call__[Shape: IntTuple](
        self, x: ScalarAsShape[T, Shape]
    ) -> T: ...

generic_marker_protocol_value: GenericMarkerProtocol[int] = generic_marker_callback
generic_multi_protocol_value: CallbackProtocol = generic_multi_scalar_callback

def constrained_union_callback[Shape: IntTuple](
    x: ScalarAsShape[int, Shape] | ScalarAsShape[float, IntTuple[2]],
) -> None: ...

constrained_union_callable: Callable[[int | float], None] = constrained_union_callback  # E: is not assignable to `(float | int) -> None`

def first_shape_error_callback(
    x: ScalarAsShape[int, IntTuple[2]] | ScalarAsShape[int, IntTuple[3]],
) -> None: ...

first_shape_error_callable: Callable[[int], None] = first_shape_error_callback  # E: is not assignable to `(int) -> None`

class ScalarArrayLikeProtocol(Protocol):
    def __call__(self, x: int) -> Array[[]]: ...

class ArrayArrayLikeProtocol(Protocol):
    def __call__(self, x: Array[[2, 3]]) -> Array[[2, 3]]: ...

scalar_protocol_value: ScalarArrayLikeProtocol = arraylike_callback
array_protocol_value: ArrayArrayLikeProtocol = arraylike_callback

def decorator(f: Callable[[int], int]) -> Callable[[int], int]:
    return f

@decorator
def decorated(x: ScalarAsShape[int, IntTuple]) -> int:
    return x

class Base:
    def method(self, x: int) -> int:
        return x

class Child(Base):
    def method[Shape: IntTuple](self, x: ScalarAsShape[int, Shape]) -> int:
        return x

class MultiScalarChild(Base):
    def method[Shape: IntTuple](self, x: MultiScalar[Shape]) -> int:
        return 0

class MixedScalarChild(Base):
    def method[Shape: IntTuple](self, x: MixedScalar[Shape]) -> int:
        return 0
"#,
);

testcase!(
    scalar_as_shape_prefers_ordinary_union_arm,
    shape_extensions_env(),
    r#"
from typing import assert_type
from shape_extensions import IntTuple, ScalarAsShape, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple]: ...

type PreferOrdinary[Shape: IntTuple] = int | ScalarAsShape[int, Shape]

def f[Shape: IntTuple](x: PreferOrdinary[Shape]) -> Array[Shape]: ...

assert_type(f(1), Array[IntTuple])
"#,
);

testcase!(
    scalar_as_shape_does_not_beat_exact_overload,
    shape_extensions_env(),
    r#"
from typing import Any, Literal, TypedDict, assert_type, overload
from shape_extensions import IntTuple, ScalarAsShape, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple]: ...

type ArrayLike[Shape: IntTuple] = Array[Shape] | ScalarAsShape[int, Shape]

@overload
def choose[Shape: IntTuple](x: ScalarAsShape[int, Shape]) -> Array[Shape]: ...
@overload
def choose(x: int) -> Literal["exact"]: ...
def choose(x: object) -> object:
    return x

assert_type(choose(1), Literal["exact"])

def ambiguous_calls(any_value: Any, unknown_value) -> None:
    assert_type(choose(any_value), Any)
    assert_type(choose(unknown_value), Any)

@overload
def choose_specific(x: object) -> Literal["object"]: ...
@overload
def choose_specific[Shape: IntTuple](
    x: ScalarAsShape[int, Shape],
) -> Literal["scalar"]: ...
def choose_specific(x: object) -> str:
    return ""

assert_type(choose_specific(1), Literal["scalar"])

@overload
def choose_fewer[A: IntTuple, B: IntTuple](
    x: ScalarAsShape[int, A], y: ScalarAsShape[int, B]
) -> Literal["two"]: ...
@overload
def choose_fewer[A: IntTuple](
    x: ScalarAsShape[int, A], y: int
) -> Literal["one"]: ...
def choose_fewer(x: int, y: int) -> str:
    return ""

assert_type(choose_fewer(1, 2), Literal["one"])

@overload
def choose_star[Shape: IntTuple](
    first: ScalarAsShape[int, Shape], second: int
) -> Literal["converted"]: ...
@overload
def choose_star(first: int, second: int) -> Literal["exact"]: ...
def choose_star(first: int, second: int) -> str:
    return ""

assert_type(choose_star(*(1, 2)), Literal["exact"])

@overload
def choose_gradual[Shape: IntTuple](
    first: ScalarAsShape[int, Shape], second: int
) -> Literal["scalar"]: ...
@overload
def choose_gradual(first: object, second: object) -> Literal["object"]: ...
def choose_gradual(first: object, second: object) -> str:
    return ""

def gradual_overload_calls(any_value: Any, unknown_value) -> None:
    assert_type(choose_gradual(1, any_value), Any)
    assert_type(choose_gradual(1, unknown_value), Any)

class GradualKeywordArguments(TypedDict):
    second: Any

def gradual_kwargs(values: GradualKeywordArguments) -> None:
    assert_type(choose_gradual(1, **values), Any)

@overload
def ordinary_gradual(first: int, second: int) -> Literal["int"]: ...
@overload
def ordinary_gradual(first: object, second: object) -> Literal["object"]: ...
def ordinary_gradual(first: object, second: object) -> str:
    return ""

def ordinary_gradual_kwargs(values: GradualKeywordArguments) -> None:
    assert_type(ordinary_gradual(1, **values), Literal["int"])

@overload
def arraylike_gradual[Shape: IntTuple](
    first: ArrayLike[Shape], second: int
) -> Literal["arraylike"]: ...
@overload
def arraylike_gradual(first: object, second: object) -> Literal["object"]: ...
def arraylike_gradual(first: object, second: object) -> str:
    return ""

class ArrayLikeGradualKeywordArguments(TypedDict):
    first: Array[[2, 3]]
    second: Any

def arraylike_gradual_kwargs(values: ArrayLikeGradualKeywordArguments) -> None:
    assert_type(arraylike_gradual(**values), Literal["arraylike"])
"#,
);

testcase!(
    scalar_as_shape_rejects_invalid_positions,
    shape_extensions_env(),
    r#"
from typing import Any, Never
from shape_extensions import IntTuple, ScalarAsShape

type Scalar = int | float
type ScalarShape[Shape: IntTuple] = ScalarAsShape[Scalar, Shape]
type Nested[Shape: IntTuple] = list[ScalarAsShape[Scalar, Shape]]  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
type AliasOfAlias[Shape: IntTuple] = ScalarShape[Shape]
type IndirectNested[Shape: IntTuple] = list[AliasOfAlias[Shape]]  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
type Alias2[Shape: IntTuple] = AliasOfAlias[Shape]
type Alias3[Shape: IntTuple] = Alias2[Shape]
type Alias4[Shape: IntTuple] = Alias3[Shape]
type Alias5[Shape: IntTuple] = Alias4[Shape]
type Alias6[Shape: IntTuple] = Alias5[Shape]
type Alias7[Shape: IntTuple] = Alias6[Shape]
type Alias8[Shape: IntTuple] = Alias7[Shape]
type Alias9[Shape: IntTuple] = Alias8[Shape]
type DeepNested[Shape: IntTuple] = list[Alias9[Shape]]  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
type DiamondLeft[Shape: IntTuple] = Alias9[Shape]
type DiamondRight[Shape: IntTuple] = Alias9[Shape]
type Diamond[Shape: IntTuple] = tuple[
    DiamondLeft[Shape],  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
    DiamondRight[Shape],  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
]
type Recursive[Shape: IntTuple] = ScalarShape[Shape] | list[Recursive[Shape]]  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
type RecursiveGeneric[T] = T | list[RecursiveGeneric[T]]
type Shared[T] = tuple[RecursiveGeneric[T], RecursiveGeneric[T]]
type Rotate[T, U] = T | list[Rotate[U, U]]

def bad_return() -> ScalarShape[IntTuple]: ...  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
def bad_nested[Shape: IntTuple](x: list[ScalarShape[Shape]]) -> None: ...  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
def bad_nested_source[Shape: IntTuple](x: ScalarAsShape[ScalarShape[Shape], Shape]) -> None: ...  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
def bad_shape(x: ScalarAsShape[Scalar, int]) -> None: ...  # E: Second argument to `ScalarAsShape` must be an `IntTuple`, got `int`
def bad_any[Shape: IntTuple](x: ScalarAsShape[Any, Shape]) -> None: ...  # E: First argument to `ScalarAsShape` may not contain `Any` or `Never`, got `Any`
def bad_unknown[Shape: IntTuple](x: ScalarAsShape[list, Shape]) -> None: ...  # E: First argument to `ScalarAsShape` may not contain `Any` or `Never`, got `list[Unknown]`
def bad_never[Shape: IntTuple](x: ScalarAsShape[Never, Shape]) -> None: ...  # E: First argument to `ScalarAsShape` may not contain `Any` or `Never`, got `Never`
def bad_default[Shape: IntTuple](x: ScalarShape[Shape] = 0) -> None: ...  # E: A parameter using `ScalarAsShape` may not have a default
def bad_specializations[Shape: IntTuple](
    x: tuple[RecursiveGeneric[int], RecursiveGeneric[ScalarShape[Shape]]],  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
) -> None: ...
def okay_shared(x: Shared[int]) -> None: ...
def bad_shared[Shape: IntTuple](x: Shared[ScalarShape[Shape]]) -> None: ...  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member
def bad_rotated_specialization[Shape: IntTuple](x: Rotate[int, ScalarShape[Shape]]) -> None: ...  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member

local: ScalarShape[IntTuple]  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member

class C:
    field: ScalarShape[IntTuple]  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member

class BadBase(ScalarAsShape[int, IntTuple]): ...  # E: `shape_extensions.ScalarAsShape` is supported only directly in a callable parameter annotation or as a direct union member

bare: ScalarAsShape  # E: `ScalarAsShape` requires two type arguments
"#,
);
