/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::shape_extensions_env;
use crate::testcase;

testcase!(
    test_map_int_tuples_pattern_infers_fixed_empty_and_keyword_arguments,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def check(x2: Box[IntTuple[2]], x34: Box[IntTuple[3, 4]]) -> None:
    assert_type(shapes((x2, x34)), tuple[IntTuple[2], IntTuple[3, 4]])
    assert_type(shapes(()), tuple[()])
    assert_type(shapes(values=(x34, x2)), tuple[IntTuple[3, 4], IntTuple[2]])
"#,
);

testcase!(
    test_map_int_tuples_pattern_preserves_list_literal_element_shapes,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def check(x2: Box[IntTuple[2]], x34: Box[IntTuple[3, 4]]) -> None:
    assert_type(shapes([x2, x34]), tuple[IntTuple[2], IntTuple[3, 4]])
    assert_type(shapes(values=[x34, x2]), tuple[IntTuple[3, 4], IntTuple[2]])
    assert_type(shapes([]), tuple[()])
"#,
);

testcase!(
    test_map_int_tuples_pattern_list_literal_boundaries,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def check(x: Box[IntTuple[2]], values: list[Box[IntTuple[2]]]) -> None:
    shapes([x, 1])  # E: is not assignable to parameter `values`
    shapes([*values])  # E: Starred list elements are not supported
    shapes([*Missing])  # E: Could not find name `Missing` # E: Starred list elements are not supported

result = shapes([Missing])  # E: Could not find name `Missing`
assert_type(result, tuple[IntTuple])
"#,
);

testcase!(
    test_map_int_tuples_pattern_promotes_list_literal_members_for_validation,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Tagged[Metadata, Shape: IntTuple]: ...

def tagged[Metadata](metadata: Metadata) -> Tagged[Metadata, IntTuple[2]]: ...

def consume[Metadata, Shapes: IntTuples](
    values: MapIntTuples[lambda S: Tagged[Metadata, S], Shapes],
    metadata: Metadata,
) -> tuple[Metadata, Shapes]: ...

assert_type(consume([tagged(1)], 2), tuple[int, tuple[IntTuple[2]]])
"#,
);

testcase!(
    test_map_int_tuples_pattern_preserves_inferred_sequence_structure,
    shape_extensions_env(),
    r#"
from collections.abc import Sequence
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def fixed(xs: tuple[Box[IntTuple[1]], Box[IntTuple[2, 3]]]) -> None:
    assert_type(shapes(xs), tuple[IntTuple[1], IntTuple[2, 3]])

def unbounded(xs: Sequence[Box[IntTuple[4]]]) -> None:
    assert_type(shapes(xs), tuple[IntTuple[4], ...])

def unpacked(
    xs: tuple[
        Box[IntTuple[1]],
        *tuple[Box[IntTuple[2]], ...],
        Box[IntTuple[3, 4]],
    ],
) -> None:
    assert_type(
        shapes(xs),
        tuple[IntTuple[1], *tuple[IntTuple[2], ...], IntTuple[3, 4]],
    )
"#,
);

testcase!(
    test_map_int_tuples_pattern_projects_fixed_and_generic_subclasses,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...
class Fixed(Box[IntTuple[2, 3]]): ...
class Generic[Shape: IntTuple](Box[Shape]): ...

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def check(fixed: Fixed, generic: Generic[IntTuple[4, 5]]) -> None:
    assert_type(shapes((fixed,)), tuple[IntTuple[2, 3]])
    assert_type(shapes((generic,)), tuple[IntTuple[4, 5]])
"#,
);

testcase!(
    test_map_int_tuples_pattern_checks_captured_parameters_and_source_bound,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Tagged[Metadata, Value]: ...

def consume[Metadata, Shapes: IntTuples](
    values: MapIntTuples[lambda S: Tagged[Metadata, S], Shapes],
    metadata: Metadata,
) -> tuple[Metadata, Shapes]: ...

def consume_bounded[Metadata: str, Shapes: IntTuples](
    values: MapIntTuples[lambda S: Tagged[Metadata, S], Shapes],
) -> Shapes: ...

def accepted(value: Tagged[int, IntTuple[2]]) -> None:
    assert_type(consume((value,), 3), tuple[int, tuple[IntTuple[2]]])
    consume((value,), "text")  # E: is not assignable to parameter `metadata`

def rejected(value: Tagged[int, str]) -> None:
    metadata, _ = consume((value,), "text")  # E: is not assignable to parameter `values`
    assert_type(metadata, str)

def captured_bound_mismatch(value: Tagged[int, IntTuple[3]]) -> None:
    # The captured Metadata bound fails during inversion, but the recovered Shapes source is valid.
    # The diagnostic therefore concerns Metadata rather than claiming the source is not assignable
    # to itself.
    consume_bounded((value,))  # E: is not assignable to parameter `values`
"#,
);

testcase!(
    test_map_int_tuples_pattern_rejects_invalid_members_and_non_sequences,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples

class Box[Shape: IntTuple]: ...

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def check(x: Box[IntTuple[2]], bad: tuple[Box[IntTuple[2]], int]) -> None:
    shapes(bad)  # E: is not assignable to parameter `values`
    shapes(1)  # E: is not assignable to parameter `values`
"#,
);

testcase!(
    test_map_int_tuples_pattern_overload_selection_and_rollback,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import Any, Literal, assert_type, overload

class Tagged[Metadata, Value]: ...

@overload
def choose[Metadata, Shapes: IntTuples](
    values: MapIntTuples[lambda S: Tagged[Metadata, S], Shapes],
    metadata: Metadata,
    flag: Literal[0],
) -> tuple[Metadata, Shapes]: ...
@overload
def choose[Metadata](values: object, metadata: Metadata, flag: Literal[1]) -> Metadata: ...
def choose(values: Any, metadata: Any, flag: Any) -> Any: ...

def check(value: Tagged[int, IntTuple[2]]) -> None:
    assert_type(choose((value,), 3, 0), tuple[int, tuple[IntTuple[2]]])
    assert_type(choose((value,), "text", 1), str)
    choose((value,), "text", 0)  # E: No matching overload found for function `choose`
"#,
);

// Every inversion probe creates and finalizes a synthetic mapper variable. Repeated rejected
// overload candidates exercise the rollback boundary that must not resurrect those variables.
testcase!(
    test_map_int_tuples_pattern_repeated_failed_probes_finish_temporary_variables,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import Any, Literal, assert_type, overload

class Box[Shape: IntTuple]: ...

@overload
def choose[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes], flag: Literal[0],
) -> Shapes: ...
@overload
def choose[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes], flag: Literal[1],
) -> tuple[Shapes]: ...
@overload
def choose(values: object, flag: Literal[2]) -> str: ...
def choose(values: Any, flag: Any) -> Any: ...

def check(x: Box[IntTuple[2]]) -> None:
    assert_type(choose((x,), 2), str)
    assert_type(choose((x,), 0), tuple[IntTuple[2]])
    assert_type(choose((x,), 1), tuple[tuple[IntTuple[2]]])
"#,
);

// The view can bind a captured variable before the recovered source fails its narrower bound.
// Both effects belong to one match, so rejecting the source must release the captured variable
// for the following parameter to solve independently.
testcase!(
    test_map_int_tuples_pattern_source_bound_failure_rolls_back_captured_parameters,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples
from typing import assert_type

class Tagged[Metadata, Shape: IntTuple]: ...

def consume[Metadata, Shapes: tuple[IntTuple[2]]](
    values: MapIntTuples[lambda S: Tagged[Metadata, S], Shapes],
    metadata: Metadata,
) -> Metadata: ...

def check(value: Tagged[int, IntTuple[3]]) -> None:
    result = consume((value,), "text")  # E: Shapes `tuple[IntTuple[3]]` recovered from this argument are not assignable
    assert_type(result, str)
"#,
);

testcase!(
    test_map_int_tuples_pattern_allows_one_evidence_source_per_parameter,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...

def duplicate[Shapes: IntTuples](
    first: MapIntTuples[lambda S: Box[S], Shapes],
    second: MapIntTuples[lambda S: Box[S], Shapes],  # E: may have only one `MapIntTuples` parameter pattern
) -> Shapes: ...

def distinct[Left: IntTuples, Right: IntTuples](
    first: MapIntTuples[lambda S: Box[S], Left],
    second: MapIntTuples[lambda S: Box[S], Right],
) -> tuple[Left, Right]: ...

def check(left: Box[IntTuple[2]], right: Box[IntTuple[3, 4]]) -> None:
    assert_type(
        distinct((left,), (right,)),
        tuple[tuple[IntTuple[2]], tuple[IntTuple[3, 4]]],
    )
"#,
);

testcase!(
    test_map_int_tuples_pattern_gradual_inputs,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import Any, assert_type

class Box[Shape: IntTuple]: ...

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def check(
    dynamic: Any,
    alternatives: tuple[Box[IntTuple[2]]] | tuple[Box[IntTuple[3, 4]]],
) -> None:
    assert_type(shapes(dynamic), tuple[IntTuple, ...])
    assert_type(shapes(alternatives), tuple[IntTuple, ...])
"#,
);

testcase!(
    test_map_int_tuples_in_callable_return,
    shape_extensions_env(),
    r#"
from collections.abc import Callable
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...

def factory[Shapes: IntTuples](
    shapes: Shapes,
) -> Callable[[], MapIntTuples[lambda S: Box[S], Shapes]]: ...

def check(shapes: tuple[IntTuple[2], IntTuple[3, 4]]) -> None:
    assert_type(factory(shapes)(), tuple[Box[IntTuple[2]], Box[IntTuple[3, 4]]])
"#,
);

testcase!(
    test_map_int_tuples_through_alias,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...
type Mapped[Shapes: IntTuples] = MapIntTuples[lambda S: Box[S], Shapes]

def forward[Shapes: IntTuples](shapes: Shapes) -> Mapped[Shapes]: ...
def reverse[Shapes: IntTuples](values: Mapped[Shapes]) -> Shapes: ...

def check(shapes: tuple[IntTuple[2]], value: Box[IntTuple[2]]) -> None:
    assert_type(forward(shapes)[0], Box[IntTuple[2]])
    assert_type(reverse((value,)), tuple[IntTuple[2]])
"#,
);

testcase!(
    test_map_int_tuples_pattern_approximates_dsl_calls_in_mapper_bodies,
    shape_extensions_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, IntTuples, MapIntTuples, type_shape_dsl_function
from typing import Callable, assert_type

class Box[Shape]: ...

@type_shape_dsl_function
def append_one(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[0], 1))

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[append_one(S)], Shapes],
) -> Shapes: ...

def nested[Shapes: IntTuples](
    values: MapIntTuples[lambda S: tuple[Box[append_one(S)]], Shapes],
) -> Shapes: ...

# A mapper body is the special callable-return position; an ordinary callable nested in a
# parameter annotation does not make shape-DSL calls legal.
def ordinary_callable(callback: Callable[[], append_one(IntTuple[1])]) -> None: ...  # E: Function call cannot be used in annotations

def check(box: Box[IntTuple[2, 1]], loose: Box[str]) -> None:
    # Inversion cannot recover the input of `append_one`, so it records a gradual `IntTuple`.
    assert_type(shapes((box,)), tuple[IntTuple])
    shapes((loose,))  # E: is not assignable to parameter `values`
    shapes((1,))  # E: is not assignable to parameter `values`

    # Approximation retains ordinary constructors around the deferred call.
    assert_type(nested(((box,),)), tuple[IntTuple])
    nested((box,))  # E: is not assignable to parameter `values`
"#,
);

testcase!(
    test_eager_map_int_tuples_finalizes_dsl_calls_from_mapper_bodies,
    shape_extensions_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, IntTuples, MapIntTuples, type_shape_dsl_function
from typing import assert_type

class Box[Shape]: ...

@type_shape_dsl_function
def append_one(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[0], 1))

@type_shape_dsl_function
def singleton(shape: IntTuple) -> IntTuples:
    return dsl.IntTuples((shape,))

type Mapped = MapIntTuples[lambda S: Box[append_one(S)], tuple[IntTuple[2]]]

def direct(
    values: MapIntTuples[lambda S: Box[append_one(S)], tuple[IntTuple[2]]],
) -> None:
    assert_type(values, tuple[Box[IntTuple[2, 1]]])

def through_alias(values: Mapped) -> None:
    assert_type(values, tuple[Box[IntTuple[2, 1]]])

def from_dsl_source() -> MapIntTuples[
    lambda S: Box[S], singleton(IntTuple[2])
]: ...

assert_type(from_dsl_source(), tuple[Box[IntTuple[2]]])
"#,
);

testcase!(
    test_map_int_tuples_rejects_dsl_calls_in_parameter_sources,
    shape_extensions_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, IntTuples, MapIntTuples, type_shape_dsl_function

class Box[Shape]: ...

@type_shape_dsl_function
def singleton(shape: IntTuple) -> IntTuples:
    return dsl.IntTuples((shape,))

def invalid(
    values: MapIntTuples[
        lambda S: Box[S], singleton(IntTuple[2])  # E: Function call cannot be used in annotations
    ],
) -> None: ...
"#,
);

testcase!(
    test_map_int_tuples_pattern_approximates_residual_maps_in_mapper_bodies,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape]: ...
class Pair[Shape, Values]: ...

def shapes[Shapes: IntTuples, Other: IntTuples](
    values: MapIntTuples[
        lambda S: Pair[S, MapIntTuples[lambda T: Box[T], Other]],
        Shapes,
    ],
    other: Other,
) -> Shapes: ...

def check(value: Pair[IntTuple[2], tuple[Box[IntTuple], ...]]) -> None:
    assert_type(
        shapes((value,), ((3,), (4, 5))),
        tuple[IntTuple],
    )
    shapes((1,), ((3,),))  # E: is not assignable to parameter `values`
"#,
);

testcase!(
    test_residual_map_mapper_finalizes_nested_dsl_calls,
    shape_extensions_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, IntTuples, MapIntTuples, type_shape_dsl_function
from typing import assert_type

class Box[Shape]: ...
class Pair[Shape, Values]: ...

@type_shape_dsl_function
def append_one(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[0], 1))

def mapped[Shapes: IntTuples, Other: IntTuples](
    values: MapIntTuples[
        lambda S: Pair[
            S,
            MapIntTuples[lambda T: Box[append_one(T)], Other],
        ],
        Shapes,
    ],
    other: Other,
) -> MapIntTuples[lambda T: Box[append_one(T)], Other]: ...

def check(value: Pair[IntTuple[2], tuple[Box[IntTuple], ...]]) -> None:
    assert_type(
        mapped((value,), ((3,), (4, 5))),
        tuple[Box[IntTuple[3, 1]], Box[IntTuple[4, 1]]],
    )
"#,
);

testcase!(
    test_deferred_int_tuple_map_composes_as_an_int_tuples_source,
    shape_extensions_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, IntTuples, MapIntTuples, type_shape_dsl_function
from typing import Literal, assert_type

class Box[Shape]: ...

@type_shape_dsl_function
def append_one(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[0], 1))

def nested[Shapes: IntTuples](shapes: Shapes) -> MapIntTuples[
    lambda S: Box[S],
    MapIntTuples[lambda S: S, Shapes],
]: ...

def nested_dsl[Shapes: IntTuples](shapes: Shapes) -> MapIntTuples[
    lambda S: Box[S],
    MapIntTuples[lambda S: append_one(S), Shapes],
]: ...

def nested_structural[Shapes: IntTuples](shapes: Shapes) -> MapIntTuples[
    lambda S: Box[S],
    MapIntTuples[lambda S: tuple[Literal[7]], Shapes],
]: ...

def check(shapes: tuple[IntTuple[2], IntTuple[3, 4]]) -> None:
    assert_type(nested(shapes), tuple[Box[IntTuple[2]], Box[IntTuple[3, 4]]])
    assert_type(nested_dsl(shapes), tuple[Box[IntTuple[2, 1]], Box[IntTuple[3, 1]]])
    assert_type(nested_structural(shapes), tuple[Box[IntTuple[7]], Box[IntTuple[7]]])
"#,
);

testcase!(
    test_map_int_tuples_pattern_unpacked_varargs_exact_and_empty,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import Unpack, assert_type

class Box[Shape: IntTuple]: ...

def starred[Shapes: IntTuples](
    *values: *MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def explicit[Shapes: IntTuples](
    *values: Unpack[MapIntTuples[lambda S: Box[S], Shapes]],
) -> Shapes: ...

def with_required_keyword[Shapes: IntTuples](
    *values: *MapIntTuples[lambda S: Box[S], Shapes],
    required: int,
) -> Shapes: ...

def check(x2: Box[IntTuple[2]], x34: Box[IntTuple[3, 4]]) -> None:
    assert_type(starred(x2, x34), tuple[IntTuple[2], IntTuple[3, 4]])
    assert_type(starred(), tuple[()])
    assert_type(starred(*[]), tuple[()])
    assert_type(explicit(x34, x2), tuple[IntTuple[3, 4], IntTuple[2]])
    assert_type(explicit(), tuple[()])
    assert_type(with_required_keyword(x2, required=0), tuple[IntTuple[2]])
    with_required_keyword(x2)  # E: Missing argument `required`
"#,
);

testcase!(
    test_map_int_tuples_pattern_unpacked_varargs_starred_sequences,
    shape_extensions_env(),
    r#"
from collections.abc import Sequence
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...

def consume[Shapes: IntTuples](
    equation: str,
    *values: *MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def fixed(xs: tuple[Box[IntTuple[2]], Box[IntTuple[3, 4]]]) -> None:
    assert_type(consume("", *xs), tuple[IntTuple[2], IntTuple[3, 4]])

def homogeneous(xs: Sequence[Box[IntTuple[5]]]) -> None:
    assert_type(consume("", *xs), tuple[IntTuple[5], ...])

def mixed(
    x: Box[IntTuple[1]],
    middle: tuple[Box[IntTuple[2]], Box[IntTuple[3]]],
    y: Box[IntTuple[4, 5]],
) -> None:
    assert_type(
        consume("", x, *middle, y),
        tuple[IntTuple[1], IntTuple[2], IntTuple[3], IntTuple[4, 5]],
    )
"#,
);

testcase!(
    test_map_int_tuples_pattern_unpacked_varargs_compose_with_shape_dsl,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples, type_shape_dsl_function
from typing import assert_type

class Box[Shape: IntTuple]: ...

@type_shape_dsl_function
def first(shapes: IntTuples) -> IntTuple:
    return shapes[0]

def collect[Shapes: IntTuples](
    *values: *MapIntTuples[lambda S: Box[S], Shapes],
) -> Box[first(Shapes)]: ...

def check(
    x: Box[IntTuple[2, 3]],
    middle: tuple[Box[IntTuple[2, 3]], ...],
) -> None:
    assert_type(collect(x, x), Box[IntTuple[2, 3]])
    assert_type(collect(x, *middle, x), Box[IntTuple[2, 3]])
"#,
);

testcase!(
    test_structural_int_tuples_union_composes_with_shape_dsl,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, type_shape_dsl_function
from typing import assert_type

class Box[Shape: IntTuple]: ...

@type_shape_dsl_function
def first(shapes: IntTuples) -> IntTuple:
    return shapes[0]

def project[
    Shapes: tuple[IntTuple[2, 3]] | tuple[IntTuple[2, 3], IntTuple[2, 3]],
](shapes: Shapes) -> Box[first(Shapes)]: ...

def project_mixed[
    Shapes: tuple[IntTuple[2]] | tuple[IntTuple[3, 4]],
](shapes: Shapes) -> Box[first(Shapes)]: ...

def check(
    shapes: tuple[IntTuple[2, 3]] | tuple[IntTuple[2, 3], IntTuple[2, 3]],
) -> None:
    assert_type(project(shapes), Box[IntTuple[2, 3]])

def check_mixed(
    shapes: tuple[IntTuple[2]] | tuple[IntTuple[3, 4]],
) -> None:
    assert_type(project_mixed(shapes), Box[IntTuple])
"#,
);

testcase!(
    test_map_int_tuples_pattern_unpacked_varargs_body_tuple_view,
    shape_extensions_env(),
    r#"
from collections.abc import Iterator
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Box[Shape: IntTuple]: ...

def inspect[Shapes: IntTuples](
    *values: *MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes:
    assert_type(values, tuple[Box[IntTuple], ...])
    assert_type(iter(values), Iterator[Box[IntTuple]])
    assert_type(values[0], Box[IntTuple])
    raise NotImplementedError
"#,
);

testcase!(
    test_map_int_tuples_pattern_unpacked_varargs_diagnostics,
    shape_extensions_env().enable_unknown_argument_type_error(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import Any

class Box[Shape: IntTuple]: ...
class Other[Shape: IntTuple]: ...

def consume[Shapes: IntTuples](
    *values: *MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def untyped(value):
    return value

def check(
    wrong: Other[IntTuple[2]],
    explicit_any: Any,
    box: Box[IntTuple[2]],
    boxes: tuple[Box[IntTuple[2]], ...],
) -> None:
    consume(wrong)  # E: is not assignable
    consume(untyped(1))  # E: The type of this argument is unknown
    consume(explicit_any)
    consume(*(untyped(1), *boxes, box))  # E: The type of this argument is unknown
    consume(*(box, *untyped(()), box))  # E: The type of this argument is unknown
    consume(*(box, *boxes, untyped(1)))  # E: The type of this argument is unknown

consume(1 + "bad")  # E: `+` is not supported between `Literal[1]` and `Literal['bad']`
"#,
);

testcase!(
    test_map_int_tuples_pattern_unpacked_varargs_boundaries,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import Annotated, TypeVarTuple, Unpack, assert_type

class Box[Shape: IntTuple]: ...

type Mapped[Shapes: IntTuples] = MapIntTuples[lambda S: Box[S], Shapes]

def aliased[Shapes: IntTuples](*values: Unpack[Mapped[Shapes]]) -> Shapes:
    assert_type(values, tuple[Box[IntTuple], ...])
    raise NotImplementedError

def annotated[Shapes: IntTuples](
    *values: Unpack[Annotated[Mapped[Shapes], "metadata"]],
) -> Shapes:
    assert_type(values, tuple[Box[IntTuple], ...])
    raise NotImplementedError

def duplicate[Shapes: IntTuples](
    first: MapIntTuples[lambda S: Box[S], Shapes],
    *rest: *MapIntTuples[lambda S: Box[S], Shapes],  # E: may have only one
) -> Shapes: ...

Ts = TypeVarTuple("Ts")
def ordinary[*Ts](*values: *Ts) -> tuple[*Ts]: ...

def check(x: Box[IntTuple[2]], y: Box[IntTuple[3, 4]]) -> None:
    assert_type(aliased(x), tuple[IntTuple[2]])
    assert_type(annotated(y), tuple[IntTuple[3, 4]])
    assert_type(ordinary(1, "x"), tuple[int, str])
"#,
);
