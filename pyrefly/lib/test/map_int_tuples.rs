/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use crate::test::util::TestEnv;
use crate::testcase;

fn shape_extensions_env() -> TestEnv {
    let path = PathBuf::from(
        std::env::var("SHAPE_EXTENSIONS_TEST_PATH")
            .expect("SHAPE_EXTENSIONS_TEST_PATH must be set"),
    );
    TestEnv::new_with_site_package_paths(&[path
        .to_str()
        .expect("SHAPE_EXTENSIONS_TEST_PATH must be valid UTF-8")])
}

testcase!(
    map_int_tuples_fixed_empty_and_structural,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples
from typing import Literal, assert_type

class Tensor[Shape: IntTuple]: ...

def fixed(
    x: MapIntTuples[lambda S: Tensor[S], tuple[IntTuple[2], IntTuple[3, 4]]],
) -> None:
    assert_type(x, tuple[Tensor[IntTuple[2]], Tensor[IntTuple[3, 4]]])

def empty(x: MapIntTuples[lambda S: Tensor[S], tuple[()]]) -> None:
    assert_type(x, tuple[()])

def structural(
    x: MapIntTuples[
        lambda S: Tensor[S],
        tuple[tuple[Literal[2]], tuple[Literal[3], Literal[4]]],
    ],
) -> None:
    assert_type(x, tuple[Tensor[IntTuple[2]], Tensor[IntTuple[3, 4]]])

def union_source(
    x: MapIntTuples[
        lambda S: Tensor[S],
        tuple[IntTuple[2]] | tuple[IntTuple[3, 4]],
    ],
) -> None:
    assert_type(
        x,
        tuple[Tensor[IntTuple[2]]] | tuple[Tensor[IntTuple[3, 4]]],
    )
"#,
);

testcase!(
    map_int_tuples_unbounded_gradual_bottom_and_unpacked,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples
from typing import Any, Never, assert_type

class Tensor[Shape: IntTuple]: ...

def unbounded(
    x: MapIntTuples[lambda S: Tensor[S], tuple[IntTuple[2], ...]],
) -> None:
    assert_type(x, tuple[Tensor[IntTuple[2]], ...])

def gradual(x: MapIntTuples[lambda S: Tensor[S], Any]) -> None:
    assert_type(x, tuple[Tensor[IntTuple], ...])

def bottom(x: MapIntTuples[lambda S: Tensor[S], Never]) -> None:
    assert_type(x, Never)

def unpacked(
    x: MapIntTuples[
        lambda S: Tensor[S],
        tuple[IntTuple[1], *tuple[IntTuple[2], ...], IntTuple[3, 4]],
    ],
) -> None:
    assert_type(
        x,
        tuple[
            Tensor[IntTuple[1]],
            *tuple[Tensor[IntTuple[2]], ...],
            Tensor[IntTuple[3, 4]],
        ],
    )
"#,
);

testcase!(
    map_int_tuples_mapper_is_an_int_tuple_type_parameter,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples
from typing import assert_type

class Tensor[Shape: IntTuple]: ...
class NeedsInt[Value: int]: ...

def captured[Prefix: IntTuple](
    x: MapIntTuples[
        lambda S: tuple[Tensor[Prefix], Tensor[S]],
        tuple[IntTuple[2], IntTuple[3, 4]],
    ],
) -> None:
    assert_type(
        x,
        tuple[
            tuple[Tensor[Prefix], Tensor[IntTuple[2]]],
            tuple[Tensor[Prefix], Tensor[IntTuple[3, 4]]],
        ],
    )

invalid: MapIntTuples[lambda S: NeedsInt[S], tuple[IntTuple[2]]]  # E: is not assignable to upper bound `int`
"#,
);

testcase!(
    map_int_tuples_mapper_substitution_normalizes_shapes,
    shape_extensions_env(),
    r#"
from shape_extensions import Elements, IntTuple, MapIntTuples
from typing import assert_type

def f(
    x: MapIntTuples[
        lambda S: IntTuple[2, *Elements[S], 3],
        tuple[IntTuple[4]],
    ],
) -> None:
    assert_type(x, tuple[IntTuple[2, 4, 3]])
"#,
);

testcase!(
    map_int_tuples_rejects_invalid_sources,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples

outer: MapIntTuples[lambda S: tuple[S], int]  # E: Source argument to `MapIntTuples` must be an `IntTuples` value, got `int`
listed: MapIntTuples[lambda S: tuple[S], list[IntTuple]]  # E: Source argument to `MapIntTuples` must be an `IntTuples` value, got `list[IntTuple]`
fixed: MapIntTuples[lambda S: tuple[S], tuple[int]]  # E: Source argument to `MapIntTuples` must contain only `IntTuple` values
unbounded: MapIntTuples[lambda S: tuple[S], tuple[int, ...]]  # E: Source argument to `MapIntTuples` must contain only `IntTuple` values
mixed: MapIntTuples[lambda S: tuple[S], tuple[IntTuple[2], int]]  # E: Source argument to `MapIntTuples` must contain only `IntTuple` values
unpacked: MapIntTuples[lambda S: tuple[S], tuple[IntTuple[1], *tuple[IntTuple[2], int]]]  # E: Source argument to `MapIntTuples` must contain only `IntTuple` values
"#,
);

testcase!(
    map_int_tuples_requires_unary_lambda,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples

missing: MapIntTuples[lambda S: tuple[S]]  # E: Expected 2 type arguments for `MapIntTuples`, got 1
extra: MapIntTuples[lambda S: tuple[S], tuple[IntTuple], tuple[IntTuple]]  # E: Expected 2 type arguments for `MapIntTuples`, got 3
not_lambda: MapIntTuples[int, tuple[IntTuple]]  # E: First argument to `MapIntTuples` must be a lambda
invalid_source: MapIntTuples[int, int]  # E: First argument to `MapIntTuples` must be a lambda  # E: Source argument to `MapIntTuples` must be an `IntTuples` value, got `int`
invalid_source_syntax: MapIntTuples[int, lambda T: T]  # E: First argument to `MapIntTuples` must be a lambda  # E: Expected a type form, got instance of `(T: Unknown) -> Unknown`
zero: MapIntTuples[lambda: IntTuple, tuple[IntTuple]]  # E: Mapper for `MapIntTuples` must have exactly one positional parameter
two: MapIntTuples[lambda S, T: tuple[S, T], tuple[IntTuple]]  # E: Mapper for `MapIntTuples` must have exactly one positional parameter
default: MapIntTuples[lambda S=IntTuple: S, tuple[IntTuple]]  # E: Mapper for `MapIntTuples` must have exactly one positional parameter without a default
positional_only: MapIntTuples[lambda S, /: S, tuple[IntTuple]]  # E: Mapper for `MapIntTuples` must have exactly one positional parameter
"#,
);

testcase!(
    map_int_tuples_rejects_bare_use,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples
from typing import reveal_type

def bare(x: MapIntTuples) -> None:  # E: Expected 2 type arguments for `MapIntTuples`, got 0
    reveal_type(x)  # E: revealed type: Unknown

Alias = MapIntTuples
def aliased(x: Alias) -> None:  # E: Expected 2 type arguments for `MapIntTuples`, got 0
    reveal_type(x)  # E: revealed type: Unknown

def subscripted(x: MapIntTuples[lambda S: tuple[S], tuple[IntTuple[2]]]) -> None:
    pass
"#,
);

testcase!(
    map_int_tuples_lambda_is_special_only_in_first_argument,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples

ordinary: tuple[lambda S: S]  # E: Expected a type form, got instance of `(S: Unknown) -> Unknown`
second: MapIntTuples[lambda S: S, lambda T: T]  # E: Expected a type form, got instance of `(T: Unknown) -> Unknown`
"#,
);

testcase!(
    map_int_tuples_value_position,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples
from typing_extensions import TypeForm

value: TypeForm[tuple[tuple[IntTuple[2]]]] = MapIntTuples[
    lambda S: tuple[S],
    tuple[IntTuple[2]],
]
"#,
);

testcase!(
    map_int_tuples_nested_and_alias_spellings,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple
from shape_extensions import MapIntTuples as MapShapes
import shape_extensions as se
from typing import assert_type

class Tensor[Shape: IntTuple]: ...

Alias = MapShapes

def nested(
    x: Alias[
        lambda S: Tensor[S],
        se.MapIntTuples[lambda T: T, tuple[IntTuple[2], IntTuple[3, 4]]],
    ],
) -> None:
    assert_type(x, tuple[Tensor[IntTuple[2]], Tensor[IntTuple[3, 4]]])

def nested_capture(
    x: MapShapes[
        lambda Outer: se.MapIntTuples[
            lambda Inner: tuple[Tensor[Outer], Tensor[Inner]],
            tuple[IntTuple[3]],
        ],
        tuple[IntTuple[1], IntTuple[2]],
    ],
) -> None:
    assert_type(
        x,
        tuple[
            tuple[tuple[Tensor[IntTuple[1]], Tensor[IntTuple[3]]]],
            tuple[tuple[Tensor[IntTuple[2]], Tensor[IntTuple[3]]]],
        ],
    )
"#,
);

testcase!(
    map_int_tuples_symbolic_return_stays_deferred,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type

class Tensor[Shape: IntTuple]: ...

def mapped[Shapes: IntTuples](
    shapes: Shapes,
) -> MapIntTuples[lambda S: Tensor[S], Shapes]: ...

assert_type(
    mapped(((2,), (3, 4))),
    tuple[Tensor[IntTuple[2]], Tensor[IntTuple[3, 4]]],
)
"#,
);

testcase!(
    map_int_tuples_parameter_root_exposes_sequence_view,
    shape_extensions_env(),
    r#"
from collections.abc import Sequence
from shape_extensions import IntTuple, IntTuples, MapIntTuples
from typing import assert_type, reveal_type

class Box[Shape: IntTuple]: ...

def direct[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> None:
    assert_type(values, Sequence[Box[IntTuple]])

def nested[Shapes: IntTuples](
    values: tuple[MapIntTuples[lambda S: Box[S], Shapes]],
) -> None:
    reveal_type(values)  # E: revealed type: tuple[MapIntTuples[lambda S: Box[S], Shapes]]
"#,
);

testcase!(
    map_int_tuples_rejects_unresolved_type_var_tuple,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, MapIntTuples

def whole[*Ts](x: MapIntTuples[lambda S: tuple[S], tuple[*Ts]]) -> None:  # E: `MapIntTuples` does not support an unresolved `TypeVarTuple`
    pass

def middle[*Ts](
    x: MapIntTuples[lambda S: tuple[S], tuple[IntTuple[1], *Ts, IntTuple[2]]],  # E: `MapIntTuples` does not support an unresolved `TypeVarTuple`
) -> None:
    pass
"#,
);

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

// TODO(stroxler): Give `MapIntTuples` a result domain so one deferred map can be the source of
// another. For now its result may contain arbitrary types and cannot satisfy `IntTuples`.
testcase!(
    test_deferred_map_is_not_an_int_tuples_source,
    shape_extensions_env(),
    r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples

class Box[Shape]: ...

def nested[Shapes: IntTuples]() -> MapIntTuples[
    lambda S: Box[S],
    MapIntTuples[lambda S: S, Shapes],  # E: Source argument to `MapIntTuples` must be an `IntTuples` value
]: ...
"#,
);
