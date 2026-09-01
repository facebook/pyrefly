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
