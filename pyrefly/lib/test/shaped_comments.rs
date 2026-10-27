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
    test_shaped_comments_replace_preserves_written_annotation_errors,
    shaped_env(),
    r#"
# enable-shape-type-comments
import typing

def replaced(
    missing: typing.MissingShapeBase,  # shaped: type = int  # E: No attribute `MissingShapeBase`
    arity: list[int, int],  # shaped: type = int  # E: Expected 1 type argument
) -> None: ...
"#,
);

testcase!(
    test_shaped_comments_bracket_selectors,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray, dtype

# shapes: M, N
def transpose(
    x: ndarray[tuple[int, int], dtype[float]],  # shaped: [0] = [M, N]
) -> ndarray[tuple[int, int], dtype[float]]:  # shaped: [0] = [N, M]
    ...

# shapes: *Batch, M, N
def nested(
    x: list[ndarray[tuple[int, ...], dtype[float]]],  # shaped: [0][0] = [*Batch, M, N]
) -> ndarray[
    tuple[int, ...],  # shaped: = [*Batch, N, M]
    dtype[float],
]: ...

def check(x: ndarray[[3, 4], dtype[float]]) -> None:
    assert_type(transpose(x), ndarray[[4, 3], dtype[float]])
    assert_type(nested([x]), ndarray[[4, 3], dtype[float]])
"#,
);

testcase!(
    test_shaped_comments_bracket_quoted_dsl_and_list_payloads,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray
from shape_extensions import broadcast

# shapes: *Left, *Right
def f(
    left: "ndarray[tuple[int, ...]]",  # shaped: [0] = Left
    right: ndarray[tuple[int, ...]],  # shaped: [0] = Right
) -> ndarray[tuple[int, ...]]:  # shaped: [0] = broadcast(Left, Right)
    ...

def zero(x: ndarray) -> ndarray: ...  # shaped: [0]

# shapes: M
def invalid(
    index: ndarray[tuple[int]],  # shaped: [12] = [M]  # E: out of range
    rank: ndarray[tuple[int, int]],  # shaped: [0] = [M]  # E: does not agree
    selector: ndarray,  # shaped: [-1] = [M]  # E: Could not parse `# shaped:` payload
) -> None: ...

def check(left: ndarray[[3, 1]], right: ndarray[[1, 4]]) -> None:
    assert_type(f(left, right), ndarray[[3, 4]])
    assert_type(zero(left), ndarray[[0]])
"#,
);

testcase!(
    test_shaped_comments_multiline_tuple_components,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray, dtype
from shape_extensions import Int

# shapes: M, N
def split(
    x: ndarray[
        tuple[int, int],  # shaped: type = [M, N]
        dtype[float],
    ],
) -> tuple[
    int,  # shaped: M
    ndarray[tuple[int, int], dtype[float]],  # shaped: .args[0] = [N, M]
]: ...

def check(x: ndarray[[3, 4], dtype[float]]) -> None:
    assert_type(split(x), tuple[Int[3], ndarray[[4, 3], dtype[float]]])
    assert_type(split(x)[1], ndarray[[4, 3], dtype[float]])
    assert_type(split(x)[0] + 1, Int[4])
"#,
);

testcase!(
    test_shaped_comments_multiline_scalar_type_argument,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from shape_extensions import IntVar

class Sized[N: IntVar]: ...

# shapes: M
def f(
    x: Sized[
        int,  # shaped: M
    ],
) -> Sized:  # shaped: M + 1
    ...

def check(x: Sized[3]) -> None:
    assert_type(f(x), Sized[4])
"#,
);

testcase!(
    test_shaped_comments_multiline_union,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray

# shapes: M, N
def f(
    x: (
        ndarray[tuple[int, int]]  # shaped: .args[0] = [M, N]
        | None
    ),
) -> ndarray:  # shaped: [N, M]
    ...

def check(x: ndarray[[3, 4]]) -> None:
    assert_type(f(x), ndarray[[4, 3]])
"#,
);

testcase!(
    test_shaped_comments_multiline_variadic_and_dsl,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray
from shape_extensions import broadcast

# shapes: *Left, *Right
def f(
    left: ndarray[
        tuple[int, ...],  # shaped: type = Left
    ],
    right: ndarray[tuple[int, ...]],  # shaped: .args[0] = Right
) -> tuple[
    ndarray,  # shaped: broadcast(Left, Right)
    ndarray,  # shaped: Left
]: ...

def check(left: ndarray[[3, 1]], right: ndarray[[1, 4]]) -> None:
    assert_type(f(left, right), tuple[ndarray[[3, 4]], ndarray[[3, 1]]])
"#,
);

testcase!(
    test_shaped_comments_multiline_quoted_components,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray

# shapes: M, N
def f(
    x: tuple[
        "ndarray[tuple[int, int]]",  # shaped: .args[0] = [M, N]
        int,  # shaped: N
    ],
) -> ndarray:  # shaped: [M, N]
    ...

def check(x: ndarray[[3, 4]]) -> None:
    assert_type(f((x, 4)), ndarray[[3, 4]])
    f((x, 3))  # E: not assignable
"#,
);

testcase!(
    test_shaped_comments_multiline_reject_overlapping_targets,
    shaped_env(),
    r#"
# enable-shape-type-comments
from arrays import ndarray

# shapes: M
def f(
    x: ndarray[
        tuple[int],  # shaped: type = [M]
    ],  # shaped: type = ndarray[[M]]  # E: disjoint type components
    y: tuple[
        ndarray[tuple[int]],  # shaped: .args[0] = [M]
    ],  # shaped: .args[0].args[0] = [M]  # E: disjoint type components
) -> None: ...
"#,
);

testcase!(
    test_shaped_comments_multiline_metadata_is_not_a_type,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import Annotated, Callable, Literal
from arrays import ndarray

# shapes: M
def f(
    x: Literal[
        3,  # shaped: M  # E: cannot select literal values
    ],
    y: Annotated[
        ndarray,
        "metadata",  # shaped: M  # E: cannot select literal values or annotation metadata
    ],
    z: Callable[
        [int],  # shaped: M  # E: cannot select a list-valued or ellipsis
        int,
    ],
) -> None: ...
"#,
);

testcase!(
    test_shaped_comments_multiline_assignments_and_qualifiers,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import ClassVar, assert_type
from arrays import ndarray
from shape_extensions import Int, Shaped

class C:
    values: ClassVar[
        tuple[
            ndarray[tuple[int]],  # shaped: .args[0] = [3]
            int,  # shaped: 3
        ]
    ]

    # shapes: M, N
    def store(
        self,
        x: ndarray[
            tuple[int, int],  # shaped: type = [M, N]
        ],
    ) -> None:
        local: tuple[
            ndarray[tuple[int, int]],  # shaped: .args[0] = [M, N]
            ndarray[tuple[int, int]],  # shaped: .args[0] = [M, N]
        ] = (x, x)
        assert_type(local[0], Shaped[ndarray, "[M, N]"])
        assert_type(local[1], Shaped[ndarray, "[M, N]"])

assert_type(C.values, tuple[ndarray[[3]], Int[3]])
"#,
);

testcase!(
    test_shaped_comments_multiline_siblings_on_one_line_are_ambiguous,
    shaped_env(),
    r#"
# enable-shape-type-comments

def f(
    x: tuple[
        int, int,  # shaped: 3  # E: could belong to several annotations
    ],
) -> None: ...
"#,
);

testcase!(
    test_shaped_comments_multiline_starred_and_non_int_dimensions_diagnose,
    shaped_env(),
    r#"
# enable-shape-type-comments
from shape_extensions import Int, IntTuple

# shapes: M
def f(
    x: tuple[
        *tuple[int, ...],  # shaped: type = [M]  # E: cannot target a starred type component
    ],
    y: Int[
        3,  # shaped: M  # E: concise shape comment on a scalar type argument requires `int`
    ],
    z: IntTuple[
        *tuple[int, ...],  # shaped: type = [M]  # E: cannot target a starred type component
    ],
) -> None: ...
"#,
);

testcase!(
    test_shaped_comments_compact_transpose_and_scalar,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray, dtype
from shape_extensions import Int

# shapes: M, N
def transpose(
    n: int,  # shaped: M
    x: ndarray[tuple[int, int], dtype[float]],  # shaped: .0 = [M, N]
) -> ndarray[tuple[int, int], dtype[float]]:  # shaped: .0 = [N, M]
    ...

# shapes: M
def increment(
    n: int,  # shaped: = Int[M]
) -> int:  # shaped: M + 1
    ...

def check(x: ndarray[[3, 4], dtype[float]]) -> None:
    assert_type(transpose(3, x), ndarray[[4, 3], dtype[float]])
    transpose(2, x)  # E: not assignable
    assert_type(increment(3), Int[4])
"#,
);

testcase!(
    test_shaped_comments_compact_nested_and_multiline,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray, dtype

# shapes: *Batch, M, N
def f(
    x: list[ndarray[tuple[int, ...], dtype[float]]],  # shaped: .0.0 = [*Batch, M, N]
) -> tuple[
    ndarray[
        tuple[int, ...],  # shaped: = [*Batch, N, M]
        dtype[float],
    ],
    ndarray[tuple[int, ...]],  # shaped: .0 = Batch
]: ...

def check(x: list[ndarray[[2, 3, 4], dtype[float]]]) -> None:
    assert_type(f(x), tuple[ndarray[[2, 4, 3], dtype[float]], ndarray[[2]]])
"#,
);

testcase!(
    test_shaped_comments_compact_quoted_dsl_and_carrier_validation,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray
from shape_extensions import broadcast

# shapes: *Left, *Right
def f(
    left: "ndarray[tuple[int, ...]]",  # shaped: .0 = Left
    right: ndarray[tuple[int, ...]],  # shaped: .0 = Right
) -> ndarray[tuple[int, ...]]:  # shaped: .0 = broadcast(Left, Right)
    ...

# shapes: M, N
def invalid(
    rank: tuple[int, int],  # shaped: = [M]  # E: does not agree
    index: ndarray[tuple[int]],  # shaped: .12 = [M]  # E: out of range
    malformed: ndarray,  # shaped: .0 == [M]  # E: Invalid `# shaped:` selector
) -> None: ...

def check(left: ndarray[[3, 1]], right: ndarray[[1, 4]]) -> None:
    assert_type(f(left, right), ndarray[[3, 4]])
"#,
);

testcase!(
    test_shaped_comments_compact_overlap_rejection,
    shaped_env(),
    r#"
# enable-shape-type-comments
from arrays import ndarray

# shapes: M
def f(
    x: tuple[
        ndarray[tuple[int]],  # shaped: .0 = [M]
    ],  # shaped: .0.0 = [M]  # E: disjoint type components
) -> None: ...
"#,
);

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
    test_shaped_comments_dsl_keyword_payload_is_not_a_selector,
    shaped_env(),
    r#"
# enable-shape-type-comments
from arrays import ndarray
from shape_extensions import broadcast

# shapes: M, N
def f(
    x: ndarray,  # shaped: [M, N]
) -> ndarray:  # shaped: broadcast(left=[M, 1], right=[1, N])  # E: does not accept keyword arguments
    ...
"#,
);

testcase!(
    test_shaped_comments_one_line_overloads,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type, overload
from arrays import ndarray
from shape_extensions import Shaped

# shapes: M
@overload
def f(x: Shaped[ndarray, "[M]"]) -> ndarray: ...  # shaped: [M]
# shapes: M, N
@overload
def f(x: Shaped[ndarray, "[M, N]"]) -> ndarray: ...  # shaped: [N, M]
def f(x: ndarray) -> ndarray: ...

# shapes: M
def g(x: Shaped[ndarray, "[M]"]) -> ndarray: ...  # shaped: [M + 1]

def check(x: ndarray[[3]], y: ndarray[[3, 4]]) -> None:
    assert_type(f(x), ndarray[[3]])
    assert_type(f(y), ndarray[[4, 3]])
    assert_type(g(x), ndarray[[4]])
"#,
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
    test_shaped_comments_literal_shapes_and_assignments,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type, ClassVar, Final, Literal
from arrays import ndarray

def matrix() -> ndarray[[3, 4]]: ...

x: ndarray = matrix()  # shaped: [3, 4]
y: tuple[int, int] = (3, 4)  # shaped: type = [3, 4]
z: Final[ndarray] = x  # shaped: [3, 4]

class C:
    x: ClassVar[ndarray]  # shaped: [3, 4]

assert_type(x, ndarray[[3, 4]])
assert_type(y, tuple[Literal[3], Literal[4]])
assert_type(z, ndarray[[3, 4]])
assert_type(C.x, ndarray[[3, 4]])
"#,
);

testcase!(
    test_shaped_comments_invalid_targets_and_carriers,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import Literal
from arrays import ndarray

# shapes: M, N
def f(
    x: ndarray[tuple[int, int]],  # shaped: .args[3] = [M, N]  # E: out of range
    y: int,  # shaped: .args[0] = M  # E: requires a written subscript
    z: ndarray[tuple[int, int]],  # shaped: .args[0] = [M]  # E: does not agree
    w: tuple[Literal[3]],  # shaped: type = [M]  # E: does not agree
    literal: Literal[3],  # shaped: .args[0] = M  # E: cannot select literal
) -> None: ...
"#,
);

testcase!(
    test_shaped_comments_mixed_declarations,
    shaped_env(),
    r#"
# enable-shape-type-comments
from shape_extensions import shape_vars

# shapes: M
@shape_vars("M")
def f() -> None: ...  # E: cannot combine
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
    test_shaped_comments_class_scope,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray

# shapes: M
class Encoder:
    def encode(
        self,
        x: ndarray,  # shaped: [M]
    ) -> ndarray:  # shaped: [M + 1]
        ...

    class Inner:
        def encode(
            self,
            x: ndarray,  # shaped: [M]  # E: Could not find name `M`
        ) -> None: ...

def check(e: Encoder[3], x: ndarray[[3]]) -> None:
    assert_type(e.encode(x), ndarray[[4]])
"#,
);

testcase!(
    test_shaped_comments_nested_qualifiers_and_dataclass,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import Annotated, ClassVar, Final, NotRequired, ReadOnly, TypedDict, assert_type
from dataclasses import dataclass
from arrays import ndarray

def matrix() -> ndarray[[3, 4]]: ...

class C:
    x: ClassVar[Annotated[ndarray, "metadata"]] = matrix()  # shaped: [3, 4]
    y: Final[Annotated[ndarray, "metadata"]] = matrix()  # shaped: [3, 4]

class Fields(TypedDict):
    x: ReadOnly[NotRequired[ndarray]]  # shaped: [3, 4]

@dataclass
class Box:
    x: ndarray  # shaped: [3, 4]

def check(fields: Fields) -> None:
    assert_type(C.x, ndarray[[3, 4]])
    assert_type(Box(matrix()).x, ndarray[[3, 4]])
    if "x" in fields:
        assert_type(fields["x"], ndarray[[3, 4]])
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

testcase!(
    test_shaped_comments_explicit_replacement_and_unsupported_targets,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import Final, assert_type
from arrays import ndarray
from shape_extensions import Int, Shaped

class Plain: ...

# shapes: M
def f(
    n: int,  # shaped: type = Int[M]
    x: Plain,  # shaped: type = ndarray[[M]]
    y: ndarray | None,  # shaped: type = ndarray[[M]]
    i: Int[int],  # shaped: .args[0] = M
) -> ndarray:  # shaped: [M + 1]
    ...

# shapes: M
def invalid(
    x: Shaped[ndarray, "[M]"],  # shaped: type = ndarray[[M]]  # E: cannot target an existing `Shaped`
    y: Shaped[ndarray, "[M]"],  # shaped: .args[0] = ndarray[[M]]  # E: cannot select inside an existing `Shaped`
    z: tuple[int, ...],  # shaped: .args[1] = int  # E: cannot select a list-valued or ellipsis
) -> None: ...

bare: Final = 3  # shaped: 3  # E: requires a type inside its annotation qualifier

def check(x: ndarray[[3]]) -> None:
    assert_type(f(3, x, x, 3), ndarray[[4]])
"#,
);

testcase!(
    test_shaped_comments_local_and_attribute_annotations,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray
from shape_extensions import Shaped

# shapes: M
class C:
    def store(
        self,
        x: ndarray,  # shaped: [M]
    ) -> None:
        local: ndarray = x  # shaped: [M]
        self.value: ndarray = local  # shaped: [M]
        assert_type(local, Shaped[ndarray, "[M]"])
        assert_type(self.value, Shaped[ndarray, "[M]"])
"#,
);

testcase!(
    test_shaped_comments_dead_branch,
    shaped_env(),
    r#"
# enable-shape-type-comments
import sys
from arrays import ndarray

if sys.version_info < (1, 0):
    # shapes: M
    def f(
        x: ndarray,  # shaped: [M]
    ) -> ndarray:  # shaped: [M]
        ...
"#,
);

testcase!(
    test_shaped_comments_invalid_placement_and_syntax,
    shaped_env(),
    r#"
# enable-shape-type-comments
from typing import Annotated, Callable
from arrays import ndarray

# shaped: [3]  # E: must end the line
def own_line() -> None: ...

def ambiguous(
    x: int, y: int,  # shaped: 3  # E: could belong to several
) -> None: ...

def missing_return(
    x: int,
):  # shaped: 3  # E: needs a return annotation
    ...

def unannotated(
    x,  # shaped: 3  # E: needs a parameter annotation
) -> None: ...

def invalid(
    a: ndarray,  # shaped: [3  # E: Could not parse
    b: ndarray,  # shaped: .args[-1] = [3]  # E: Invalid `# shaped:` selector
    c: Annotated[ndarray, "metadata"],  # shaped: .args[1] = [3]  # E: cannot select literal values or annotation metadata
    d: Callable[[int], int],  # shaped: .args[0] = [3]  # E: cannot select a list-valued or ellipsis type argument
) -> None: ...
"#,
);

testcase!(
    test_shaped_comments_docstring_header_and_suppression,
    shaped_env(),
    r#"
"""Module documentation."""
# enable-shape-type-comments
from typing import assert_type
from arrays import ndarray

# shapes: M  # ordinary trailing comment
def f(
    x: ndarray,  # shaped: [M]
    y: ndarray,  # shaped: [Missing]  # pyrefly: ignore[unknown-name]
) -> ndarray:  # shaped: [M + 1]
    ...

def check(x: ndarray[[3]], y: ndarray) -> None:
    assert_type(f(x, y), ndarray[[4]])
"#,
);
