/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::test::util::shape_extensions_env;
use crate::testcase;

fn shaped_env() -> TestEnv {
    let mut env = shape_extensions_env();
    env.add_with_path(
        "arrays",
        "arrays.pyi",
        r#"
from typing import Any
from shape_extensions import IntTuple

class dtype[T]: ...
class ndarray[Shape: IntTuple = IntTuple, DType = Any]: ...
"#,
    );
    env
}

testcase!(
    test_shaped_carrier_and_shorthand,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import dtype, ndarray
from shape_extensions import Elements, Shaped, shape_vars

@shape_vars("M, N")
def transpose(
    x: ndarray[Shaped[tuple[int, int], "[M, N]"], dtype[float]],
) -> ndarray[Shaped[tuple[int, int], "[N, M]"], dtype[float]]: ...

@shape_vars("N")
def pad(x: Shaped[ndarray, "[N]"]) -> Shaped[ndarray, "[N + 1]"]: ...

@shape_vars("*Batch, M, N")
def batched(
    x: ndarray[Shaped[tuple[int, ...], "[*Elements[Batch], M, N]"], dtype[float]],
) -> ndarray[Shaped[tuple[int, ...], "[*Elements[Batch], N, M]"], dtype[float]]: ...

Pair = tuple[int, int]

@shape_vars("M, N")
def aliased(x: ndarray[Shaped[Pair, "[M, N]"], dtype[float]]) -> Shaped[ndarray, "[N, M]"]: ...

def check(
    matrix: ndarray[[3, 4], dtype[float]],
    vector: ndarray[[5]],
    stack: ndarray[[2, 7, 3, 4], dtype[float]],
) -> None:
    assert_type(transpose(matrix), ndarray[[4, 3], dtype[float]])
    assert_type(pad(vector), ndarray[[6]])
    assert_type(batched(stack), ndarray[[2, 7, 4, 3], dtype[float]])
    assert_type(aliased(matrix), ndarray[[4, 3]])
"#,
);

testcase!(
    test_shaped_is_annotated_outside_a_declaration,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray
from shape_extensions import Shaped

def f(x: Shaped[ndarray, "[M, N"]) -> None:
    assert_type(x, ndarray)

Alias = Shaped[ndarray, "[M]"]
value = Shaped[ndarray, "[M]"]
"#,
);

testcase!(
    test_shaped_malformed_annotations,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import dtype, ndarray
from shape_extensions import Shaped, shape_vars

@shape_vars("M, N")
def f(
    unparsable: ndarray[Shaped[tuple[int, int], "[M, N"], dtype[float]],  # E: Could not parse `Shaped` shape string
    extra: ndarray[Shaped[tuple[int, int], "[M, N]", "extra"], dtype[float]],  # E: `Shaped` takes a type and a shape string
    text: ndarray[Shaped[tuple[int], "str"], dtype[float]],  # E: A `Shaped` shape must be a list of dimensions, a variadic shape, or a shape function call, got `str`
    undefined: ndarray[Shaped[tuple[int], "Q"], dtype[float]],  # E: Could not find name `Q`
    # An appended shape is an ordinary type argument, checked as one.
    appended_text: Shaped[ndarray, "str"],  # E: `str` is not assignable to upper bound `IntTuple` of type variable `Shape`
    appended_dim: Shaped[ndarray, "M"],  # E: `M` is an `IntVar` and cannot be used as an ordinary type
    in_value: object = Shaped[ndarray, "[M"],
) -> None:
    # A malformed `Shaped` means its base alone, as it does to other checkers.
    assert_type(unparsable, ndarray[tuple[int, int], dtype[float]])
    assert_type(extra, ndarray[tuple[int, int], dtype[float]])
    assert_type(text, ndarray[tuple[int], dtype[float]])
    assert_type(undefined, ndarray[tuple[int], dtype[float]])
"#,
);

testcase!(
    test_shaped_names_must_be_valid_dimensions,
    shaped_env(),
    r#"
from arrays import ndarray
from shape_extensions import Shaped, shape_vars

BATCH = 32

@shape_vars("N")
def f(
    undeclared: Shaped[ndarray, "[Q, N]"],  # E: Could not find name `Q`
    constant: Shaped[ndarray, "[BATCH, N]"],  # E: Tensor shape dimensions must be integer literals or type variables  # E: `BATCH` is not a valid type alias
) -> None: ...
"#,
);

testcase!(
    test_shaped_scopes_nest,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray
from shape_extensions import IntVar, Shaped, shape_vars

L = IntVar("L")

@shape_vars("Dim, Hidden")
class Encoder:
    weight: Shaped[ndarray, "[Hidden, Dim]"]

    @shape_vars("Batch")
    def encode(self, x: Shaped[ndarray, "[Batch, Dim]"]) -> Shaped[ndarray, "[Batch, Hidden]"]:
        def inner(y: Shaped[ndarray, "[Batch, Dim]"]) -> Shaped[ndarray, "[Batch, Hidden]"]: ...
        return inner(x)

@shape_vars("M")
def legacy(x: Shaped[ndarray, "[M, L]"]) -> Shaped[ndarray, "[L, M]"]: ...

# `inner` captures the outer `M` as a fixed dimension and solves only its own `N`.
@shape_vars("M")
def outer(x: Shaped[ndarray, "[M]"], z: ndarray[[5, 2]]) -> Shaped[ndarray, "[2]"]:
    @shape_vars("N")
    def inner(y: Shaped[ndarray, "[M, N]"]) -> Shaped[ndarray, "[N]"]: ...
    def make() -> Shaped[ndarray, "[M, 2]"]: ...
    inner(z)  # E: Argument `ndarray[[5, 2]]` is not assignable to parameter `y` with type `ndarray[[M, @_]]`
    return inner(make())

def check(e: Encoder[4, 8], x: ndarray[[2, 4]], y: ndarray[[3, 5]]) -> None:
    assert_type(e.weight, ndarray[[8, 4]])
    assert_type(e.encode(x), ndarray[[2, 8]])
    assert_type(legacy(y), ndarray[[5, 3]])
    assert_type(outer(ndarray[[7]](), ndarray[[5, 2]]()), ndarray[[2]])
"#,
);

testcase!(
    test_shaped_declarations_do_not_mix,
    shaped_env(),
    r#"
from arrays import ndarray
from shape_extensions import Shaped, shape_vars, static_jaxtyping

@shape_vars("N")
@static_jaxtyping("M")  # E: `@shape_vars` and `@static_jaxtyping` cannot both declare one definition
def both(x: Shaped[ndarray, "[N]"]) -> None: ...

@shape_vars("N")
class Outer:
    @static_jaxtyping("N")
    def shadow(self) -> None: ...  # E: `N` is declared by `@static_jaxtyping` and is already declared by `@shape_vars` on an enclosing definition

@shape_vars("T")
class Clash[T]: ...  # E: `T` is declared by `@shape_vars` and is already a type parameter of this definition
"#,
);

testcase!(
    test_shaped_carrier_must_agree_with_its_shape,
    shaped_env(),
    r#"
from typing import Literal
from arrays import dtype, ndarray
from shape_extensions import Elements, Shaped, shape_vars

@shape_vars("*Batch, M, N")
def f(
    rank: ndarray[Shaped[tuple[int, int], "[M]"], dtype[float]],  # E: `Shaped` tuple `tuple[int, int]` does not agree with its shape `IntTuple[M]`
    literal: ndarray[Shaped[tuple[Literal[3], int], "[4, N]"], dtype[float]],  # E: `Shaped` tuple `tuple[Literal[3], int]` does not agree
    variadic: ndarray[Shaped[tuple[int, int], "[*Elements[Batch], N]"], dtype[float]],  # E: `Shaped` tuple `tuple[int, int]` does not agree
    agreeing: ndarray[Shaped[tuple[Literal[3], int], "[3, N]"], dtype[float]],
) -> None: ...
"#,
);

testcase!(
    test_shaped_base_must_take_a_shape,
    shaped_env(),
    r#"
from typing import Any, assert_type, reveal_type
from arrays import dtype, ndarray
from shape_extensions import Shaped, shape_vars

class Tensor: ...

@shape_vars("N")
def f(
    union: Shaped[int | None, "[N]"],  # E: `Shaped` needs a class, an integer tuple, or `Any`, got `int | None`
    # A class without type parameters, such as real torch's `Tensor`, ignores the shape.
    non_generic: Shaped[Tensor, "[N]"],
    # A generic class takes the shape as extra type arguments, even if it has no room for them.
    generic: Shaped[list[int], "[N]"],  # E: Expected 1 type argument for `list`, got 2  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    # A tuple of unknown rank, or `Any`, agrees with any shape.
    any_carrier: ndarray[Shaped[Any, "[N, 2]"], dtype[float]],
    bare_tuple: ndarray[Shaped[tuple, "[N, 2]"], dtype[float]],
    unbounded_tuple: ndarray[Shaped[tuple[int, ...], "[N, 2]"], dtype[float]],
) -> None:
    assert_type(union, int | None)
    assert_type(non_generic, Tensor)
    reveal_type(any_carrier)  # E: revealed type: ndarray[[N, 2], dtype[float]]
    reveal_type(bare_tuple)  # E: revealed type: ndarray[[N, 2], dtype[float]]
    reveal_type(unbounded_tuple)  # E: revealed type: ndarray[[N, 2], dtype[float]]
"#,
);

testcase!(
    test_shaped_bases_that_are_not_carriers,
    shaped_env(),
    r#"
from typing import Optional, assert_type
from arrays import dtype, ndarray
from missing import Unresolved  # E: Cannot find module `missing`
from shape_extensions import Elements, IntTuple, Shaped, shape_vars

@shape_vars("N, *S")
def f[T: IntTuple](
    # An unresolved base was already reported, so it means itself.
    unresolved: Shaped[Unresolved, "[N]"],
    # A type variable is not a carrier, even one bounded by `IntTuple`.
    type_var: ndarray[Shaped[T, "[N]"], dtype[float]],  # E: `Shaped` needs a class, an integer tuple, or `Any`, got `T`
    # A bare variadic shape has no known rank.
    variadic: ndarray[Shaped[tuple[int, int], "S"], dtype[float]],  # E: `Shaped` tuple `tuple[int, int]` does not agree
    optional: Optional[Shaped[ndarray, "[N]"]],
    union: Shaped[ndarray, "[N]"] | None,
    concatenated: Shaped[ndarray, "[N" "]"],  # E: A `Shaped` shape string must be a single string literal
) -> None:
    assert_type(type_var, ndarray[T, dtype[float]])
    assert_type(variadic, ndarray[tuple[int, int], dtype[float]])
    assert_type(concatenated, ndarray)

def g(x: ndarray[[3]]) -> None:
    @shape_vars("N")
    def h(
        optional: Optional[Shaped[ndarray, "[N]"]],
        union: Shaped[ndarray, "[N]"] | None,
    ) -> Shaped[ndarray, "[N]"]: ...
    assert_type(h(x, x), ndarray[[3]])
    h(x, ndarray[[4]]())  # E: Argument `ndarray[[4]]` is not assignable to parameter `union`
"#,
);

testcase!(
    test_shaped_call_site_mismatches,
    shaped_env(),
    r#"
from arrays import dtype, ndarray
from shape_extensions import Shaped, shape_vars

@shape_vars("M, N")
def transpose(
    x: ndarray[Shaped[tuple[int, int], "[M, N]"], dtype[float]],
) -> ndarray[Shaped[tuple[int, int], "[N, M]"], dtype[float]]: ...

@shape_vars("M, K, N")
def matmul(a: Shaped[ndarray, "[M, K]"], b: Shaped[ndarray, "[K, N]"]) -> Shaped[ndarray, "[M, N]"]: ...

def check(vector: ndarray[[5], dtype[float]], a: ndarray[[2, 3]], b: ndarray[[4, 5]]) -> None:
    transpose(vector)  # E: Argument `ndarray[[5], dtype[float]]` is not assignable to parameter `x`
    matmul(a, b)  # E: Argument `ndarray[[4, 5]]` is not assignable to parameter `b`
"#,
);

testcase!(
    test_shaped_names_resolve_only_inside_shape_strings,
    shaped_env(),
    r#"
from typing import Literal, assert_type
from arrays import ndarray
from shape_extensions import Shaped, shape_vars

N = 3

@shape_vars("M, N")
def f(x: Shaped[ndarray, "[M, N]"], y: "Shaped[ndarray, '[N]']", z: list[M]) -> None:  # E: Could not find name `M`
    local: Shaped[ndarray, "[N, M]"] = g(x)
    assert_type(local, Shaped[ndarray, "[N, M]"])

@shape_vars("M, N")
def g(x: Shaped[ndarray, "[M, N]"]) -> Shaped[ndarray, "[N, M]"]: ...

def check(x: ndarray[[2, 4]], y: ndarray[[4]]) -> None:
    f(x, y, [])
    f(x, x, [])  # E: Argument `ndarray[[2, 4]]` is not assignable to parameter `y`
    assert_type(N, Literal[3])
"#,
);

// Declared class dimensions default to gradual ones, so adding `@shape_vars` to a
// published generic class does not break a consumer's `GenericEncoder[Input]`.
testcase!(
    test_shape_vars_class_dimensions_have_defaults,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray
from shape_extensions import Shaped, shape_vars

class Input: ...

@shape_vars("A, *Rest")
class GenericEncoder[T]:
    weight: Shaped[ndarray, "[A, *Rest]"]

@shape_vars("A")
class Defaulted[T = int]: ...

@shape_vars("A", required=True)
class Strict[T]: ...

@shape_vars("A", required=True)
class StrictDefaulted[T = int]: ...  # E: Type parameter `A` without a default cannot follow type parameter `T` with a default

def check(
    g: GenericEncoder[Input],
    d: Defaulted,
    s: Strict[Input],  # E: Expected 2 type arguments for `Strict`, got 1
) -> None:
    assert_type(g.weight, ndarray[[int, *tuple[int, ...]]])
"#,
);

testcase!(
    test_shaped_appends_the_shape_to_a_generic_class,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray
from shape_extensions import IntTuple, Shaped, shape_vars

class Input: ...

@shape_vars("A, B")
class Encoder: ...

@shape_vars("A, B")
class GenericEncoder[T]: ...

class Two[S1: IntTuple = IntTuple, S2: IntTuple = IntTuple]: ...

@shape_vars("M, N")
def swap(x: Shaped[Encoder, "M, N"]) -> Shaped[Encoder, "N, M"]: ...

# `@shape_vars` class dimensions follow the class's own type parameters.
@shape_vars("M, N")
def swap_generic(
    x: Shaped[GenericEncoder[Input], "M, N"],
) -> Shaped[GenericEncoder[Input], "N, M"]: ...

# A list is a single `IntTuple` argument, so this is `Two[[3], [N]]`.
@shape_vars("N")
def second(x: Shaped[Two[[3]], "[N]"]) -> Shaped[ndarray, "[N]"]: ...

@shape_vars("M, N")
def f(
    too_many: Shaped[Encoder[3, 4], "M, N"],  # E: Expected 2 type arguments for `Encoder`, got 4  # E: `M` is an `IntVar`  # E: `N` is an `IntVar`
) -> None: ...

# A non-generic alias takes no arguments, appended ones included.
type Array = ndarray

@shape_vars("N")
def alias(x: Shaped[Array, "[N]"]) -> None: ...  # E: `TypeAlias[Array, type[ndarray]]` is not subscriptable

def check(e: Encoder[3, 4], g: GenericEncoder[Input, 3, 4], t: Two[[3], [5]]) -> None:
    assert_type(swap(e), Encoder[4, 3])
    assert_type(swap_generic(g), GenericEncoder[Input, 4, 3])
    assert_type(second(t), ndarray[[5]])
"#,
);

testcase!(
    test_shaped_int_is_a_dimension,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray
from shape_extensions import Int, Shaped, shape_vars

@shape_vars("N")
def size(x: Shaped[ndarray, "[N]"]) -> Shaped[int, "N"]: ...

@shape_vars("N")
def zeros(n: Shaped[int, "N"]) -> Shaped[ndarray, "[N]"]: ...

@shape_vars("N")
def grow(n: Shaped[int, "N"]) -> Shaped[int, "N + 1"]: ...

def check(x: ndarray[[5]]) -> None:
    assert_type(size(x), Int[5])
    assert_type(zeros(3), ndarray[[3]])
    assert_type(zeros(size(x)), ndarray[[5]])
    assert_type(zeros(grow(size(x))), ndarray[[6]])
    as_int: int = size(x)
"#,
);

testcase!(
    test_shaped_appends_to_a_tuple_bounded_shape_parameter,
    shaped_env(),
    r#"
from typing import Any, Literal, reveal_type
from shape_extensions import Shaped, shape_vars

# Real numpy's `ndarray` takes its shape as an ordinary `tuple[int, ...]`.
class NpLike[S: tuple[int, ...] = tuple[Any, ...], D = Any]: ...

@shape_vars("M, N")
def transpose(x: Shaped[NpLike, "[M, N]"]) -> Shaped[NpLike, "[N, M]"]: ...

def check(x: NpLike[tuple[Literal[3], Literal[4]]]) -> None:
    reveal_type(transpose(x))  # E: revealed type: NpLike[IntTuple[4, 3]]
"#,
);

testcase!(
    test_shaped_carrier_agreement_edge_cases,
    shaped_env(),
    r#"
from typing import Literal, assert_type
from arrays import dtype, ndarray
from shape_extensions import Elements, IntTuple, Shaped, broadcast, shape_vars

@shape_vars("*S0, *S1")
def call(
    a: ndarray[Shaped[tuple[int, ...], "[*Elements[S0]]"], dtype[float]],
    b: ndarray[Shaped[tuple[int, ...], "[*Elements[S1]]"], dtype[float]],
) -> ndarray[Shaped[tuple[int, ...], "broadcast(S0, S1)"], dtype[float]]: ...

# A shape function's result has no known rank.
@shape_vars("*S0, *S1")
def fixed_rank_call(
    a: ndarray[Shaped[tuple[int, ...], "[*Elements[S0]]"], dtype[float]],
    b: ndarray[Shaped[tuple[int, ...], "[*Elements[S1]]"], dtype[float]],
) -> ndarray[Shaped[tuple[int, int], "broadcast(S0, S1)"], dtype[float]]: ...  # E: `Shaped` tuple `tuple[int, int]` does not agree

@shape_vars("N")
def f(
    min_rank: ndarray[Shaped[tuple[int, *tuple[int, ...]], "[]"], dtype[float]],  # E: `Shaped` tuple `tuple[int, *tuple[int, ...]]` does not agree
    enough_rank: ndarray[Shaped[tuple[int, *tuple[int, ...]], "[N]"], dtype[float]],
    unknown_rank: ndarray[Shaped[tuple[int, ...], "[3, N]"], dtype[float]],
    narrower: ndarray[Shaped[tuple[Literal[3]], "[N]"], dtype[float]],  # E: `Shaped` tuple `tuple[Literal[3]]` does not agree
    literal: ndarray[Shaped[tuple[Literal[3], int], "[3, N]"], dtype[float]],
    gradual: ndarray[Shaped[tuple[int, int], "IntTuple"], dtype[float]],  # E: `Shaped` tuple `tuple[int, int]` does not agree
    gradual_any_rank: ndarray[Shaped[tuple[int, ...], "IntTuple"], dtype[float]],
) -> None: ...

@shape_vars("*S")
def g(
    literals: ndarray[Shaped[tuple[Literal[3], *tuple[int, ...]], "[4, *Elements[S]]"], dtype[float]],  # E: `Shaped` tuple `tuple[Literal[3], *tuple[int, ...]]` does not agree
    too_short: ndarray[Shaped[tuple[int, *tuple[int, ...]], "[*Elements[S]]"], dtype[float]],  # E: `Shaped` tuple `tuple[int, *tuple[int, ...]]` does not agree
    both_unpacked: ndarray[Shaped[tuple[Literal[3], *tuple[int, ...]], "[3, *Elements[S]]"], dtype[float]],
    empty_middle: ndarray[Shaped[tuple[Literal[3], *tuple[int, ...]], "[*Elements[S], 4]"], dtype[float]],  # E: `Shaped` tuple `tuple[Literal[3], *tuple[int, ...]]` does not agree
    one_in_middle: ndarray[Shaped[tuple[Literal[1], Literal[2], *tuple[int, ...]], "[*Elements[S], 1, 2]"], dtype[float]],  # E: `Shaped` tuple `tuple[Literal[1], Literal[2], *tuple[int, ...]]` does not agree
    suffixed: ndarray[Shaped[tuple[*tuple[int, ...], Literal[2]], "[*Elements[S], 3]"], dtype[float]],  # E: `Shaped` tuple `tuple[*tuple[int, ...], Literal[2]]` does not agree
    suffix_agrees: ndarray[Shaped[tuple[*tuple[int, ...], Literal[2]], "[1, *Elements[S], 2]"], dtype[float]],
    crossed: ndarray[Shaped[tuple[Literal[1], Literal[2], *tuple[int, ...]], "[1, *Elements[S], 3]"], dtype[float]],  # E: `Shaped` tuple `tuple[Literal[1], Literal[2], *tuple[int, ...]]` does not agree
) -> None:
    # A disagreeing carrier means itself, as it does to other checkers.
    assert_type(literals, ndarray[tuple[Literal[3], *tuple[int, ...]], dtype[float]])
"#,
);

testcase!(
    test_shaped_tuple_is_replaced_anywhere_in_an_annotation,
    shaped_env(),
    r#"
from typing import Any, Optional, assert_type, reveal_type
from arrays import ndarray
from shape_extensions import IntTuple, Shaped, shape_vars

@shape_vars("M, N")
def shape_of(x: Shaped[ndarray, "[M, N]"]) -> Shaped[tuple[int, int], "[M, N]"]: ...

@shape_vars("M, N")
def zeros(shape: Shaped[tuple[int, int], "[M, N]"] | None) -> Shaped[ndarray, "[M, N]"]: ...

@shape_vars("M, N")
def f(
    x: list[Shaped[tuple[int, int], "[M, N]"]],
    optional: Optional[Shaped[tuple[int, int], "[M, N]"]],
    union: Shaped[tuple[int, int], "[M, N]"] | None,
    any_shape: Shaped[Any, "[M, N]"],
) -> None:
    reveal_type(x)  # E: revealed type: list[IntTuple[M, N]]
    reveal_type(optional)  # E: revealed type: IntTuple[M, N] | None
    reveal_type(union)  # E: revealed type: IntTuple[M, N] | None
    reveal_type(any_shape)  # E: revealed type: IntTuple[M, N]

def g(x: ndarray[[3, 4]]) -> None:
    assert_type(shape_of(x), IntTuple[3, 4])
    assert_type(zeros(shape_of(x)), ndarray[[3, 4]])
    assert_type(zeros((3, 4)), ndarray[[3, 4]])
    as_tuple: tuple[int, int] = shape_of(x)
"#,
);

testcase!(
    test_shaped_reads_only_its_own_special_form,
    shaped_env(),
    r#"
from typing import Annotated, assert_type
from arrays import ndarray
from shape_extensions import Shaped, shape_vars

@shape_vars("")
def rank_zero(x: Shaped[ndarray, "[]"]) -> None:
    assert_type(x, ndarray[[]])

@shape_vars("N")
def plain(x: Annotated[ndarray, "[N]"]) -> None:
    assert_type(x, ndarray)

def shadowed() -> None:
    Shaped = Annotated
    @shape_vars("N")
    def f(x: Shaped[ndarray, "[N]"]) -> None:
        assert_type(x, ndarray)

# A type parameter bound reads the shape string too.
@shape_vars("N")
def bounded[T: Shaped[ndarray, "[N]"]](x: T) -> T: ...  # E: Type variable bounds and constraints must be concrete
"#,
);

testcase!(
    test_shaped_in_cast,
    shaped_env(),
    r#"
from typing import Any, assert_type, cast
from arrays import ndarray
from shape_extensions import IntTuple, Shaped, shape_vars

@shape_vars("N")
def symbolic(value: Any) -> None:
    assert_type(cast(Shaped[ndarray, "[N, 2]"], value), Shaped[ndarray, "[N, 2]"])

@shape_vars("")
def literal(value: Any) -> None:
    assert_type(cast(Shaped[ndarray, "[3, 2]"], value), ndarray[[3, 2]])
    assert_type(cast(typ=Shaped[ndarray, "[3, 2]"], val=value), ndarray[[3, 2]])
    assert_type(cast(list[Shaped[tuple[int, int], "[3, 2]"]], value), list[IntTuple[3, 2]])

def outside_scope(value: Any) -> None:
    assert_type(cast(Shaped[ndarray, "[N]"], value), ndarray)
"#,
);

testcase!(
    test_shaped_in_type_aliases,
    shaped_env(),
    r#"
from typing import Any, TypeAlias, TypeAliasType, assert_type, cast
from arrays import ndarray
from shape_extensions import IntTuple, Shaped, shape_vars

@shape_vars("")
class Scope:
    type Scoped = Shaped[ndarray, "[3, 2]"]
    type Nested = list[Shaped[tuple[int, int], "[3, 2]"]]
    ViaCall = TypeAliasType("ViaCall", Shaped[ndarray, "[3, 2]"])

@shape_vars("")
def literal(value: Any) -> None:
    Legacy: TypeAlias = Shaped[ndarray, "[3, 2]"]
    assert_type(cast(Legacy, value), ndarray[[3, 2]])
    assert_type(cast(Scope.Scoped, value), ndarray[[3, 2]])
    assert_type(cast(Scope.Nested, value), list[IntTuple[3, 2]])
    assert_type(cast(Scope.ViaCall, value), ndarray[[3, 2]])
"#,
);

testcase!(
    test_shaped_legacy_alias_cannot_capture_declared_dimensions,
    shaped_env(),
    r#"
from typing import TypeAlias
from arrays import ndarray
from shape_extensions import Shaped, shape_vars

@shape_vars("N")
def function(x: Shaped[ndarray, "[N]"]) -> None:
    Alias: TypeAlias = Shaped[ndarray, "[N]"]  # E: Type variable `N` is not in scope

@shape_vars("N")
class Container:
    Alias: TypeAlias = Shaped[ndarray, "[N]"]  # E: Type variable `N` is not in scope
"#,
);

testcase!(
    test_shaped_type_alias_is_not_a_class_value,
    shaped_env(),
    r#"
from typing import Annotated, TypeAlias
from shape_extensions import Shaped, shape_vars

@shape_vars("N")
class Base: ...

@shape_vars("")
class Scope:
    type Scoped = Shaped[Base, "3"]

@shape_vars("")
def check() -> None:
    Legacy: TypeAlias = Shaped[Base, "3"]
    Plain: TypeAlias = Annotated[Base[3], "3"]

    legacy: type[Base[3]] = Legacy  # E: not assignable
    scoped: type[Base[3]] = Scope.Scoped  # E: not assignable
    plain: type[Base[3]] = Plain  # E: not assignable
    Legacy()  # E: Expected a callable
    Scope.Scoped()  # E: Expected a callable
    Plain()  # E: Expected a callable
"#,
);

testcase!(
    test_shaped_in_base_class_argument,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray
from shape_extensions import IntTuple, Shaped, shape_vars

class Box[T]:
    def get(self) -> T: ...

@shape_vars("N")
class Sub(Box[Shaped[tuple[int], "[N]"]]): ...

def check() -> None:
    assert_type(Sub[3]().get(), IntTuple[3])
"#,
);

testcase!(
    test_shaped_in_base_classes,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray
from shape_extensions import IntTuple, Shaped, shape_vars

class Box[T]:
    def get(self) -> T: ...

@shape_vars("N")
class Base:
    field: Shaped[ndarray, "[N]"]

@shape_vars("N")
class Direct(Shaped[Base, "N"]): ...

@shape_vars("")
class Literal(Shaped[Base, "3"]): ...

@shape_vars("N")
class Nested(Box[Shaped[tuple[int], "[N]"]]): ...

@shape_vars("N")
class NestedArray(Box[Shaped[ndarray, "[N]"]]): ...

@shape_vars("N")
class GenericBase[T]:
    def get(self) -> T: ...

@shape_vars("")
class GenericChild(Shaped[GenericBase[int], "3"]): ...

@shape_vars("N", required=True)
class RequiredBase[T]:
    def get(self) -> T: ...

@shape_vars("")
class RequiredChild(Shaped[RequiredBase[int], "3"]): ...

class TooMany(Box[int, str]): ...  # E: Expected 1 type argument

def check() -> None:
    assert_type(Direct[3]().field, ndarray[[3]])
    assert_type(Literal().field, ndarray[[3]])
    assert_type(Nested[3]().get(), IntTuple[3])
    assert_type(NestedArray[3]().get(), ndarray[[3]])
    assert_type(GenericChild().get(), int)
    assert_type(RequiredChild().get(), int)
"#,
);

testcase!(
    test_shaped_int_base_inherits_integer_operations,
    shaped_env(),
    r#"
from typing import assert_type
from shape_extensions import Shaped, shape_vars

@shape_vars("N")
class IntChild(Shaped[int, "N"]): ...

class Plain(int): ...

@shape_vars("N")
class Invalid(Shaped[int, "Missing"]): ...  # E: Could not find name `Missing`

def check(shaped: IntChild, plain: Plain) -> None:
    assert_type(shaped + 1, int)
    assert_type(plain + 1, int)
"#,
);

testcase!(
    test_shaped_int_nested_base_argument_preserves_dimensions,
    shaped_env(),
    r#"
from typing import assert_type
from shape_extensions import Int, Shaped, shape_vars

class Box[T]:
    def get(self) -> T: ...

@shape_vars("N")
class ViaShaped(Box[Shaped[int, "N"]]): ...

def check() -> None:
    assert_type(ViaShaped[3]().get(), Int[3])
"#,
);

testcase!(
    test_shaped_base_preserves_strict_alias,
    {
        let pydantic = std::env::var("PYDANTIC_TEST_PATH").expect("Pydantic test path");
        let shapes =
            std::env::var("SHAPE_EXTENSIONS_TEST_PATH").expect("Shape extensions test path");
        TestEnv::new_with_site_package_paths(&[&pydantic, &shapes])
    },
    r#"
from typing import Annotated, TypeAlias
from pydantic import RootModel, Strict
from shape_extensions import Shaped, shape_vars

class Base(RootModel[int]): ...
StrictRoot: TypeAlias = Annotated[Base, Strict()]

class Direct(StrictRoot): ...

@shape_vars("")
class ShapedBase(Shaped[StrictRoot, "3"]): ...

Direct("3")  # E: not assignable
ShapedBase("3")  # E: not assignable
"#,
);

testcase!(
    test_shaped_through_aliases,
    {
        let mut env = shaped_env();
        env.add(
            "reexport",
            "from shape_extensions import Shaped as Sh, shape_vars as sv",
        );
        env
    },
    r#"
from typing import assert_type
import shape_extensions as se
from arrays import ndarray
from reexport import Sh, sv

@sv("N")
def f(x: Sh[ndarray, "[N]"]) -> se.Shaped[ndarray, "[N + 1]"]: ...

def check(x: ndarray[[3]]) -> None:
    assert_type(f(x), ndarray[[4]])
"#,
);

testcase!(
    test_shaped_goes_inside_qualifiers,
    shaped_env(),
    r#"
from typing import ClassVar, Final, NotRequired, TypedDict, assert_type
from arrays import ndarray
from shape_extensions import Shaped, shape_vars

@shape_vars("N")
class Record(TypedDict):
    a: NotRequired[Shaped[ndarray, "[N]"]]

@shape_vars("N")
class Holder:
    b: Final[Shaped[ndarray, "[N]"]]
    c: ClassVar[Shaped[ndarray, "[3]"]]

    def __init__(self, b: Shaped[ndarray, "[N]"]) -> None:
        self.b = b

def check(record: Record[4], holder: Holder[5]) -> None:
    assert_type(record["a"], ndarray[[4]])
    assert_type(holder.b, ndarray[[5]])
    assert_type(Holder.c, ndarray[[3]])
"#,
);

testcase!(
    test_shape_vars_do_not_reach_nested_classes,
    shaped_env(),
    r#"
from typing import assert_type
from arrays import ndarray
from shape_extensions import Shaped, shape_vars

class M: ...

@shape_vars("M")
def f(x: Shaped[ndarray, "[M]"], y: M) -> None:
    # Like native type parameters, declarations do not reach into nested classes,
    # so this `Shaped` is plain `Annotated`.
    class Inner:
        def g(self, z: Shaped[ndarray, "[M]"]) -> None:
            assert_type(z, ndarray)
"#,
);
