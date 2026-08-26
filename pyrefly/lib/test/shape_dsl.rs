/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_types::dimension::is_gradual_size;
use pyrefly_types::function::FunctionKind;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_level_dsl::MAX_HELPER_GRAPH_EDGES;
use pyrefly_types::type_level_dsl::MAX_HELPER_GRAPH_NODES;
use pyrefly_types::type_level_dsl::TypeShapeDslDomain;
use pyrefly_types::type_level_dsl::TypeShapeDslInputDomain;
use pyrefly_types::type_var::FlagDomain;
use pyrefly_types::type_var::FlagMember;
use pyrefly_types::type_var::Restriction;
use ruff_python_ast::name::Name;

use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTParams;
use crate::state::lsp::attribute_symbol_kind_from_type;
use crate::test::class_keywords::get_class_metadata;
use crate::test::util::TestEnv;
use crate::test::util::get_class;
use crate::test::util::testcase_for_macro;
use crate::testcase;
use crate::types::types::Type;

fn shaped_array_env() -> TestEnv {
    let path = PathBuf::from(
        std::env::var("SHAPE_EXTENSIONS_TEST_PATH")
            .expect("SHAPE_EXTENSIONS_TEST_PATH must be set"),
    );
    assert!(
        path.join("shape_extensions").is_dir(),
        "SHAPE_EXTENSIONS_TEST_PATH must point to a search path containing `shape_extensions`, got `{}`",
        path.display()
    );
    let path = path
        .to_str()
        .expect("SHAPE_EXTENSIONS_TEST_PATH must be valid UTF-8")
        .to_owned();
    TestEnv::new_with_site_package_paths(&[&path])
}

fn shaped_array_env_with_plain_torch() -> TestEnv {
    let mut env = shaped_array_env();
    env.add_with_path(
        "torch",
        "torch.pyi",
        r#"
class Tensor[*Shape]:
    def __getitem__(self, idx: int) -> Tensor[*Shape]: ...
"#,
    );
    env
}

fn shaped_array_env_with_shaped_torch() -> TestEnv {
    let mut env = shaped_array_env();
    env.add_with_path(
        "torch",
        "torch.pyi",
        r#"
from shape_extensions import Elements, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Tensor[Shape: IntTuple]: ...
"#,
    );
    env
}

fn add_jaxtyping(env: &mut TestEnv) {
    env.add_with_path(
        "jaxtyping",
        "jaxtyping.pyi",
        r#"
from typing import (
    Annotated as BFloat16,
    Annotated as Bool,
    Annotated as Complex,
    Annotated as Complex128,
    Annotated as Complex64,
    Annotated as Float,
    Annotated as Float16,
    Annotated as Float32,
    Annotated as Float64,
    Annotated as Inexact,
    Annotated as Int,
    Annotated as Int16,
    Annotated as Int32,
    Annotated as Int64,
    Annotated as Int8,
    Annotated as Integer,
    Annotated as Key,
    Annotated as Num,
    Annotated as Real,
    Annotated as Shaped,
    Annotated as UInt,
    Annotated as UInt16,
    Annotated as UInt32,
    Annotated as UInt64,
    Annotated as UInt8,
)
"#,
    );
}

fn plain_torch_and_jaxtyping_env() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "torch",
        "torch.pyi",
        r#"
class Tensor[*Shape]:
    def __getitem__(self, idx: int) -> Tensor[*Shape]: ...
"#,
    );
    add_jaxtyping(&mut env);
    env
}

fn shaped_array_env_with_plain_torch_and_jaxtyping() -> TestEnv {
    let mut env = shaped_array_env_with_plain_torch();
    add_jaxtyping(&mut env);
    env
}

fn reexporting_shape_extensions_env() -> TestEnv {
    let mut env = shaped_array_env_with_shaped_torch();
    env.add(
        "reexport",
        r#"
from shape_extensions import *
"#,
    );
    env
}

fn shaped_array_env_with_shaped_torch_and_jaxtyping() -> TestEnv {
    let mut env = shaped_array_env_with_shaped_torch();
    add_jaxtyping(&mut env);
    env
}

fn shaped_array_env_with_numpy() -> TestEnv {
    let mut env = shaped_array_env();
    env.add_with_path(
        "numpy",
        "numpy/__init__.pyi",
        r#"
from shape_extensions import uses_shape_dsl
from shape_extensions import shaped_array
from shape_extensions import IntTuple
from shape_extensions.dsl import ShapedArray, shape_dsl_function
from typing import Any

type AnyShape = tuple[Any, ...]

@shape_dsl_function
def add_leading_axis_ir(x: ShapedArray) -> ShapedArray:
    return ShapedArray(shape=[1] + x.shape)

@shaped_array(shape="Shape")
class ndarray[Shape: IntTuple, DType]:
    shape: Shape
    def copy(self) -> ndarray[Shape, DType]: ...
    def item(self) -> DType: ...

@uses_shape_dsl(add_leading_axis_ir)
def add_leading_axis[Shape: IntTuple, DType](x: ndarray[Shape, DType]) -> ndarray[Shape, DType]: ...

@shaped_array(shape="Shape")
class tcarray[Shape: IntTuple = AnyShape, DType = int]:
    shape: Shape
    def dtype(self) -> DType: ...
    @uses_shape_dsl(add_leading_axis_ir)
    def add_leading_axis(self) -> tcarray[Shape, DType]: ...

@uses_shape_dsl(add_leading_axis_ir)
def tc_add_leading_axis[Shape: IntTuple, DType](x: tcarray[Shape, DType]) -> tcarray[Shape, DType]: ...

def tc_identity[Shape: IntTuple, DType](x: tcarray[Shape, DType]) -> tcarray[Shape, DType]: ...
"#,
    );
    env
}

fn shape_dsl_base_env() -> TestEnv {
    shaped_array_env()
}

fn shape_dsl_tensor_env() -> TestEnv {
    let mut env = shape_dsl_base_env();
    env.add_with_path(
        "torch",
        "torch.pyi",
        r#"
from shape_extensions import Elements, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Tensor[Shape: IntTuple]:
    shape: Shape
"#,
    );
    env
}

testcase!(
    test_flag_int_accepts_shape_int_capture,
    shaped_array_env(),
    r#"
from shape_extensions import Flag, Int, IntVar
from typing import assert_type

def capture[K: Flag[int]](value: K) -> K: ...
def capture_bool[K: Flag[bool]](value: K) -> K: ...
def capture_str[K: Flag[str]](value: K) -> K: ...

def test[N: IntVar](symbolic: Int[N], literal: Int[3], broad: Int) -> None:
    assert_type(capture(symbolic), Int[N])
    assert_type(capture(literal), Int[3])
    assert_type(capture(broad), Int)
    capture_bool(symbolic)  # E: is not a valid `Flag[bool]` value
    capture_str(symbolic)  # E: is not a valid `Flag[str]` value
"#,
);

testcase!(
    test_type_shape_dsl_dimension_equality,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, IntTuple, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def require_equal(left: IntTuple, right: IntTuple) -> IntTuple:
    if left[0] == right[0]:
        return left
    return dsl.Invalid("dimensions differ")

@type_shape_dsl_function
def select_not_equal(left: IntTuple, right: IntTuple) -> IntTuple:
    if left[0] != right[0]:
        return dsl.IntTuple(())
    return left

@type_shape_dsl_function
def require_equal_local(left: IntTuple, right: IntTuple) -> IntTuple:
    left_item = left[0]
    right_item = right[0]
    if left_item != right_item:
        return dsl.Invalid("local dimensions differ")
    return left

@type_shape_dsl_function
def require_equal_negated(left: IntTuple, right: IntTuple) -> IntTuple:
    if not (left[0] == right[0]):
        return dsl.Invalid("negated dimensions differ")
    return left

@type_shape_dsl_function
def reflexive_local(shape: IntTuple) -> IntTuple:
    item = shape[0]
    if item == item:
        return dsl.IntTuple(())
    return dsl.Invalid("dimension is not reflexive")

@type_shape_dsl_function
def irreflexive_local(shape: IntTuple) -> IntTuple:
    item = shape[0]
    if item != item:
        return dsl.Invalid("dimension is irreflexive")
    return dsl.IntTuple(())

@type_shape_dsl_function
def literal_left_equal(right: IntTuple) -> IntTuple:
    if 2 == right[0]:
        return right
    return dsl.Invalid("right dimension differs")

@type_shape_dsl_function
def literal_right_equal(left: IntTuple) -> IntTuple:
    if left[0] == 2:
        return left
    return dsl.Invalid("left dimension differs")

@type_shape_dsl_function
def conditional_equal(shape: IntTuple, choose_first: bool) -> IntTuple:
    if (shape[0] if choose_first else shape[1]) == shape[0]:
        return shape
    return dsl.IntTuple(())

@type_shape_dsl_function
def compare_out_of_bounds(left: IntTuple, right: IntTuple) -> IntTuple:
    if left[1] == right[0]:
        return left
    return right

@type_shape_dsl_function
def reflexive_out_of_bounds(shape: IntTuple) -> IntTuple:
    if shape[1] == shape[1]:
        return shape
    return shape

def apply_equal[Left: IntTuple, Right: IntTuple](
    left: Tensor[Left], right: Tensor[Right],
) -> Tensor[require_equal(Left, Right)]: ...

def apply_not_equal[Left: IntTuple, Right: IntTuple](
    left: Tensor[Left], right: Tensor[Right],
) -> Tensor[select_not_equal(Left, Right)]: ...

def apply_equal_local[Left: IntTuple, Right: IntTuple](
    left: Tensor[Left], right: Tensor[Right],
) -> Tensor[require_equal_local(Left, Right)]: ...

def apply_equal_negated[Left: IntTuple, Right: IntTuple](
    left: Tensor[Left], right: Tensor[Right],
) -> Tensor[require_equal_negated(Left, Right)]: ...

def apply_reflexive[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[reflexive_local(Shape)]: ...
def apply_irreflexive[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[irreflexive_local(Shape)]: ...
def apply_literal_left[Right: IntTuple](right: Tensor[Right]) -> Tensor[literal_left_equal(Right)]: ...
def apply_literal_right[Left: IntTuple](left: Tensor[Left]) -> Tensor[literal_right_equal(Left)]: ...
def apply_conditional[Shape: IntTuple, ChooseFirst: Flag[bool]](
    shape: Tensor[Shape], choose_first: ChooseFirst,
) -> Tensor[conditional_equal(Shape, ChooseFirst)]: ...

def apply_out_of_bounds[Left: IntTuple, Right: IntTuple](
    left: Tensor[Left], right: Tensor[Right],
) -> Tensor[compare_out_of_bounds(Left, Right)]: ...
def apply_reflexive_out_of_bounds[Shape: IntTuple](
    x: Tensor[Shape],
) -> Tensor[reflexive_out_of_bounds(Shape)]: ...

def test(
    two: Tensor[[2]],
    another_two: Tensor[[2]],
    three: Tensor[[3]],
    pair: Tensor[[2, 3]],
    fixed_gradual: Tensor[[int]],
    gradual: Tensor[IntTuple],
) -> None:
    assert_type(apply_equal(two, another_two), Tensor[[2]])
    apply_equal(two, three)  # E: Cannot evaluate type-level shape DSL call: dimensions differ
    assert_type(apply_not_equal(two, another_two), Tensor[[2]])
    assert_type(apply_not_equal(two, three), Tensor[[]])
    apply_equal_local(two, three)  # E: Cannot evaluate type-level shape DSL call: local dimensions differ
    apply_equal_negated(two, three)  # E: Cannot evaluate type-level shape DSL call: negated dimensions differ
    assert_type(apply_reflexive(fixed_gradual), Tensor[[]])
    assert_type(apply_irreflexive(fixed_gradual), Tensor[[]])
    assert_type(apply_reflexive(gradual), Tensor[[]])
    assert_type(apply_irreflexive(gradual), Tensor[[]])
    assert_type(apply_literal_left(two), Tensor[[2]])
    apply_literal_left(three)  # E: Cannot evaluate type-level shape DSL call: right dimension differs
    assert_type(apply_literal_right(two), Tensor[[2]])
    apply_literal_right(three)  # E: Cannot evaluate type-level shape DSL call: left dimension differs
    assert_type(apply_conditional(pair, True), Tensor[[2, 3]])
    assert_type(apply_conditional(pair, False), Tensor[[]])
    reveal_type(apply_equal(fixed_gradual, fixed_gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_equal(gradual, gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_not_equal(fixed_gradual, fixed_gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_not_equal(gradual, gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    apply_out_of_bounds(two, another_two)  # E: Cannot evaluate type-level shape DSL call: IntTuple index out of bounds
    apply_reflexive_out_of_bounds(two)  # E: Cannot evaluate type-level shape DSL call: IntTuple index out of bounds

def test_symbolic[N: IntVar, M: IntVar](
    same_left: Tensor[[N]], same_right: Tensor[[N]], other: Tensor[[M]],
) -> None:
    assert_type(apply_equal(same_left, same_right), Tensor[[N]])
    reveal_type(apply_equal(same_left, other))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_not_equal(same_left, same_right), Tensor[[N]])
    reveal_type(apply_not_equal(same_left, other))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_equal_local(same_left, same_right), Tensor[[N]])
    reveal_type(apply_equal_local(same_left, other))  # E: revealed type: Tensor[tuple[Unknown, ...]]
"#,
);

// Shape-derived tuples mix literals with symbolic `Int[N]` dimensions, so a tuple domain has
// to admit both rather than only integer classes.
testcase!(
    test_flag_tuple_accepts_symbolic_shape_ints,
    shaped_array_env(),
    r#"
from shape_extensions import Flag, Int, IntVar
from typing import Literal, assert_type

def capture_axes[A: Flag[tuple[int, ...]]](axes: A) -> A: ...

def test[N: IntVar](symbolic: Int[N], literal: Int[3], broad: Int) -> None:
    assert_type(capture_axes((symbolic, 3)), tuple[Int[N], Literal[3]])
    assert_type(capture_axes((literal, broad)), tuple[Int[3], Int])
    capture_axes((symbolic, "x"))  # E: is not a valid `Flag[tuple[int, ...]]` value
"#,
);

fn type_shape_dsl_gradual_env() -> TestEnv {
    let mut env = shape_dsl_tensor_env();
    env.add(
        "gradual_reexport",
        r#"
from shape_extensions.dsl import Int as ReexportedInt
"#,
    );
    env
}

fn type_shape_dsl_predicate_env() -> TestEnv {
    let mut env = shape_dsl_tensor_env();
    env.add(
        "predicate_reexport",
        "from shape_extensions.dsl import is_concrete_int as predicate\n",
    );
    env.add(
        "predicate_lookalike",
        "def is_concrete_int(value: object) -> bool: ...\n",
    );
    env
}

fn type_shape_dsl_import_env() -> TestEnv {
    let mut env = shape_dsl_tensor_env();
    env.add(
        "identities",
        r#"
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def int_identity(x: Int) -> Int:
    return x

@type_shape_dsl_function
def shape_identity(x: IntTuple) -> IntTuple:
    return x

@type_shape_dsl_function
def select_shape(dim: Int, shape: IntTuple) -> IntTuple:
    return shape

@type_shape_dsl_function
def diag_extent(n: Int, k: int) -> Int:
    if k < 0:
        return n - k
    return n + k
"#,
    );
    env
}

fn type_shape_dsl_broadcast_env() -> TestEnv {
    let mut env = shape_dsl_tensor_env();
    env.add(
        "broadcast_reexport",
        "from shape_extensions import broadcast as reexported_broadcast\n",
    );
    env.add(
        "broadcast_lookalike",
        r#"
from shape_extensions import IntTuple

def broadcast(left: IntTuple, right: IntTuple) -> IntTuple:
    return left
"#,
    );
    env
}

#[test]
fn test_type_shape_dsl_function_declarations() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def int_identity(x: Int) -> Int:
    return x

@type_shape_dsl_function
def shape_identity(shape: IntTuple) -> IntTuple:
    return shape

@type_shape_dsl_function
def select_int(shape: IntTuple, dim: Int) -> Int:
    return dim

@type_shape_dsl_function
def select_shape(dim: Int, shape: IntTuple) -> IntTuple:
    return shape

@type_shape_dsl_function
def diag_extent(n: Int, k: int) -> Int:
    if k < 0:
        return n - k
    return n + k
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let solutions = state
        .transaction()
        .get_solutions(&main)
        .expect("module should solve");
    for (name, expected_parameters, expected_result) in [
        (
            "int_identity",
            vec![TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)],
            TypeShapeDslDomain::Int,
        ),
        (
            "shape_identity",
            vec![TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)],
            TypeShapeDslDomain::IntTuple,
        ),
        (
            "select_int",
            vec![
                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple),
                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int),
            ],
            TypeShapeDslDomain::Int,
        ),
        (
            "select_shape",
            vec![
                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int),
                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple),
            ],
            TypeShapeDslDomain::IntTuple,
        ),
        (
            "diag_extent",
            vec![
                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int),
                TypeShapeDslInputDomain::Flag(FlagDomain::of(FlagMember::Int)),
            ],
            TypeShapeDslDomain::Int,
        ),
    ] {
        let ty = solutions.get(&KeyExport(Name::new(name)));
        assert!(
            matches!(ty, Type::Function(function)
                if matches!(&function.metadata.kind,
                    FunctionKind::TypeShapeDsl(_, function)
                        if function.parameter_domains() == expected_parameters
                            && function.result_domain() == expected_result)),
            "expected `{name}` to retain type-level DSL metadata, got `{ty}`"
        );
        assert_eq!(attribute_symbol_kind_from_type(ty), SymbolKind::Function);
    }
}

#[test]
fn test_invalid_type_shape_dsl_function_recovers_as_def() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import Int, type_shape_dsl_function
from shape_extensions.dsl import shape_dsl_function

@type_shape_dsl_function
def invalid(x: Int) -> Int:
    return abs(x)

@type_shape_dsl_function
def invalid_domain(x: str) -> str:
    return x

@type_shape_dsl_function
def duplicate(x: Int, x: Int) -> Int:
    return x

@shape_dsl_function
@type_shape_dsl_function
def conflicting(x: Int) -> Int:
    return x
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let solutions = state
        .transaction()
        .get_solutions(&main)
        .expect("module should solve");
    for name in ["invalid", "invalid_domain", "duplicate", "conflicting"] {
        let ty = solutions.get(&KeyExport(Name::new(name)));
        assert!(
            matches!(ty, Type::Function(function)
                if matches!(&function.metadata.kind, FunctionKind::Def(_))),
            "expected invalid DSL declaration `{name}` to recover as an ordinary function, got `{ty}`"
        );
    }
}

testcase!(
    test_type_shape_dsl_function_invalid_syntax,
    shaped_array_env(),
    r#"
from shape_extensions import Int, type_shape_dsl_function
from shape_extensions.dsl import shape_dsl_function

@type_shape_dsl_function
async def asynchronous(x: Int) -> Int:  # E: @type_shape_dsl_function does not support async functions
    return x

@type_shape_dsl_function
def generic[T](x: Int) -> Int:  # E: @type_shape_dsl_function does not support type parameters
    return x

@type_shape_dsl_function
def zero_parameters() -> Int:  # E: @type_shape_dsl_function supports only ordinary positional parameters and requires at least one
    return x  # E: Could not find name `x`

@type_shape_dsl_function
def default(x: Int = 1) -> Int:  # E: @type_shape_dsl_function does not support parameter defaults
    return x

@type_shape_dsl_function
def positional_only(x: Int, /, y: Int) -> Int:  # E: @type_shape_dsl_function supports only ordinary positional parameters and requires at least one
    return y

@type_shape_dsl_function
def keyword_only(x: Int, *, y: Int) -> Int:  # E: @type_shape_dsl_function supports only ordinary positional parameters and requires at least one
    return x

@type_shape_dsl_function
def variadic(x: Int, *args: Int) -> Int:  # E: @type_shape_dsl_function supports only ordinary positional parameters and requires at least one
    return x

@type_shape_dsl_function
def keyword_variadic(x: Int, **kwargs: Int) -> Int:  # E: @type_shape_dsl_function supports only ordinary positional parameters and requires at least one
    return x

@type_shape_dsl_function
def duplicate(x: Int, x: Int) -> Int:  # E: @type_shape_dsl_function parameter names must be unique  # E: Duplicate parameter "x"
    return x

@type_shape_dsl_function
def expression(x: Int) -> Int:
    return x + 1

@type_shape_dsl_function
def wrong_name(x: Int) -> Int:
    return other  # E: @type_shape_dsl_function returned name must match a parameter name  # E: Could not find name `other`

def outer() -> None:
    @type_shape_dsl_function
    def nested(x: Int) -> Int:  # E: @type_shape_dsl_function must decorate a top-level function
        return x

@shape_dsl_function
@type_shape_dsl_function
def conflicting(x: Int) -> Int:  # E: `@shape_dsl_function` and `@type_shape_dsl_function` cannot be combined
    return x
"#,
);

testcase!(
    test_type_shape_dsl_function_invalid_annotations,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def missing_parameter(x) -> Int:  # E: parameter `x` must be annotated as `Int`, `IntTuple`, or a supported Flag value type
    return x

@type_shape_dsl_function
def missing_return(x: Int):  # E: `@type_shape_dsl_function` return must be annotated as `Int` or `IntTuple`
    return x

@type_shape_dsl_function
def wrong_type(x: str) -> str:  # E: Flag values are input-only
    return x

@type_shape_dsl_function
def cross_domain(x: Int) -> IntTuple:
    return x  # E: `@type_shape_dsl_function` return annotation must match returned parameter `x`  # E: Returned type `Int[int]` is not assignable to declared return type `IntTuple`

@type_shape_dsl_function
def missing_second(x: Int, y) -> Int:  # E: parameter `y` must be annotated as `Int`, `IntTuple`, or a supported Flag value type
    return x

@type_shape_dsl_function
def valid_unused_flag(x: str, y: Int) -> Int:
    return y

@type_shape_dsl_function
def mixed_unused_domain(shape: IntTuple, dim: Int) -> Int:
    return dim
"#,
);

fn assert_shaped_array_shape(shape: &Quantified, name: &str, kind: QuantifiedKind) {
    assert_eq!(shape.name().as_str(), name);
    assert_eq!(shape.kind, kind);
}

#[test]
fn test_shaped_array_imports_are_metadata() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
import shape_extensions as se
from shape_extensions import IntTuple, shaped_array
from shape_extensions import shaped_array as shaped_array_alias

@shaped_array(shape="Shape")
class ImportedArray[Shape: IntTuple]: ...

@shaped_array_alias(shape="Shape")
class ImportAliasArray[Shape: IntTuple]: ...

@se.shaped_array(shape="Shape")
class ModuleAliasArray[DType, Shape: IntTuple]: ...

class PlainArray[*Shape]: ...
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    for class_name in ["ImportedArray", "ImportAliasArray", "ModuleAliasArray"] {
        let metadata = get_class_metadata(class_name, &main, &state);
        let shape = metadata
            .shaped_array_shape()
            .expect("shaped array shape should be present");
        assert_shaped_array_shape(shape, "Shape", QuantifiedKind::TypeVar);
    }
    assert!(!get_class_metadata("PlainArray", &main, &state).is_shaped_array());
}

#[test]
fn test_shaped_array_typevar_shape_is_metadata() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class TupleCarrierArray[Shape, DType]: ...
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let metadata = get_class_metadata("TupleCarrierArray", &main, &state);
    let shape = metadata
        .shaped_array_shape()
        .expect("shaped array shape should be present");
    assert_shaped_array_shape(shape, "Shape", QuantifiedKind::TypeVar);
}

#[test]
fn test_shaped_array_class_targ_shape_is_first_class_inttuple() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

x: Array[[2, 3], int]
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let solutions = state.transaction().get_solutions(&main).unwrap();
    match solutions.get(&KeyExport(Name::new("x"))) {
        Type::ShapedArray(array) => {
            let shape_arg = &array.base_class.targs().as_slice()[0];
            assert!(
                matches!(shape_arg, Type::IntTuple(_)),
                "expected normalized shape argument to be `IntTuple`, got `{shape_arg}`"
            );
        }
        ty => panic!("expected `x` to solve to a shaped array, got `{ty}`"),
    }
}

#[test]
fn test_legacy_intvar_binding_has_intvar_kind() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import IntVar

N = IntVar("N")
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let solutions = state.transaction().get_solutions(&main).unwrap();
    match solutions.get(&KeyExport(Name::new("N"))) {
        Type::TypeVar(tv) => assert_eq!(tv.kind(), QuantifiedKind::IntVar),
        ty => panic!("expected `N` to solve to a raw IntVar, got `{ty}`"),
    }
}

#[test]
fn test_legacy_intvar_generic_class_tparam_has_intvar_kind() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import IntVar
from typing import Generic

N = IntVar("N")

class Box(Generic[N]): ...
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let cls = get_class("Box", &main, &state);
    let solutions = state.transaction().get_solutions(&main).unwrap();
    let tparams = solutions.get(&KeyTParams(cls.index()));
    assert_eq!(tparams.len(), 1);
    let param = tparams
        .iter()
        .next()
        .expect("Box should have one type parameter");
    assert_eq!(param.name().as_str(), "N");
    assert_eq!(param.kind(), QuantifiedKind::IntVar);
}

#[test]
fn test_jaxtyping_dim_cache_distinguishes_kinds() {
    // The per-module jaxtyping dim cache must key on `QuantifiedKind`, not just the
    // name. The same dimension name legitimately arrives as a scalar dim (`TypeVar`)
    // and as a variadic `*name` (`TypeVarTuple`); if the cache dropped the kind,
    // whichever kind was requested first would be cached and returned for both,
    // silently producing a quantified of the wrong kind.
    let mut env = TestEnv::new();
    env.add("main", "");
    let (state, handle) = env.to_state();
    let main = handle("main");
    let (type_var, type_var_tuple) = state
        .transaction()
        .ad_hoc_solve(&main, "test_jaxtyping_dim_cache", |solver| {
            let name = Name::new("batch");
            let type_var =
                solver.get_or_create_jaxtyping_dim(name.clone(), QuantifiedKind::TypeVar);
            let type_var_tuple =
                solver.get_or_create_jaxtyping_dim(name, QuantifiedKind::TypeVarTuple);
            (type_var, type_var_tuple)
        })
        .expect("ad_hoc_solve should succeed for the `main` module");
    assert_eq!(type_var.name().as_str(), "batch");
    assert_eq!(type_var.kind, QuantifiedKind::TypeVar);
    assert_eq!(type_var_tuple.name().as_str(), "batch");
    assert_eq!(type_var_tuple.kind, QuantifiedKind::TypeVarTuple);
}

#[test]
fn test_non_shape_intvar_is_not_a_kind_marker() {
    let mut env = shaped_array_env();
    env.add(
        "other",
        r#"
class IntVar: ...
"#,
    );
    env.add(
        "main",
        r#"
from other import IntVar
from typing import Generic

class Box[N: IntVar](Generic[N]): ...
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let cls = get_class("Box", &main, &state);
    let solutions = state.transaction().get_solutions(&main).unwrap();
    let tparams = solutions.get(&KeyTParams(cls.index()));
    let param = tparams
        .iter()
        .next()
        .expect("Box should have one type parameter");
    assert_eq!(param.name().as_str(), "N");
    assert_eq!(param.kind(), QuantifiedKind::TypeVar);
    assert!(matches!(
        param.restriction(),
        Restriction::Bound(Type::ClassType(cls)) if cls.has_qname("other", "IntVar")
    ));
}

testcase!(
    test_shaped_array_invalid_metadata,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array
from typing import Any, Generic, TypeVarTuple

kwargs: Any = {}

@shaped_array  # E: `@shaped_array` requires a `shape` keyword argument
class BareDecorator[Shape]: ...

@shaped_array()  # E: `@shaped_array` requires a `shape` keyword argument  # E: Missing argument `shape` in function `shape_extensions.shaped_array`
class MissingShape[Shape]: ...

@shaped_array("Shape")  # E: `@shaped_array` expects `shape` as a keyword argument  # E: Expected argument `shape` to be passed by name in function `shape_extensions.shaped_array`
class PositionalShape[Shape]: ...

@shaped_array(dtype="Shape")  # E: Unexpected keyword argument `dtype` for `@shaped_array`; expected `shape`  # E: Missing argument `shape` in function `shape_extensions.shaped_array`  # E: Unexpected keyword argument `dtype` in function `shape_extensions.shaped_array`
class WrongShapeKeyword[Shape]: ...

@shaped_array(shape="Shape", **kwargs)  # E: Unpacking is not supported in `@shaped_array`
class KwargsShape[Shape]: ...

@shaped_array(shape="Shape", shape="Shape")  # E: Parse error: Duplicate keyword argument "shape"  # E: Multiple values for argument `shape` in function `shape_extensions.shaped_array`
class DuplicateShapeKeyword[Shape]: ...

@shaped_array(shape=123)  # E: `@shaped_array` `shape` argument must be a string literal  # E: Argument `Literal[123]` is not assignable to parameter `shape` with type `str` in function `shape_extensions.shaped_array`
class NonStringShape[Shape]: ...

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a scoped (PEP-695-style) type parameter of class `NoTypeParams`
class NoTypeParams: ...

Shape = TypeVarTuple("Shape")

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a scoped (PEP-695-style) type parameter of class `LegacyGeneric`
class LegacyGeneric(Generic[*Shape]): ...

@shaped_array(shape="Shape")
@shaped_array(shape="Shape")  # E: Duplicate `@shaped_array` decorator
class DuplicateDecorator[Shape]: ...

@shaped_array  # E: `@shaped_array` requires a `shape` keyword argument
@shaped_array(shape="Shape")  # E: Duplicate `@shaped_array` decorator
class DuplicateDecoratorAfterInvalid[Shape]: ...

@shaped_array(shape="Missing")  # E: Shape parameter `Missing` is not a type parameter of class `ShapeNotFound`
class ShapeNotFound[Shape]: ...

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a `TypeVar` or `IntVar`, got `TypeVarTuple`
class TypeVarTupleShape[*Shape]: ...

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a `TypeVar` or `IntVar`, got `ParamSpec`
class ShapeIsParamSpec[**Shape, DType]: ...
"#,
);

testcase!(
    test_shaped_array_compact_list_carrier,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]:
    def dtype(self) -> DType: ...

@shaped_array(shape="Shape")
class DTypeFirstArray[DType, Shape]: ...

def f(
    compact: Array[[2, 3], int],
    pep484: Array[tuple[Literal[2], Literal[3]], int],
    scalar: Array[[], int],
    dtype_first: DTypeFirstArray[int, [2, 3]],
) -> None:
    # Compact and PEP-484 forms reveal identically.
    reveal_type(compact)  # E: revealed type: Array[[2, 3], int]
    reveal_type(pep484)  # E: revealed type: Array[[2, 3], int]
    reveal_type(scalar)  # E: revealed type: Array[[], int]
    reveal_type(dtype_first)  # E: revealed type: DTypeFirstArray[int, [2, 3]]
    reveal_type(compact.dtype())  # E: revealed type: int
"#,
);

testcase!(
    test_shaped_array_pep484_tuple_carrier_canonicalization,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(
    compact: Array[[2, 3], int],
    pep484: Array[tuple[Literal[2], Literal[3]], int],
    compact_scalar: Array[[], int],
    pep484_scalar: Array[tuple[()], int],
) -> None:
    # The compact and PEP-484 carriers canonicalize to the same shape.
    reveal_type(compact)  # E: revealed type: Array[[2, 3], int]
    reveal_type(pep484)  # E: revealed type: Array[[2, 3], int]
    reveal_type(compact_scalar)  # E: revealed type: Array[[], int]
    reveal_type(pep484_scalar)  # E: revealed type: Array[[], int]

    # Closed concrete shapes are mutually assignable in both directions.
    p: Array[tuple[Literal[2], Literal[3]], int] = compact
    c: Array[[2, 3], int] = pep484
    ps: Array[tuple[()], int] = compact_scalar
    cs: Array[[], int] = pep484_scalar

    wrong_rank2: Array[[2, 4], int] = pep484  # E: `Array[[2, 3], int]` is not assignable to `Array[[2, 4], int]`
    wrong_rank0: Array[[1], int] = pep484_scalar  # E: `Array[[], int]` is not assignable to `Array[[1], int]`
"#,
);

testcase!(
    test_shaped_array_inttuple_bound,
    shaped_array_env(),
    r#"
from typing import Any, Literal, reveal_type
from shape_extensions import Int, Elements, IntTuple, IntVar, assert_shape, shaped_array

type _Shape = IntTuple
type _AnyShape = tuple[Any, ...]

@shaped_array(shape="Shape")
class Array[Shape: _Shape = _AnyShape, DType = Any]:
    shape: Shape

def f[N: IntVar](
    compact: Array[[2, 3], int],
    pep484: Array[tuple[Literal[2], Literal[3]], int],
    int_tuple: Array[IntTuple[2, 3], int],
    mixed_int_tuple: Array[IntTuple[2, 3, N], int],
    bare_dim: Int[N],
    bare_list: Array[[N], int],
    bare_int_tuple: Array[IntTuple[N], int],
    any_dim: Array[[Any], int],
    carrier: IntTuple[2, 3],
    mixed_carrier: IntTuple[2, 3, N],
    unbounded: IntTuple,
) -> None:
    reveal_type(compact)  # E: revealed type: Array[[2, 3], int]
    reveal_type(pep484)  # E: revealed type: Array[[2, 3], int]
    reveal_type(int_tuple)  # E: revealed type: Array[[2, 3], int]
    reveal_type(mixed_int_tuple)  # E: revealed type: Array[[2, 3, N], int]
    reveal_type(bare_dim)  # E: revealed type: Int[N]
    reveal_type(bare_list)  # E: revealed type: Array[[N], int]
    reveal_type(bare_int_tuple)  # E: revealed type: Array[[N], int]
    reveal_type(any_dim)  # E: revealed type: Array[[int], int]
    reveal_type(carrier)  # E: revealed type: IntTuple[2, 3]
    reveal_type(mixed_carrier)  # E: revealed type: IntTuple[2, 3, N]
    reveal_type(unbounded)  # E: revealed type: IntTuple
    p: Array[tuple[Literal[2], Literal[3]], int] = compact
    c: Array[[2, 3], int] = pep484
    st: Array[IntTuple[2, 3], int] = compact
    mst: Array[tuple[Literal[2], Literal[3], Int[N]], int] = mixed_int_tuple

def append_dim[S: IntTuple, OUT: IntVar](
    explicit: Array[IntTuple[*Elements[S], OUT], int],
    compact: Array[[*Elements[S], OUT], int],
) -> Array[[*Elements[S], OUT], int]:
    reveal_type(explicit)  # E: revealed type: Array[[*Elements[S], OUT], int]
    reveal_type(compact)  # E: revealed type: Array[[*Elements[S], OUT], int]
    return explicit

def prepend_and_append[S: IntTuple, OUT: IntVar](
    source: Array[S, int],
    result: Array[[1, *Elements[S], OUT], int],
) -> Array[[1, *Elements[S], OUT], int]:
    return result

def concrete_unpack[M: IntVar, N: IntVar](
    source: Array[[4, M], int],
    result: Array[[1, 4, M, N], int],
) -> None:
    reveal_type(prepend_and_append(source, result))  # E: revealed type: Array[[1, 4, M, N], int]

def nested_unpack[S0: IntTuple, M: IntVar, N: IntVar](
    source: Array[[4, *Elements[S0], M], int],
    result: Array[[1, 4, *Elements[S0], M, N], int],
) -> None:
    reveal_type(prepend_and_append(source, result))  # E: revealed type: Array[[1, 4, *Elements[S0], M, N], int]

def gradual_middle(
    result: Array[[1, *Elements[IntTuple], 3], int],
) -> None:
    reveal_type(result)  # E: revealed type: Array[[1, *tuple[int, ...], 3], int]

def concrete_elements_middle(
    result: Array[[1, *Elements[IntTuple[2, 3]], 4], int],
) -> None:
    reveal_type(result)  # E: revealed type: Array[[1, 2, 3, 4], int]

def assert_single_dim(x: Array[[3], int]) -> None:
    reveal_type(assert_shape(x, (3,)))  # E: revealed type: Array[[3], int]
"#,
);

testcase!(
    test_intvar_rejects_non_int_specialization_with_int_recovery,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import Int, IntVar

class Box[N: IntVar]:
    dim: Int[N]

type Dim[N: IntVar] = Int[N]

def explicit_class(bad: Box[str]) -> None:  # E: Tensor shape dimensions must be integer literals or type variables
    reveal_type(bad.dim)  # E: revealed type: Int[int]

def explicit_class_non_shape_arg(bad: Box[list[int]]) -> None:  # E: Tensor shape dimensions must be positive integer literals, string literals, type variables, or expressions
    reveal_type(bad.dim)  # E: revealed type: Int[int]

def explicit_alias(x: Dim[str]) -> None:  # E: Tensor shape dimensions must be integer literals or type variables
    reveal_type(x)  # E: revealed type: Int[int]
"#,
);

testcase!(
    test_intvar_bad_call_bound_recovers_to_int_gradual,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import Int, IntVar

def takes_dim[N: IntVar](x: Int[N]) -> Int[N]:
    return x

def bad_call(x: str) -> None:
    y = takes_dim(x)  # E: Argument `str` is not assignable to parameter `x`
    reveal_type(y)  # E: revealed type: Int[int]

def bad_upper_bound() -> None:
    y: str = takes_dim(3)  # E: `Int[3]` is not assignable to `str`
    reveal_type(y)  # E: revealed type: str
"#,
);

testcase!(
    test_ordinary_typevar_still_solves_to_int,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import Int, IntVar

def identity[T](x: T) -> T:
    return x

def f[N: IntVar](x: Int[N]) -> None:
    reveal_type(identity(x))  # E: revealed type: Int[N]
"#,
);

testcase!(
    test_intvar_inference_chains_without_losing_kind,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import Int, IntVar

def identity[T](x: T) -> T:
    return x

def same_dim[N: IntVar](x: Int[N]) -> Int[N]:
    return x

def f[N: IntVar](x: Int[N], s: str) -> None:
    reveal_type(same_dim(same_dim(x)))  # E: revealed type: Int[N]
    reveal_type(same_dim(identity(x)))  # E: revealed type: Int[N]
    reveal_type(identity(same_dim(x)))  # E: revealed type: Int[N]
    same_dim(identity(s))  # E: Argument `str` is not assignable to parameter `x`
"#,
);

testcase!(
    test_intvar_inference_with_bounded_typevar_keeps_int_kind,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import Int, IntVar

def bounded_identity[T: object](x: T) -> T:
    return x

def same_dim[N: IntVar](x: Int[N]) -> Int[N]:
    return x

def f[N: IntVar](x: Int[N], s: str) -> None:
    reveal_type(same_dim(bounded_identity(x)))  # E: revealed type: Int[N]
    same_dim(bounded_identity(s))  # E: Argument `str` is not assignable to parameter `x`
"#,
);

testcase!(
    test_shaped_array_elements_tuple_carriers_rfc,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import Elements, IntTuple, IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def concrete_tuple_carrier(
    result: Array[[1, *Elements[tuple[Literal[2], Literal[3]]], 4], int],
) -> None:
    reveal_type(result)  # E: revealed type: Array[[1, 2, 3, 4], int]

def nested_concrete_tuple_carrier(
    result: Array[[1, *Elements[tuple[Literal[2], *tuple[Literal[3]], Literal[4]]], 5], int],
) -> None:
    reveal_type(result)  # E: revealed type: Array[[1, 2, 3, 4, 5], int]

def nested_unbounded_tuple_carrier(
    result: Array[[1, *Elements[tuple[Literal[2], *tuple[int, ...], Literal[4]]], 5], int],
) -> None:
    reveal_type(result)  # E: revealed type: Array[[1, 2, *tuple[int, ...], 4, 5], int]

def tuple_bound_carrier[S: tuple[int, ...], OUT: IntVar](
    result: Array[[*Elements[S], OUT], int],
) -> None:
    reveal_type(result)  # E: revealed type: Array[[*Elements[S], OUT], int]

def independent_tuple_bound_carriers[
    S: tuple[int, ...],
    Q: tuple[int, ...],
    M: IntVar,
    N: IntVar,
](
    left: Array[[*Elements[S], M], int],
    right: Array[[*Elements[Q], N], int],
) -> None:
    reveal_type(left)  # E: revealed type: Array[[*Elements[S], M], int]
    reveal_type(right)  # E: revealed type: Array[[*Elements[Q], N], int]

def inttuple_bound_still_works[S: IntTuple, OUT: IntVar](
    result: Array[[*Elements[S], OUT], int],
) -> None:
    reveal_type(result)  # E: revealed type: Array[[*Elements[S], OUT], int]
"#,
);

testcase!(
    test_shaped_array_unpacked_middle_solver_round_trip,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import Elements, Int, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

def identity[Shape: IntTuple](x: Array[Shape, int]) -> Array[Shape, int]:
    return x

def gradual_middle(
    x: Array[[1, *Elements[IntTuple], 4], int],
) -> None:
    reveal_type(identity(x))  # E: revealed type: Array[[1, *tuple[int, ...], 4], int]

def shapeful_unbounded_middle(
    x: Array[[1, *Elements[tuple[Int[5], ...]], 4], int],
) -> None:
    reveal_type(identity(x))  # E: revealed type: Array[[1, *tuple[Int[5], ...], 4], int]
"#,
);

testcase!(
    test_shaped_array_inttuple_shape_arg_return_reprojection,
    shaped_array_env(),
    r#"
from shape_extensions import IntTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]:
    def clone(self) -> Array[Shape, DType]: ...

def f(x: Array[[2, 3], int]) -> None:
    y = x.clone()
    reveal_type(y)  # E: revealed type: Array[[2, 3], int]
    reveal_type(y[0])  # E: revealed type: Array[[3], int]
"#,
);

testcase!(
    test_type_level_dsl_broadcast_return_boundary,
    shaped_array_env_with_shaped_torch(),
    r#"
import shape_extensions
import shape_extensions as shapes
from shape_extensions import IntTuple, broadcast
from torch import Tensor
from typing import overload, reveal_type

class Foo[T]: ...
class Bar[T]: ...
class Baz[T]: ...
def ordinary(x: object) -> object: ...

def deeply_wrapped[S: IntTuple]() -> Foo[Bar[Baz[Bar[Foo[Tensor[broadcast(S, S)]]]]]]: ...
def invalid_call() -> Tensor[ordinary(IntTuple[2])]: ...  # E: Expected a type-level DSL function

def add_qualified[S0: IntTuple, S1: IntTuple](x: Tensor[S0], y: Tensor[S1]) -> Tensor[shape_extensions.broadcast(S0, S1)]: ...
def add_imported[S0: IntTuple, S1: IntTuple](x: Tensor[S0], y: Tensor[S1]) -> Tensor[broadcast(S0, S1)]: ...
def add_alias[S0: IntTuple, S1: IntTuple](x: Tensor[S0], y: Tensor[S1]) -> Tensor[shapes.broadcast(S0, S1)]: ...
def add_same[S: IntTuple](x: Tensor[S], y: Tensor[S]) -> Tensor[broadcast(S, S)]: ...
def add_nested[S0: IntTuple, S1: IntTuple, S2: IntTuple](
    x: Tensor[S0],
    y: Tensor[S1],
    z: Tensor[S2],
) -> Tensor[broadcast(broadcast(S0, S1), S2)]: ...
def add_repeated[S0: IntTuple, S1: IntTuple](
    x: Tensor[S0],
    y: Tensor[S1],
) -> Tensor[broadcast(broadcast(S0, S1), broadcast(S0, S1))]: ...

@overload
def add_overloaded(x: Tensor[[2, 3]], y: Tensor[[1, 3]]) -> Tensor[broadcast(IntTuple[2, 3], IntTuple[1, 3])]: ...
@overload
def add_overloaded(x: Tensor[[2, 3]], y: Tensor[[4, 3]]) -> Tensor[broadcast(IntTuple[2, 3], IntTuple[4, 3])]: ...
def add_overloaded(x: Tensor, y: Tensor) -> Tensor: ...

def add_expanded(
    args: tuple[Tensor[[2, 3]], Tensor[[1, 3]]]
    | tuple[Tensor[[2, 3]], Tensor[[4, 3]]],
) -> None:
    add_overloaded(*args)  # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[2] with dimension Int[4] at position 0

def bad_domain[S0: IntTuple](x: Tensor[S0]) -> Tensor[broadcast(int, S0)]: ...  # E: Expected an `IntTuple` argument to `broadcast`
def bad_arity[S0: IntTuple](x: Tensor[S0]) -> Tensor[broadcast(S0)]: ...  # E: Expected 2 arguments for `broadcast`, got 1
def bad_keyword[S0: IntTuple](x: Tensor[S0]) -> Tensor[broadcast(S0, right=S0)]: ...  # E: `broadcast` does not accept keyword arguments

def test_same[S: IntTuple](x: Tensor[S]) -> None:
    reveal_type(add_same(x, x))  # E: revealed type: Tensor[S]

def test(x: Tensor[[2, 3]], y: Tensor[[1, 3]], z: Tensor[[2, 1]], bad: Tensor[[4, 3]], unknown: Tensor[IntTuple]) -> None:
    reveal_type(add_qualified(x, y))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(add_imported(x, y))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(add_alias(x, y))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(add_nested(x, z, y))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(add_imported(x, unknown))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    add_imported(x, bad)  # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[2] with dimension Int[4] at position 0
    add_nested(x, bad, y)  # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[2] with dimension Int[4] at position 0
    add_repeated(x, bad)  # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[2] with dimension Int[4] at position 0
"#,
);

testcase!(
    test_type_shape_dsl_identity_calls,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, IntVar, broadcast, type_shape_dsl_function
from torch import Tensor
from typing import Annotated, Any, Literal, overload, reveal_type

@type_shape_dsl_function
def int_identity(x: Int) -> Int:
    return x

@type_shape_dsl_function
def shape_identity(x: IntTuple) -> IntTuple:
    return x

def keep_dim[N: IntVar](x: Tensor[[N]]) -> Tensor[[int_identity(N)]]: ...
def gradual_dim(x: Tensor[[int]]) -> Tensor[[int_identity(int)]]: ...
def any_dim(x: Tensor[[int]]) -> Tensor[[int_identity(Any)]]: ...
def keep_shape[S: IntTuple](x: Tensor[S]) -> Tensor[shape_identity(S)]: ...
def gradual_shape(x: Tensor[IntTuple]) -> Tensor[shape_identity(IntTuple)]: ...
def any_shape(x: Tensor[IntTuple]) -> Tensor[shape_identity(Any)]: ...
def compose[S0: IntTuple, S1: IntTuple](
    x: Tensor[S0],
    y: Tensor[S1],
) -> Tensor[shape_identity(broadcast(shape_identity(S0), S1))]: ...
def wrapped[S: IntTuple](x: Tensor[S]) -> tuple[Tensor[shape_identity(S)]]: ...
type Wrapped[T] = tuple[T]
def wrapped_alias[S: IntTuple](x: Tensor[S]) -> Wrapped[Tensor[shape_identity(S)]]: ...
def annotated[S: IntTuple](x: Tensor[S]) -> Annotated[Tensor[shape_identity(S)], "shape"]: ...

class DimBox[N: IntVar]: ...
def wrapped_dim[N: IntVar](x: Tensor[[N]]) -> DimBox[int_identity(N)]: ...
class ShapeBox[S: IntTuple]: ...
def wrapped_compact_shape[N: IntVar](x: Tensor[[N]]) -> ShapeBox[[int_identity(N)]]: ...
def wrapped_shape_call[N: IntVar](x: Tensor[[N]]) -> Tensor[shape_identity(IntTuple[int_identity(N)])]: ...
def wrapped_int_call[N: IntVar](x: Tensor[[N]]) -> Tensor[[int_identity(Int[int_identity(N)])]]: ...
def wrapped_broadcast_call[N: IntVar](x: Tensor[[N]]) -> Tensor[broadcast(IntTuple[int_identity(N)], IntTuple[1])]: ...
class ParamSpecBox[**P]: ...
def wrapped_paramspec_list() -> ParamSpecBox[[int, str]]: ...

@overload
def overloaded[S: IntTuple](x: Tensor[S]) -> Tensor[shape_identity(S)]: ...
@overload
def overloaded(x: int) -> int: ...
def overloaded(x: Any) -> Any: ...

def runtime_identity[T](x: T) -> T:
    return x

def test(
    dim: Tensor[[3]],
    unknown_dim: Tensor[[int]],
    x: Tensor[[2, 3]],
    y: Tensor[[1, 3]],
    unknown_shape: Tensor[IntTuple],
    text: str,
) -> None:
    exact_dim: Tensor[[3]] = keep_dim(dim)
    exact_shape: Tensor[[2, 3]] = keep_shape(x)
    reveal_type(keep_dim(dim))  # E: revealed type: Tensor[[3]]
    reveal_type(gradual_dim(unknown_dim))  # E: revealed type: Tensor[[int]]
    reveal_type(any_dim(unknown_dim))  # E: revealed type: Tensor[[int]]
    reveal_type(keep_shape(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(gradual_shape(unknown_shape))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(any_shape(unknown_shape))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(compose(x, y))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(wrapped(x))  # E: revealed type: tuple[Tensor[[2, 3]]]
    reveal_type(wrapped_alias(x))  # E: revealed type: tuple[Tensor[[2, 3]]]
    reveal_type(annotated(x))  # E: revealed type: Tensor[[2, 3]]
    exact_dim_box: DimBox[3] = wrapped_dim(dim)
    exact_shape_box: ShapeBox[[3]] = wrapped_compact_shape(dim)
    reveal_type(wrapped_shape_call(dim))  # E: revealed type: Tensor[[3]]
    reveal_type(wrapped_int_call(dim))  # E: revealed type: Tensor[[3]]
    reveal_type(wrapped_broadcast_call(dim))  # E: revealed type: Tensor[[3]]
    reveal_type(overloaded(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(runtime_identity(text))  # E: revealed type: str

def symbolic[N: IntVar](dim: Tensor[[N]]) -> None:
    reveal_type(keep_dim(dim))  # E: revealed type: Tensor[[N]]
"#,
);

testcase!(
    test_type_shape_dsl_broadcast_returns,
    type_shape_dsl_broadcast_env(),
    r#"
import shape_extensions
import shape_extensions as shapes
from broadcast_reexport import reexported_broadcast
from shape_extensions import IntTuple, IntVar, broadcast as imported_broadcast
from shape_extensions import type_shape_dsl_function
from torch import Tensor
from typing import reveal_type

broadcast_alias = imported_broadcast

@type_shape_dsl_function
def qualified(left: IntTuple, right: IntTuple) -> IntTuple:
    return shape_extensions.broadcast(left, right)

@type_shape_dsl_function
def module_alias(left: IntTuple, right: IntTuple) -> IntTuple:
    return shapes.broadcast(left, right)

@type_shape_dsl_function
def imported(left: IntTuple, right: IntTuple) -> IntTuple:
    return imported_broadcast(left, right)

@type_shape_dsl_function
def value_alias(left: IntTuple, right: IntTuple) -> IntTuple:
    return broadcast_alias(left, right)

@type_shape_dsl_function
def reexported(left: IntTuple, right: IntTuple) -> IntTuple:
    return reexported_broadcast(left, right)

def concrete() -> Tensor[qualified(IntTuple[2, 1, 4, 1], IntTuple[1, 3, 1, 5])]: ...
def imported_result() -> Tensor[imported(IntTuple[2, 3], IntTuple[1, 3])]: ...
def aliased_result() -> Tensor[value_alias(IntTuple[2, 3], IntTuple[1, 3])]: ...
def reexported_result() -> Tensor[reexported(IntTuple[2, 3], IntTuple[1, 3])]: ...
def gradual() -> Tensor[module_alias(IntTuple, IntTuple[2, 3])]: ...
def symbolic[N: IntVar, M: IntVar](
    x: Tensor[[N]], y: Tensor[[M]],
) -> Tensor[qualified(IntTuple[N, 1], IntTuple[1, M])]: ...
def nested() -> Tensor[qualified(
    shape_extensions.broadcast(IntTuple[2, 1], IntTuple[1, 3]),
    IntTuple[2, 3],
)]: ...
def incompatible() -> Tensor[qualified(IntTuple[2, 3], IntTuple[4, 3])]: ...

def test() -> None:
    reveal_type(concrete())  # E: revealed type: Tensor[[2, 3, 4, 5]]
    reveal_type(imported_result())  # E: revealed type: Tensor[[2, 3]]
    reveal_type(aliased_result())  # E: revealed type: Tensor[[2, 3]]
    reveal_type(reexported_result())  # E: revealed type: Tensor[[2, 3]]
    reveal_type(gradual())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    incompatible()  # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[2] with dimension Int[4] at position 0

def test_symbolic[N: IntVar, M: IntVar](x: Tensor[[N]], y: Tensor[[M]]) -> None:
    reveal_type(symbolic(x, y))  # E: revealed type: Tensor[[N, M]]
    reveal_type(nested())  # E: revealed type: Tensor[[2, 3]]
"#,
);

testcase!(
    test_type_shape_dsl_invalid_broadcast_returns,
    type_shape_dsl_broadcast_env(),
    r#"
import broadcast_lookalike as lookalike_module
from broadcast_lookalike import broadcast as imported_lookalike
from shape_extensions import Int, IntTuple, broadcast as native_broadcast
from shape_extensions import type_shape_dsl_function
from torch import Tensor

def broadcast(left: IntTuple, right: IntTuple) -> IntTuple:
    return left

@type_shape_dsl_function
def local_lookalike(left: IntTuple, right: IntTuple) -> IntTuple:
    return broadcast(left, right)  # E: return value must be a bare parameter name

@type_shape_dsl_function
def module_lookalike(left: IntTuple, right: IntTuple) -> IntTuple:
    return imported_lookalike(left, right)  # E: return value must be a bare parameter name

@type_shape_dsl_function
def qualified_lookalike(left: IntTuple, right: IntTuple) -> IntTuple:
    return lookalike_module.broadcast(left, right)  # E: return value must be a bare parameter name

@type_shape_dsl_function
def shadowed(left: IntTuple, right: IntTuple, native_broadcast: IntTuple) -> IntTuple:
    return native_broadcast(left, right)  # E: return value must be a bare parameter name  # E: Expected a callable

@type_shape_dsl_function
def missing(left: IntTuple, right: IntTuple) -> IntTuple:
    return native_broadcast(left)  # E: `broadcast` requires exactly two positional arguments

@type_shape_dsl_function
def keyword(left: IntTuple, right: IntTuple) -> IntTuple:
    return native_broadcast(left, right=right)  # E: `broadcast` requires exactly two positional arguments  # E: Unexpected keyword argument

@type_shape_dsl_function
def expression(left: IntTuple, right: IntTuple) -> IntTuple:
    return native_broadcast(native_broadcast(left, right), right)  # E: `broadcast` arguments must be bare parameter names

@type_shape_dsl_function
def wrong_parameter(left: Int, right: IntTuple) -> IntTuple:
    return native_broadcast(left, right)  # E: broadcast return requires two `IntTuple` parameters

@type_shape_dsl_function
def wrong_result(left: IntTuple, right: IntTuple) -> Int:
    return native_broadcast(left, right)  # E: broadcast return requires two `IntTuple` parameters  # E: Returned type

def invalid_metadata() -> Tensor[local_lookalike(IntTuple[2], IntTuple[2])]: ...  # E: Expected a type-level DSL function
"#,
);

testcase!(
    test_type_shape_dsl_identity_import_resolution,
    type_shape_dsl_import_env(),
    r#"
import identities
import identities as identities_alias
from identities import shape_identity
from identities import shape_identity as renamed_identity
from identities import select_shape
from identities import select_shape as renamed_select_shape
from identities import diag_extent
from identities import diag_extent as renamed_diag_extent
from shape_extensions import Int, IntTuple
from torch import Tensor
from typing import reveal_type

def qualified[S: IntTuple](x: Tensor[S]) -> Tensor[identities.shape_identity(S)]: ...
def module_alias[S: IntTuple](x: Tensor[S]) -> Tensor[identities_alias.shape_identity(S)]: ...
def imported[S: IntTuple](x: Tensor[S]) -> Tensor[shape_identity(S)]: ...
def import_alias[S: IntTuple](x: Tensor[S]) -> Tensor[renamed_identity(S)]: ...
value_alias = shape_identity
def value_aliased[S: IntTuple](x: Tensor[S]) -> Tensor[value_alias(S)]: ...
select_alias = select_shape
def multi_qualified[S: IntTuple](x: Tensor[S]) -> Tensor[identities.select_shape(Int[1], S)]: ...
def multi_module_alias[S: IntTuple](x: Tensor[S]) -> Tensor[identities_alias.select_shape(Int[1], S)]: ...
def multi_imported[S: IntTuple](x: Tensor[S]) -> Tensor[select_shape(Int[1], S)]: ...
def multi_import_alias[S: IntTuple](x: Tensor[S]) -> Tensor[renamed_select_shape(Int[1], S)]: ...
def multi_value_alias[S: IntTuple](x: Tensor[S]) -> Tensor[select_alias(Int[1], S)]: ...
diag_alias = diag_extent
def flag_imported() -> Tensor[[diag_extent(Int[3], -2)]]: ...
def flag_import_alias() -> Tensor[[renamed_diag_extent(Int[3], 2)]]: ...
def flag_value_alias() -> Tensor[[diag_alias(Int[3], 2)]]: ...
def flag_qualified() -> Tensor[[identities.diag_extent(Int[3], -2)]]: ...

def test(x: Tensor[[2, 3]]) -> None:
    reveal_type(qualified(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(module_alias(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(imported(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(import_alias(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(value_aliased(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(multi_qualified(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(multi_module_alias(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(multi_imported(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(multi_import_alias(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(multi_value_alias(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(flag_imported())  # E: revealed type: Tensor[[5]]
    reveal_type(flag_import_alias())  # E: revealed type: Tensor[[5]]
    reveal_type(flag_value_alias())  # E: revealed type: Tensor[[5]]
    reveal_type(flag_qualified())  # E: revealed type: Tensor[[5]]
"#,
);

testcase!(
    test_type_shape_dsl_multi_parameter_calls,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, IntVar, broadcast, type_shape_dsl_function
from torch import Tensor
from typing import overload, reveal_type

@type_shape_dsl_function
def select_int(shape: IntTuple, dim: Int) -> Int:
    return dim

@type_shape_dsl_function
def select_shape(dim: Int, shape: IntTuple) -> IntTuple:
    return shape

@type_shape_dsl_function
def select_dim(dim: Int, shape: IntTuple) -> Int:
    return dim

@type_shape_dsl_function
def first(a: Int, b: Int, c: Int) -> Int:
    return a

@type_shape_dsl_function
def second(a: Int, b: Int, c: Int) -> Int:
    return b

@type_shape_dsl_function
def third(a: Int, b: Int, c: Int) -> Int:
    return c

def concrete_first() -> Tensor[[first(Int[2], Int[3], Int[4])]]: ...
def concrete_second() -> Tensor[[second(Int[2], Int[3], Int[4])]]: ...
def concrete_third() -> Tensor[[third(Int[2], Int[3], Int[4])]]: ...
def concrete_shape() -> Tensor[select_shape(Int[9], IntTuple[2, 3])]: ...
def concrete_dim() -> Tensor[[select_dim(Int[9], IntTuple[2, 3])]]: ...
def unused_gradual() -> Tensor[[select_int(IntTuple, Int[7])]]: ...
def selected_gradual() -> Tensor[[select_int(IntTuple[2], int)]]: ...

def symbolic[N: IntVar, S: IntTuple](x: Tensor[[N]], shape: Tensor[S]) -> Tensor[[select_int(S, N)]]: ...
def nested[S0: IntTuple, S1: IntTuple](x: Tensor[S0], y: Tensor[S1]) -> Tensor[
    select_shape(select_int(S0, Int[1]), select_shape(Int[2], broadcast(S0, S1)))
]: ...

@overload
def overloaded[N: IntVar, S: IntTuple](x: Tensor[[N]], shape: Tensor[S]) -> Tensor[[select_int(S, N)]]: ...
@overload
def overloaded(x: int, shape: int) -> int: ...
def overloaded(x: object, shape: object) -> object: ...

def test(dim: Tensor[[5]], shape: Tensor[[2, 3]], other: Tensor[[1, 3]]) -> None:
    reveal_type(concrete_first())  # E: revealed type: Tensor[[2]]
    reveal_type(concrete_second())  # E: revealed type: Tensor[[3]]
    reveal_type(concrete_third())  # E: revealed type: Tensor[[4]]
    reveal_type(concrete_shape())  # E: revealed type: Tensor[[2, 3]]
    reveal_type(concrete_dim())  # E: revealed type: Tensor[[9]]
    reveal_type(unused_gradual())  # E: revealed type: Tensor[[7]]
    reveal_type(selected_gradual())  # E: revealed type: Tensor[[int]]
    reveal_type(symbolic(dim, shape))  # E: revealed type: Tensor[[5]]
    reveal_type(nested(shape, other))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(overloaded(dim, shape))  # E: revealed type: Tensor[[5]]
"#,
);

testcase!(
    test_type_shape_dsl_gradual_returns,
    type_shape_dsl_gradual_env(),
    r#"
import shape_extensions.dsl as shape_dsl
import shape_extensions.dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function
from shape_extensions.dsl import Int as DslInt, IntTuple as DslIntTuple
from gradual_reexport import ReexportedInt
from torch import Tensor
from typing import Any, assert_type, reveal_type

gradual_assignment = DslInt.gradual

@type_shape_dsl_function
def gradual_int(x: Int) -> Int:
    return shape_dsl.Int.gradual()

@type_shape_dsl_function
def gradual_shape(x: IntTuple) -> IntTuple:
    return DslIntTuple.gradual()

@type_shape_dsl_function
def gradual_assignment_alias(x: Int) -> Int:
    return gradual_assignment()

@type_shape_dsl_function
def gradual_reexport(x: Int) -> Int:
    return ReexportedInt.gradual()

@type_shape_dsl_function
def gradual_multi(dim: Int, shape: IntTuple) -> IntTuple:
    return shape_dsl.IntTuple.gradual()

@type_shape_dsl_function
def gradual_nested_import(x: Int) -> Int:
    return shape_extensions.dsl.Int.gradual()

@type_shape_dsl_function
def identity_shape(shape: IntTuple) -> IntTuple:
    return shape

def int_result() -> Tensor[[gradual_int(Int[2])]]: ...
def shape_result() -> Tensor[gradual_shape(IntTuple[2, 3])]: ...
def assignment_alias_result() -> Tensor[[gradual_assignment_alias(Int[2])]]: ...
def reexport_result() -> Tensor[[gradual_reexport(Int[2])]]: ...
def nested_import_result() -> Tensor[[gradual_nested_import(Int[2])]]: ...
def nested_multi_result() -> Tensor[identity_shape(gradual_multi(Int[2], IntTuple[3, 4]))]: ...

def test() -> None:
    assert_type(int_result(), Tensor[[int]])
    assert_type(shape_result(), Tensor[IntTuple])
    assert_type(assignment_alias_result(), Tensor[[int]])
    assert_type(reexport_result(), Tensor[[int]])
    assert_type(nested_import_result(), Tensor[[int]])
    assert_type(nested_multi_result(), Tensor[IntTuple])
    reveal_type(DslInt.gradual)  # E: revealed type: () -> Any
    assert_type(DslInt.gradual(), Any)
"#,
);

testcase!(
    test_type_shape_dsl_invalid_gradual_returns,
    type_shape_dsl_gradual_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function
from shape_extensions.dsl import Int as DslInt, IntTuple as DslIntTuple

official_gradual = DslInt.gradual

def ordinary() -> Int: ...
def gradual() -> Int: ...

class SpoofInt:
    @staticmethod
    def gradual() -> object: ...

@type_shape_dsl_function
def positional(x: Int) -> Int:
    return official_gradual(x)  # E: @type_shape_dsl_function gradual return does not accept arguments  # E: Expected 0 positional arguments

@type_shape_dsl_function
def keyword(x: Int) -> Int:
    return official_gradual(x=x)  # E: @type_shape_dsl_function gradual return does not accept arguments  # E: Unexpected keyword argument

@type_shape_dsl_function
def starred(x: Int) -> Int:
    return official_gradual(*())  # E: @type_shape_dsl_function gradual return does not accept arguments

@type_shape_dsl_function
def keyword_starred(x: Int) -> Int:
    return official_gradual(**{})  # E: @type_shape_dsl_function gradual return does not accept arguments

@type_shape_dsl_function
def bare(x: Int) -> Int:
    return official_gradual  # E: @type_shape_dsl_function gradual return must be called  # E: Returned type

@type_shape_dsl_function
def bare_qualified_int(x: Int) -> Int:
    return dsl.Int.gradual  # E: @type_shape_dsl_function gradual return must be called  # E: Returned type

@type_shape_dsl_function
def bare_qualified_shape(x: IntTuple) -> IntTuple:
    return dsl.IntTuple.gradual  # E: @type_shape_dsl_function gradual return must be called  # E: Returned type

@type_shape_dsl_function
def nested(x: Int) -> Int:
    return (official_gradual(),)  # E: return value must be a bare parameter name, gradual return, `broadcast(...)`, `dsl.Invalid(...)`, an Int/IntTuple expression, or a validated DSL helper call  # E: Returned type

@type_shape_dsl_function
def statement(x: Int) -> Int:
    official_gradual()  # E: @type_shape_dsl_function body supports only `if` and `return`
    return x

@type_shape_dsl_function
def non_intrinsic(x: Int) -> Int:
    return ordinary()  # E: DSL helper callee must be a validated

@type_shape_dsl_function
def same_spelling_is_not_intrinsic(x: Int) -> Int:
    return gradual()  # E: DSL helper callee must be a validated

@type_shape_dsl_function
def spoof_class(x: Int) -> Int:
    return SpoofInt.gradual()  # E: DSL helper callee must be a validated  # E: Returned type

@type_shape_dsl_function
def direct_cycle(x: Int) -> Int:
    return direct_cycle(x)  # E: recursive DSL helper calls are not supported

@type_shape_dsl_function
def mutual_cycle_left(x: Int) -> Int:
    return mutual_cycle_right(x)  # E: DSL helper callee must be a validated

@type_shape_dsl_function
def mutual_cycle_right(x: Int) -> Int:
    return mutual_cycle_left(x)  # E: DSL helper callee must be a validated

@type_shape_dsl_function
def shadowed_module_alias(dsl: Int) -> Int:
    return dsl.Int.gradual()  # E: DSL helper callee must be a validated  # E: Object of class `int` has no attribute `Int`

@type_shape_dsl_function
def wrong_domain(x: Int) -> Int:
    return DslIntTuple.gradual()  # E: `@type_shape_dsl_function` declares return domain `Int`, but `shape_extensions.dsl.IntTuple.gradual()` returns `IntTuple`
"#,
);

testcase!(
    test_type_shape_dsl_if_equality,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, IntVar, type_shape_dsl_function
from shape_extensions.dsl import Int as DslInt
from torch import Tensor
from typing import assert_type

@type_shape_dsl_function
def choose(a: Int, b: Int, equal: Int, different: Int) -> Int:
    if a == b:
        return equal
    return different

@type_shape_dsl_function
def choose_not_equal(a: Int, b: Int, equal: Int, different: Int) -> Int:
    if a != b:
        return different
    return equal

@type_shape_dsl_function
def nested(a: Int, b: Int, c: Int, first: Int, second: Int, third: Int) -> Int:
    if a == b:
        if b == c:
            return first
        return second
    return third

@type_shape_dsl_function
def gradual_if_equal(a: Int, b: Int, different: Int) -> Int:
    if a == b:
        return DslInt.gradual()
    return different

@type_shape_dsl_function
def reflexive(a: Int, equal: Int, different: Int) -> Int:
    if a == a:
        return equal
    return different

def concrete_equal() -> Tensor[[choose(Int[2], Int[2], Int[7], Int[8])]]: ...
def concrete_different() -> Tensor[[choose(Int[2], Int[3], Int[7], Int[8])]]: ...
def not_equal_concrete_equal() -> Tensor[[choose_not_equal(Int[2], Int[2], Int[7], Int[8])]]: ...
def not_equal_concrete_different() -> Tensor[[choose_not_equal(Int[2], Int[3], Int[7], Int[8])]]: ...
def same_symbol[N: IntVar](x: Tensor[[N]]) -> Tensor[[choose(N, N, Int[7], Int[8])]]: ...
def different_symbols[N: IntVar, M: IntVar](x: Tensor[[N]], y: Tensor[[M]]) -> Tensor[[choose(N, M, Int[7], Int[8])]]: ...
def mixed_symbol_literal[N: IntVar](x: Tensor[[N]]) -> Tensor[[choose(N, Int[2], Int[7], Int[8])]]: ...
def nested_first() -> Tensor[[nested(Int[2], Int[2], Int[2], Int[5], Int[6], Int[7])]]: ...
def nested_second() -> Tensor[[nested(Int[2], Int[2], Int[3], Int[5], Int[6], Int[7])]]: ...
def nested_third() -> Tensor[[nested(Int[2], Int[3], Int[2], Int[5], Int[6], Int[7])]]: ...
def gradual_branch() -> Tensor[[gradual_if_equal(Int[2], Int[2], Int[9])]]: ...
def precise_branch() -> Tensor[[gradual_if_equal(Int[2], Int[3], Int[9])]]: ...
def reflexive_gradual() -> Tensor[[reflexive(Int, Int[7], Int[8])]]: ...
def distinct_gradual() -> Tensor[[choose(Int, Int, Int[7], Int[8])]]: ...

def test(x: Tensor[[2]], y: Tensor[[3]]) -> None:
    assert_type(concrete_equal(), Tensor[[7]])
    assert_type(concrete_different(), Tensor[[8]])
    assert_type(not_equal_concrete_equal(), Tensor[[7]])
    assert_type(not_equal_concrete_different(), Tensor[[8]])
    assert_type(nested_first(), Tensor[[5]])
    assert_type(nested_second(), Tensor[[6]])
    assert_type(nested_third(), Tensor[[7]])
    assert_type(gradual_branch(), Tensor[[int]])
    assert_type(precise_branch(), Tensor[[9]])
    assert_type(reflexive_gradual(), Tensor[[7]])
    assert_type(distinct_gradual(), Tensor[[int]])

def test_symbolic[N: IntVar, M: IntVar](x: Tensor[[N]], y: Tensor[[M]]) -> None:
    assert_type(same_symbol(x), Tensor[[7]])
    assert_type(different_symbols(x, y), Tensor[[int]])
    assert_type(mixed_symbol_literal(x), Tensor[[int]])
"#,
);

testcase!(
    test_type_shape_dsl_flag_values,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Flag, Int, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import Any, Literal, assert_type

@type_shape_dsl_function
def diag_extent(n: Int, k: int) -> Int:
    if k < 0:
        return n - k
    return n + k

@type_shape_dsl_function
def subtract_offset(n: Int, k: int) -> Int:
    return n - k

@type_shape_dsl_function
def ignore_flags(n: Int, enabled: bool, label: str) -> Int:
    return n

@type_shape_dsl_function
def below_minimum(n: Int, k: int) -> Int:
    if k < -9223372036854775808:
        return n + k
    return n

@type_shape_dsl_function
def negative_cutoff(n: Int, k: int) -> Int:
    if k < -1:
        return n - k
    return n + k

@type_shape_dsl_function
def above_maximum_threshold(n: Int, k: int) -> Int:
    if k < 9223372036854775808:
        return n
    return n

@type_shape_dsl_function
def below_minimum_threshold(n: Int, k: int) -> Int:
    if k < -9223372036854775809:
        return n
    return n

def positive() -> Tensor[[diag_extent(Int[3], 2)]]: ...
def negative() -> Tensor[[diag_extent(Int[3], -2)]]: ...
def zero() -> Tensor[[diag_extent(Int[3], 0)]]: ...
def broad() -> Tensor[[diag_extent(Int[3], int)]]: ...
def dynamic() -> Tensor[[diag_extent(Int[3], Any)]]: ...
def oversized() -> Tensor[[diag_extent(Int[3], 999999999999999999999999999999)]]: ...
def overflow() -> Tensor[[diag_extent(Int[9223372036854775807], 1)]]: ...
def ignored() -> Tensor[[ignore_flags(Int[4], True, "mode")]]: ...
def ignored_broad() -> Tensor[[ignore_flags(Int[4], bool, str)]]: ...
def nested_arithmetic_call() -> Tensor[[diag_extent(diag_extent(Int[3], 2), -1)]]: ...
def minimum_threshold() -> Tensor[[below_minimum(Int[3], -9223372036854775808)]]: ...
def negative_cutoff_true() -> Tensor[[negative_cutoff(Int[3], -2)]]: ...
def negative_cutoff_false() -> Tensor[[negative_cutoff(Int[3], -1)]]: ...
def positive_unrepresentable_threshold() -> Tensor[[above_maximum_threshold(Int[3], 0)]]: ...
def negative_unrepresentable_threshold() -> Tensor[[below_minimum_threshold(Int[3], 0)]]: ...
def wrong_bool() -> Tensor[[diag_extent(Int[3], True)]]: ...  # E: Expected a `Flag[int]` argument

def resize[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[diag_extent(N, K)]]: ...
def captured[K: Flag[int]](k: K) -> Tensor[[diag_extent(Int[3], K)]]: ...
def captured_twice[K: Flag[int]](first: K, second: K) -> Tensor[[diag_extent(Int[3], K)]]: ...  # E: `Flag` type parameter `K` must directly annotate exactly one function parameter, found 2
def symbolic_add[N: IntVar](x: Tensor[[N]]) -> Tensor[[diag_extent(N, 2)]]: ...
def symbolic_sub[N: IntVar](x: Tensor[[N]]) -> Tensor[[subtract_offset(N, 2)]]: ...
# Symbolic shape integers are valid `Flag[int]` values, but the DSL can inspect only values that
# are already concrete. Later generic instantiation does not re-evaluate the DSL call.
def instantiated_flag[N: IntVar](x: Tensor[[N]]) -> Tensor[[
    diag_extent(Int[3], Int[N])
]]: ...
def instantiated_product_overflow[N: IntVar](x: Tensor[[N]]) -> Tensor[[
    diag_extent(Int[N * 9223372036854775807], 1)
]]: ...
def instantiated_pow_overflow[N: IntVar](x: Tensor[[N]]) -> Tensor[[
    diag_extent(Int[2 ** N], 1)
]]: ...
def symbolic_add_overflow[N: IntVar](x: Tensor[[N]]) -> Tensor[[
    diag_extent(Int[N + 9223372036854775807], 1)
]]: ...
def symbolic_sub_overflow[N: IntVar](x: Tensor[[N]]) -> Tensor[[
    subtract_offset(Int[N - 9223372036854775807], 2)
]]: ...
def symbolic_min_subtraction[N: IntVar](x: Tensor[[N]]) -> Tensor[[
    subtract_offset(Int[N - 1], -9223372036854775808)
]]: ...

def test(x: Tensor[[3]], two: Tensor[[2]], sixty_three: Tensor[[63]]) -> None:
    assert_type(positive(), Tensor[[5]])
    assert_type(negative(), Tensor[[5]])
    assert_type(zero(), Tensor[[3]])
    assert_type(broad(), Tensor[[int]])
    assert_type(dynamic(), Tensor[[int]])
    assert_type(oversized(), Tensor[[int]])
    assert_type(overflow(), Tensor[[int]])
    assert_type(ignored(), Tensor[[4]])
    assert_type(ignored_broad(), Tensor[[4]])
    assert_type(nested_arithmetic_call(), Tensor[[6]])
    assert_type(minimum_threshold(), Tensor[[3]])
    assert_type(negative_cutoff_true(), Tensor[[5]])
    assert_type(negative_cutoff_false(), Tensor[[2]])
    assert_type(positive_unrepresentable_threshold(), Tensor[[int]])
    assert_type(negative_unrepresentable_threshold(), Tensor[[int]])
    assert_type(resize(x, 2), Tensor[[5]])
    assert_type(resize(x, -2), Tensor[[5]])
    assert_type(captured_twice(2, 2), Tensor[[5]])
    captured_twice(2, 3)  # E: Argument `Literal[3]` is not assignable to parameter `second` with type `Literal[2]`
    assert_type(instantiated_flag(two), Tensor[[int]])
    assert_type(instantiated_product_overflow(two), Tensor[[int]])
    assert_type(instantiated_pow_overflow(sixty_three), Tensor[[int]])

def test_symbolic[N: IntVar](x: Tensor[[N]], k: Int[N], literal: Int[2], broad: Int) -> None:
    assert_type(captured(literal), Tensor[[5]])
    assert_type(captured(k), Tensor[[int]])
    assert_type(captured(broad), Tensor[[int]])
    assert_type(symbolic_add(x), Tensor[[(2 + N)]])
    assert_type(symbolic_sub(x), Tensor[[(-2 + N)]])
    assert_type(symbolic_add_overflow(x), Tensor[[int]])
    assert_type(symbolic_sub_overflow(x), Tensor[[int]])
    assert_type(symbolic_min_subtraction(x), Tensor[[int]])

def test_union_flag(k: Literal[1, 2]) -> None:
    assert_type(captured(k), Tensor[[int]])
"#,
);

testcase!(
    test_type_shape_dsl_invalid_flag_values,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def scalar_result(n: Int, k: int) -> int:  # E: Flag values are input-only
    return k

@type_shape_dsl_function
def return_flag(n: Int, k: int) -> Int:
    return k  # E: Flag parameter `k` is input-only

@type_shape_dsl_function
def reversed(n: Int, k: int) -> Int:
    if 0 < k:
        return n
    return n

@type_shape_dsl_function
def mixed_comparison(n: Int, k: int) -> Int:
    if n < k:  # E: comparison operands must both be annotated as `Int` or both be `Flag[int]`
        return n
    return n

@type_shape_dsl_function
def nonliteral(n: Int, k: int, limit: int) -> Int:
    if k < limit:
        return n
    return n

@type_shape_dsl_function
def wrong_condition_domain(n: Int, enabled: bool) -> Int:
    if enabled < 0:  # E: Flag operation requires a compatible Flag parameter
        return n
    return n

@type_shape_dsl_function
def union_condition_domain(n: Int, offset: int | bool) -> Int:
    if offset < 0:  # E: Flag operation requires a compatible Flag parameter
        return n
    return n

@type_shape_dsl_function
def nested_arithmetic(n: Int, k: int) -> Int:
    return (n + k) + k

@type_shape_dsl_function
def multiplication(n: Int, k: int) -> Int:
    return n * k

@type_shape_dsl_function
def reversed_arithmetic(n: Int, k: int) -> Int:
    return k + n

@type_shape_dsl_function
def union_arithmetic(n: Int, k: int | bool) -> Int:
    return n + k  # E: dimension arithmetic operands must be annotated as `Int` or `Flag[int]`

@type_shape_dsl_function
def boolean_arithmetic(n: Int, enabled: bool) -> Int:
    return n + enabled  # E: dimension arithmetic operands must be annotated as `Int` or `Flag[int]`

@type_shape_dsl_function
def sequence_arithmetic(n: Int, values: tuple[int, ...]) -> Int:
    return n + values  # E: is not supported between  # E: dimension arithmetic operands must be annotated as `Int` or `Flag[int]`

@type_shape_dsl_function
def shape_arithmetic(n: Int, shape: IntTuple) -> Int:
    return n + shape  # E: is not supported between  # E: dimension arithmetic operands must be annotated as `Int` or `Flag[int]`

@type_shape_dsl_function
def self_referencing_local(n: Int, k: int) -> Int:
    result = result + k  # E: definitely assigned before use  # E: is uninitialized
    return result

@type_shape_dsl_function
def power(n: Int, k: int) -> Int:
    return n ** k  # E: dimension arithmetic supports only `+`, `-`, `*`, `//`, and `%`

@type_shape_dsl_function
def wrong_result(n: Int, k: int) -> IntTuple:
    return n + k  # E: returned expression requires a result in the `Int` domain  # E: Returned type

@type_shape_dsl_function
def inconsistent_local(n: Int, shape: IntTuple, k: int) -> Int:
    result = n + k  # E: an integer local cannot be used as both a dimension and a Flag value
    dimension = result + shape[0]
    return shape[result] + dimension

@type_shape_dsl_function
def flag_helper(n: Int, k: int) -> Int:
    return n + k

@type_shape_dsl_function
def inconsistent_helper(n: Int, shape: IntTuple, k: int) -> Int:
    result = n + k
    dimension = result + shape[0]
    return flag_helper(dimension, result)  # E: DSL helper argument domains must exactly match

@type_shape_dsl_function
def defaulted_helper_mismatch(n: Int, k: int) -> Int:
    result = n + k
    return flag_helper(n, result)  # E: DSL helper argument domains must exactly match

@type_shape_dsl_function
def int_helper(n: Int) -> Int:
    return n

@type_shape_dsl_function
def inconsistent_helper_branches(n: Int, k: int, first: bool) -> Int:
    result = n + k
    if first:
        return int_helper(result)
    return flag_helper(n, result)  # E: DSL helper argument domains must exactly match

@type_shape_dsl_function
def conflicting_branch_return_dimension_first(
    n: Int, shape: IntTuple, k: int, first: bool
) -> Int:
    if first:
        result = n + k
        dimension = result + shape[0]
    else:
        result = n - k  # E: an integer local cannot be used as both a dimension and a Flag value
        selected = shape[result]
    return result

@type_shape_dsl_function
def conflicting_branch_return_flag_first(
    n: Int, shape: IntTuple, k: int, first: bool
) -> Int:
    if first:
        result = n + k
        selected = shape[result]
    else:
        result = n - k  # E: an integer local cannot be used as both a dimension and a Flag value
        dimension = result + shape[0]
    return result

@type_shape_dsl_function
def conflicting_branch_int_helper_dimension_first(
    n: Int, shape: IntTuple, k: int, first: bool
) -> Int:
    if first:
        result = n + k
        dimension = result + shape[0]
    else:
        result = n - k  # E: an integer local cannot be used as both a dimension and a Flag value
        selected = shape[result]
    return int_helper(result)

@type_shape_dsl_function
def conflicting_branch_int_helper_flag_first(
    n: Int, shape: IntTuple, k: int, first: bool
) -> Int:
    if first:
        result = n + k
        selected = shape[result]
    else:
        result = n - k  # E: an integer local cannot be used as both a dimension and a Flag value
        dimension = result + shape[0]
    return int_helper(result)

@type_shape_dsl_function
def conflicting_branch_flag_helper_dimension_first(
    n: Int, shape: IntTuple, k: int, first: bool
) -> Int:
    if first:
        result = n + k
        dimension = result + shape[0]
    else:
        result = n - k  # E: an integer local cannot be used as both a dimension and a Flag value
        selected = shape[result]
    return flag_helper(n, result)

@type_shape_dsl_function
def conflicting_branch_flag_helper_flag_first(
    n: Int, shape: IntTuple, k: int, first: bool
) -> Int:
    if first:
        result = n + k
        selected = shape[result]
    else:
        result = n - k  # E: an integer local cannot be used as both a dimension and a Flag value
        dimension = result + shape[0]
    return flag_helper(n, result)
"#,
);

testcase!(
    test_type_shape_dsl_flag_less_than,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, Int, IntTuple, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import reveal_type

@type_shape_dsl_function
def flag_less(shape: IntTuple, left: int, right: int) -> IntTuple:
    if left < right:
        return shape
    return dsl.IntTuple(())

@type_shape_dsl_function
def dimension_less(left: Int, right: Int) -> Int:
    if left < right:
        return left
    return right

@type_shape_dsl_function
def mixed_less(shape: IntTuple, left: Int, right: int) -> IntTuple:
    if left < right:  # E: comparison operands must both be annotated as `Int` or both be `Flag[int]`
        return shape
    return shape

def apply[Shape: IntTuple, Left: Flag[int], Right: Flag[int]](
    x: Tensor[Shape], left: Left, right: Right,
) -> Tensor[flag_less(Shape, Left, Right)]: ...

def apply_dimension[N: IntVar, M: IntVar](
    left: Tensor[[N]], right: Tensor[[M]],
) -> Tensor[[dimension_less(N, M)]]: ...

def test(x: Tensor[[2, 3]], broad_left: int, broad_right: int) -> None:
    reveal_type(apply(x, 1, 2))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply(x, 2, 1))  # E: revealed type: Tensor[[]]
    reveal_type(apply(x, 2, 2))  # E: revealed type: Tensor[[]]
    reveal_type(apply(x, broad_left, broad_right))  # E: revealed type: Tensor[tuple[Unknown, ...]]

def test_symbolic[N: IntVar, M: IntVar](left: Tensor[[N]], right: Tensor[[M]]) -> None:
    reveal_type(apply_dimension(left, right))  # E: revealed type: Tensor[[int]]
"#,
);

testcase!(
    test_type_shape_dsl_invalid_if_declarations,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, type_shape_dsl_function
from shape_extensions.dsl import IntTuple as DslIntTuple

@type_shape_dsl_function
def chained(a: Int, b: Int, c: Int) -> Int:
    if a == b == c:  # E: @type_shape_dsl_function comparison must be exactly
        return a
    return b

@type_shape_dsl_function
def other_comparison(a: Int, b: Int) -> Int:
    if a <= b:  # E: `Int` comparisons support only `==`, `!=`, and `<`
        return a
    return b

@type_shape_dsl_function
def non_parameter(a: Int, b: Int) -> Int:
    if a == 1:  # E: Flag operation requires a compatible Flag parameter
        return a
    return b

@type_shape_dsl_function
def with_else(a: Int, b: Int) -> Int:
    if a == b:
        return a
    else:
        return b

@type_shape_dsl_function
def unsupported_statement(a: Int) -> Int:
    x = a
    return x

@type_shape_dsl_function
def unreachable(a: Int) -> Int:
    return a
    return a  # E: @type_shape_dsl_function statement is unreachable  # E: This `return` statement is unreachable

@type_shape_dsl_function
def fallthrough(a: Int, b: Int) -> Int:  # E: @type_shape_dsl_function every control-flow path must return  # E: one or more paths are missing an explicit `return`
    if a == b:
        return a

@type_shape_dsl_function
def tuple_condition(a: IntTuple, b: IntTuple) -> IntTuple:
    if a == b:  # E: comparison operands must both be annotated as `Int` or both be `Flag[int]`
        return a
    return b

@type_shape_dsl_function
def mixed_comparison(a: Int, b: int) -> Int:
    if a == b:  # E: comparison operands must both be annotated as `Int` or both be `Flag[int]`
        return a
    return a

@type_shape_dsl_function
def bool_comparison(a: bool, b: bool, result: Int) -> Int:
    if a == b:  # E: comparison operands must both be annotated as `Int` or both be `Flag[int]`
        return result
    return result

@type_shape_dsl_function
def indexed_order(shape: IntTuple) -> IntTuple:
    if shape[0] < shape[1]:  # E: derived dimension comparisons support only `==` and `!=`
        return shape
    return shape

@type_shape_dsl_function
def local_derived_order(shape: IntTuple) -> IntTuple:
    item = shape[0]
    if item < shape[1]:  # E: derived dimension comparisons support only `==` and `!=`
        return shape
    return shape

@type_shape_dsl_function
def branch_local(shape: IntTuple, choose: bool) -> IntTuple:
    if choose:
        item = shape[0]
    if item == 0:  # E: local value must be definitely assigned before use  # E: may be uninitialized
        return shape
    return shape

@type_shape_dsl_function
def mismatched_return(a: Int, shape: IntTuple) -> Int:
    if a == a:
        return shape  # E: return annotation must match returned parameter `shape`  # E: Returned type
    return a

@type_shape_dsl_function
def mismatched_gradual_paths(a: Int, b: Int) -> Int:
    if a == b:
        return DslIntTuple.gradual()  # E: declares return domain `Int`, but `shape_extensions.dsl.IntTuple.gradual()` returns `IntTuple`
    return DslIntTuple.gradual()  # E: declares return domain `Int`, but `shape_extensions.dsl.IntTuple.gradual()` returns `IntTuple`
"#,
);

testcase!(
    test_type_shape_dsl_is_concrete_int_resolution,
    type_shape_dsl_predicate_env(),
    r#"
import shape_extensions.dsl as dsl
import shape_extensions.dsl
from predicate_reexport import predicate as reexported_predicate
from shape_extensions import Int, IntVar, type_shape_dsl_function
from shape_extensions.dsl import is_concrete_int
from shape_extensions.dsl import is_concrete_int as imported_alias
from torch import Tensor
from typing import Any, reveal_type

value_alias = is_concrete_int

@type_shape_dsl_function
def direct(x: Int, yes: Int, no: Int) -> Int:
    if is_concrete_int(x):
        return yes
    return no

@type_shape_dsl_function
def qualified(x: Int, yes: Int, no: Int) -> Int:
    if dsl.is_concrete_int(x):
        return yes
    return no

@type_shape_dsl_function
def fully_qualified(x: Int, yes: Int, no: Int) -> Int:
    if shape_extensions.dsl.is_concrete_int(x):
        return yes
    return no

@type_shape_dsl_function
def imported(x: Int, yes: Int, no: Int) -> Int:
    if imported_alias(x):
        return yes
    return no

@type_shape_dsl_function
def value_aliased(x: Int, yes: Int, no: Int) -> Int:
    if value_alias(x):
        return yes
    return no

@type_shape_dsl_function
def reexported(x: Int, yes: Int, no: Int) -> Int:
    if reexported_predicate(x):
        return yes
    return no

def literal() -> Tensor[[direct(Int[2], Int[7], Int[8])]]: ...
def computed_literal() -> Tensor[[qualified(Int[1 + 1], Int[7], Int[8])]]: ...
def gradual() -> Tensor[[fully_qualified(Int, Int[7], Int[8])]]: ...
def symbolic[N: IntVar](x: Tensor[[N]]) -> Tensor[[imported(N, Int[7], Int[8])]]: ...
def solved_literal[N: IntVar](x: Tensor[[N]]) -> Tensor[[direct(N, Int[7], Int[8])]]: ...
def aliased_literal() -> Tensor[[value_aliased(Int[2], Int[7], Int[8])]]: ...
def reexported_literal() -> Tensor[[reexported(Int[2], Int[7], Int[8])]]: ...
# `Any` is admitted without error but is not readable as an `Int`, so the guard is unknown and
# must fall back gradually instead of taking the precise `Int[8]` false branch.
def any_argument() -> Tensor[[direct(Any, Int[7], Int[8])]]: ...

def test(x: Tensor[[2]]) -> None:
    reveal_type(literal())  # E: revealed type: Tensor[[7]]
    reveal_type(computed_literal())  # E: revealed type: Tensor[[7]]
    reveal_type(gradual())  # E: revealed type: Tensor[[8]]
    reveal_type(solved_literal(x))  # E: revealed type: Tensor[[7]]
    reveal_type(aliased_literal())  # E: revealed type: Tensor[[7]]
    reveal_type(reexported_literal())  # E: revealed type: Tensor[[7]]
    reveal_type(any_argument())  # E: revealed type: Tensor[[int]]

def test_symbolic[N: IntVar](x: Tensor[[N]]) -> None:
    reveal_type(symbolic(x))  # E: revealed type: Tensor[[8]]
"#,
);

testcase!(
    test_type_shape_dsl_dimension_arithmetic,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, Int, IntTuple, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import reveal_type

@type_shape_dsl_function
def add_multiply(n: Int, k: int) -> Int:
    return (n + k) * 2

@type_shape_dsl_function
def local_add_multiply(n: Int, k: int) -> Int:
    result = n + k
    return result * 2

@type_shape_dsl_function
def local_add(n: Int, k: int) -> Int:
    result = n + k
    return result

@type_shape_dsl_function
def branch_local_add(n: Int, k: int, use_add: bool) -> Int:
    if use_add:
        result = n + k
    else:
        result = n - k
    return result * 2

@type_shape_dsl_function
def mixed_dimension_branch(n: Int, shape: IntTuple, k: int, first: bool) -> Int:
    if first:
        result = n + k
    else:
        result = shape[0] + k
    return result

@type_shape_dsl_function
def mixed_flag_branch(shape: IntTuple, index: int, first: bool) -> Int:
    if first:
        next_index = index + 1
    else:
        next_index = len(shape) - 1
    result = shape[next_index]
    return result

@type_shape_dsl_function
def resolved_dimension_branches(
    n: Int, shape: IntTuple, k: int, first: bool
) -> Int:
    if first:
        result = n + k
        dimension = result + shape[0]
    else:
        result = n - k
        dimension = result + shape[0]
    return result

@type_shape_dsl_function
def resolved_flag_branches(
    shape: IntTuple, index: int, k: int, first: bool
) -> Int:
    if first:
        result = index + k
        selected = shape[result]
    else:
        result = index - k
        selected = shape[result]
    return selected

@type_shape_dsl_function
def inherited_dimension_branch(
    n: Int, shape: IntTuple, k: int, first: bool
) -> Int:
    if first:
        result = n + k
        dimension = result + shape[0]
    else:
        result = n - k
    return result

@type_shape_dsl_function
def inherited_flag_branch(
    shape: IntTuple, index: int, k: int, first: bool
) -> Int:
    if first:
        result = index + k
        branch_selected = shape[result]
    else:
        result = index - k
    selected = shape[result]
    return selected

@type_shape_dsl_function
def boundary_add(n: Int) -> Int:
    return n + 9223372036854775807

@type_shape_dsl_function
def boundary_subtract(n: Int) -> Int:
    return n - 9223372036854775807

@type_shape_dsl_function
def boundary_reverse_subtract(n: Int) -> Int:
    return 9223372036854775807 - n

@type_shape_dsl_function
def coefficient_overflow(n: Int) -> Int:
    return n * 9223372036854775807 + n

@type_shape_dsl_function
def floor_divide(n: Int, k: int) -> Int:
    return n // k

@type_shape_dsl_function
def modulo(n: Int, k: int) -> Int:
    return n % k

@type_shape_dsl_function
def computed_zero_divisor(n: Int) -> Int:
    return n // (1 - 1)

@type_shape_dsl_function
def local_extent(shape: IntTuple, k: int) -> Int:
    extent = shape[0] * k
    return extent

@type_shape_dsl_function
def tuple_extent(shape: IntTuple, k: int) -> IntTuple:
    return dsl.IntTuple((shape[0] + k, shape[1] // 2))

@type_shape_dsl_function
def generator_extent(shape: IntTuple, k: int) -> IntTuple:
    return dsl.IntTuple((item * k for item in shape))

@type_shape_dsl_function
def operation_matrix(n: Int, k: int) -> IntTuple:
    return dsl.IntTuple((
        n + k, k + n, n - k + 10, k - n + 10, n * k, k * n,
        n // k, k // n, n % k, k % n,
    ))

@type_shape_dsl_function
def helper_extent(n: Int, k: int) -> Int:
    return n * k

@type_shape_dsl_function
def helper_identity(n: Int) -> Int:
    return n

@type_shape_dsl_function
def call_helper(n: Int, k: int) -> Int:
    return helper_extent(n, k)

@type_shape_dsl_function
def call_int_helper(n: Int, k: int) -> Int:
    result = n + k
    return helper_identity(result)

@type_shape_dsl_function
def call_flag_helper(n: Int, k: int) -> Int:
    next_k = k + 1
    return helper_extent(n, next_k)

@type_shape_dsl_function
def call_chained_local_helper(n: Int, k: int, offset: int) -> Int:
    first = n + k
    second = first + offset
    return helper_identity(second)

@type_shape_dsl_function
def flag_floor(left: int, right: int) -> Int:
    return left // right

@type_shape_dsl_function
def flag_modulo(left: int, right: int) -> Int:
    return left % right

@type_shape_dsl_function
def flag_add(left: int, right: int) -> Int:
    return left + right

@type_shape_dsl_function
def flag_subtract(left: int, right: int) -> Int:
    return left - right

@type_shape_dsl_function
def flag_multiply(left: int, right: int) -> Int:
    return left * right

@type_shape_dsl_function
def negative_floor(left: int, right: int, offset: int) -> Int:
    return (left // right) + offset

@type_shape_dsl_function
def negative_modulo(left: int, right: int, offset: int) -> Int:
    return (left % right) + offset

def apply_add_multiply[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[add_multiply(N, K)]]: ...
def apply_local_add_multiply[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[local_add_multiply(N, K)]]: ...
def apply_local_add[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[local_add(N, K)]]: ...
def apply_branch_local_add[N: IntVar, K: Flag[int], UseAdd: Flag[bool]](
    x: Tensor[[N]], k: K, use_add: UseAdd,
) -> Tensor[[branch_local_add(N, K, UseAdd)]]: ...
def apply_mixed_dimension_branch[
    N: IntVar, Shape: IntTuple, K: Flag[int], First: Flag[bool]
](x: Tensor[[N]], shape: Tensor[Shape], k: K, first: First) -> Tensor[[
    mixed_dimension_branch(N, Shape, K, First)
]]: ...
def apply_mixed_flag_branch[
    Shape: IntTuple, Index: Flag[int], First: Flag[bool]
](shape: Tensor[Shape], index: Index, first: First) -> Tensor[[
    mixed_flag_branch(Shape, Index, First)
]]: ...
def apply_resolved_dimension_branches[
    N: IntVar, Shape: IntTuple, K: Flag[int], First: Flag[bool]
](x: Tensor[[N]], shape: Tensor[Shape], k: K, first: First) -> Tensor[[
    resolved_dimension_branches(N, Shape, K, First)
]]: ...
def apply_resolved_flag_branches[
    Shape: IntTuple, Index: Flag[int], K: Flag[int], First: Flag[bool]
](shape: Tensor[Shape], index: Index, k: K, first: First) -> Tensor[[
    resolved_flag_branches(Shape, Index, K, First)
]]: ...
def apply_inherited_dimension_branch[
    N: IntVar, Shape: IntTuple, K: Flag[int], First: Flag[bool]
](x: Tensor[[N]], shape: Tensor[Shape], k: K, first: First) -> Tensor[[
    inherited_dimension_branch(N, Shape, K, First)
]]: ...
def apply_inherited_flag_branch[
    Shape: IntTuple, Index: Flag[int], K: Flag[int], First: Flag[bool]
](shape: Tensor[Shape], index: Index, k: K, first: First) -> Tensor[[
    inherited_flag_branch(Shape, Index, K, First)
]]: ...
def apply_boundary_add[N: IntVar](x: Tensor[[N]]) -> Tensor[[boundary_add(N)]]: ...
def apply_boundary_subtract[N: IntVar](x: Tensor[[N]]) -> Tensor[[boundary_subtract(N)]]: ...
def apply_boundary_reverse_subtract[N: IntVar](x: Tensor[[N]]) -> Tensor[[boundary_reverse_subtract(N)]]: ...
def apply_coefficient_overflow[N: IntVar](x: Tensor[[N]]) -> Tensor[[coefficient_overflow(N)]]: ...
def apply_floor_divide[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[floor_divide(N, K)]]: ...
def apply_modulo[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[modulo(N, K)]]: ...
def apply_computed_zero_divisor[N: IntVar](x: Tensor[[N]]) -> Tensor[[computed_zero_divisor(N)]]: ...
def apply_local[Shape: IntTuple, K: Flag[int]](
    x: Tensor[Shape], k: K,
) -> Tensor[[local_extent(Shape, K)]]: ...
def apply_tuple[Shape: IntTuple, K: Flag[int]](
    x: Tensor[Shape], k: K,
) -> Tensor[tuple_extent(Shape, K)]: ...
def apply_generator[Shape: IntTuple, K: Flag[int]](
    x: Tensor[Shape], k: K,
) -> Tensor[generator_extent(Shape, K)]: ...
def apply_operation_matrix[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[operation_matrix(N, K)]: ...
def apply_helper[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[call_helper(N, K)]]: ...
def apply_int_helper[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[call_int_helper(N, K)]]: ...
def apply_flag_helper[N: IntVar, K: Flag[int]](
    x: Tensor[[N]], k: K,
) -> Tensor[[call_flag_helper(N, K)]]: ...
def apply_chained_local_helper[N: IntVar, K: Flag[int], Offset: Flag[int]](
    x: Tensor[[N]], k: K, offset: Offset,
) -> Tensor[[call_chained_local_helper(N, K, Offset)]]: ...
def apply_flag_floor[Left: Flag[int], Right: Flag[int]](
    left: Left, right: Right,
) -> Tensor[[flag_floor(Left, Right)]]: ...
def apply_flag_modulo[Left: Flag[int], Right: Flag[int]](
    left: Left, right: Right,
) -> Tensor[[flag_modulo(Left, Right)]]: ...

def exact_negative() -> Tensor[[
    negative_floor(-5, 2, 10), negative_modulo(-5, 2, 10), negative_modulo(5, -2, 10)
]]: ...
def exact_overflow() -> Tensor[[add_multiply(Int[9223372036854775807], 1)]]: ...
def add_overflow() -> Tensor[[flag_add(9223372036854775807, 1)]]: ...
def add_overflow_reversed() -> Tensor[[flag_add(1, 9223372036854775807)]]: ...
def subtract_overflow() -> Tensor[[flag_subtract(-9223372036854775808, 1)]]: ...
def subtract_overflow_reversed() -> Tensor[[flag_subtract(9223372036854775807, -1)]]: ...
def multiply_overflow() -> Tensor[[flag_multiply(9223372036854775807, 2)]]: ...
def multiply_overflow_reversed() -> Tensor[[flag_multiply(2, 9223372036854775807)]]: ...
def divide_overflow() -> Tensor[[flag_floor(-9223372036854775808, -1)]]: ...
def modulo_min_by_negative_one() -> Tensor[[flag_modulo(-9223372036854775808, -1)]]: ...

def tuple_overflow() -> Tensor[tuple_extent(
    IntTuple[9223372036854775807, 8], 1
)]: ...

def test(one: Tensor[[6]], concrete: Tensor[[6, 8]], broad: int) -> None:
    reveal_type(apply_add_multiply(one, 1))  # E: revealed type: Tensor[[14]]
    reveal_type(apply_local_add_multiply(one, 1))  # E: revealed type: Tensor[[14]]
    reveal_type(apply_local_add(one, 1))  # E: revealed type: Tensor[[7]]
    reveal_type(apply_branch_local_add(one, 1, True))  # E: revealed type: Tensor[[14]]
    reveal_type(apply_branch_local_add(one, 1, False))  # E: revealed type: Tensor[[10]]
    reveal_type(apply_mixed_dimension_branch(one, concrete, 1, True))  # E: revealed type: Tensor[[7]]
    reveal_type(apply_mixed_dimension_branch(one, concrete, 1, False))  # E: revealed type: Tensor[[7]]
    reveal_type(apply_mixed_flag_branch(concrete, 0, True))  # E: revealed type: Tensor[[8]]
    reveal_type(apply_mixed_flag_branch(concrete, 0, False))  # E: revealed type: Tensor[[8]]
    reveal_type(apply_resolved_dimension_branches(one, concrete, 1, True))  # E: revealed type: Tensor[[7]]
    reveal_type(apply_resolved_dimension_branches(one, concrete, 1, False))  # E: revealed type: Tensor[[5]]
    reveal_type(apply_resolved_flag_branches(concrete, 0, 1, True))  # E: revealed type: Tensor[[8]]
    reveal_type(apply_resolved_flag_branches(concrete, 0, 1, False))  # E: revealed type: Tensor[[8]]
    reveal_type(apply_inherited_dimension_branch(one, concrete, 1, True))  # E: revealed type: Tensor[[7]]
    reveal_type(apply_inherited_dimension_branch(one, concrete, 1, False))  # E: revealed type: Tensor[[5]]
    reveal_type(apply_inherited_flag_branch(concrete, 0, 1, True))  # E: revealed type: Tensor[[8]]
    reveal_type(apply_inherited_flag_branch(concrete, 0, 1, False))  # E: revealed type: Tensor[[8]]
    reveal_type(apply_floor_divide(one, 4))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_modulo(one, 4))  # E: revealed type: Tensor[[2]]
    reveal_type(apply_local(concrete, 3))  # E: revealed type: Tensor[[18]]
    reveal_type(apply_tuple(concrete, 2))  # E: revealed type: Tensor[[8, 4]]
    reveal_type(apply_generator(concrete, 2))  # E: revealed type: Tensor[[12, 16]]
    reveal_type(apply_operation_matrix(one, 2))  # E: revealed type: Tensor[[8, 8, 14, 6, 12, 12, 3, 0, 0, 2]]
    reveal_type(apply_helper(one, 3))  # E: revealed type: Tensor[[18]]
    reveal_type(apply_int_helper(one, 3))  # E: revealed type: Tensor[[9]]
    reveal_type(apply_flag_helper(one, 2))  # E: revealed type: Tensor[[18]]
    reveal_type(apply_chained_local_helper(one, 2, 3))  # E: revealed type: Tensor[[11]]
    reveal_type(apply_add_multiply(one, broad))  # E: revealed type: Tensor[[int]]
    reveal_type(exact_negative())  # E: revealed type: Tensor[[7, 11, 9]]
    reveal_type(exact_overflow())  # E: revealed type: Tensor[[int]]
    reveal_type(add_overflow())  # E: revealed type: Tensor[[int]]
    reveal_type(add_overflow_reversed())  # E: revealed type: Tensor[[int]]
    reveal_type(subtract_overflow())  # E: revealed type: Tensor[[int]]
    reveal_type(subtract_overflow_reversed())  # E: revealed type: Tensor[[int]]
    reveal_type(multiply_overflow())  # E: revealed type: Tensor[[int]]
    reveal_type(multiply_overflow_reversed())  # E: revealed type: Tensor[[int]]
    reveal_type(divide_overflow())  # E: revealed type: Tensor[[int]]
    reveal_type(modulo_min_by_negative_one())  # E: revealed type: Tensor[[0]]
    reveal_type(tuple_overflow())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    apply_flag_floor(broad, 0)  # E: dimension integer division by zero
    apply_flag_modulo(broad, 0)  # E: dimension integer modulo by zero

def test_symbolic[N: IntVar](x: Tensor[[N]]) -> None:
    reveal_type(apply_add_multiply(x, 1))  # E: revealed type: Tensor[[(2 + (2 * N))]]
    reveal_type(apply_local_add_multiply(x, 1))  # E: revealed type: Tensor[[(2 + (2 * N))]]
    reveal_type(apply_local_add(x, 1))  # E: revealed type: Tensor[[(1 + N)]]
    reveal_type(apply_int_helper(x, 3))  # E: revealed type: Tensor[[(3 + N)]]
    reveal_type(apply_flag_helper(x, 2))  # E: revealed type: Tensor[[(3 * N)]]
    reveal_type(apply_chained_local_helper(x, 2, 3))  # E: revealed type: Tensor[[(5 + N)]]
    reveal_type(apply_boundary_add(x))  # E: revealed type: Tensor[[(9223372036854775807 + N)]]
    reveal_type(apply_boundary_subtract(x))  # E: revealed type: Tensor[[(-9223372036854775807 + N)]]
    reveal_type(apply_boundary_reverse_subtract(x))  # E: revealed type: Tensor[[(9223372036854775807 + (-1 * N))]]
    reveal_type(apply_coefficient_overflow(x))  # E: revealed type: Tensor[[int]]
    reveal_type(apply_floor_divide(x, 2))  # E: revealed type: Tensor[[(N // 2)]]
    reveal_type(apply_modulo(x, 2))  # E: revealed type: Tensor[[int]]
    apply_floor_divide(x, 0)  # E: dimension integer division by zero
    apply_modulo(x, 0)  # E: dimension integer modulo by zero
    apply_computed_zero_divisor(x)  # E: dimension integer division by zero
"#,
);

// `scaled` mixes a conditional dimension with a `Flag[int]` parameter, so the bare parameter
// names a helper argument can be traced back to do not determine its integer domain.
testcase!(
    test_type_shape_dsl_untraceable_deferred_integer_resolves_as_dimension,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Flag, Int, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import reveal_type

@type_shape_dsl_function
def scale(n: Int, k: int) -> Int:
    return n * k

@type_shape_dsl_function
def add_dimensions(n: Int, m: Int) -> Int:
    return n + m

@type_shape_dsl_function
def call_flag_helper(n: Int, k: int, first: bool) -> Int:
    scaled = (n if first else n) + k
    return scale(n, scaled)  # E: DSL helper argument domains must exactly match `scale`

@type_shape_dsl_function
def call_dimension_helper(n: Int, k: int, first: bool) -> Int:
    scaled = (n if first else n) + k
    return add_dimensions(n, scaled)

def apply_dimension_helper[N: IntVar, K: Flag[int], First: Flag[bool]](
    x: Tensor[[N]], k: K, first: First,
) -> Tensor[[call_dimension_helper(N, K, First)]]: ...

def test[N: IntVar](x: Tensor[[N]]) -> None:
    reveal_type(apply_dimension_helper(x, 2, True))  # E: revealed type: Tensor[[(2 + (2 * N))]]
"#,
);

testcase!(
    test_type_shape_dsl_is_concrete_int_and_lt,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntVar, type_shape_dsl_function
from shape_extensions.dsl import Int as DslInt, is_concrete_int
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def guarded_lt(a: Int, b: Int, yes: Int, no: Int) -> Int:
    if is_concrete_int(a) and a < b:
        return yes
    return no

@type_shape_dsl_function
def unguarded_lt(a: Int, b: Int, yes: Int, no: Int) -> Int:
    if a < b:
        return yes
    return no

@type_shape_dsl_function
def reflexive_lt(a: Int, yes: Int, no: Int) -> Int:
    if a < a:
        return yes
    return no

@type_shape_dsl_function
def int_min(a: Int, b: Int) -> Int:
    if a == b:
        return a
    if is_concrete_int(a) and is_concrete_int(b):
        if a < b:
            return a
        return b
    return DslInt.gradual()

def guarded_true() -> Tensor[[guarded_lt(Int[2], Int[3], Int[7], Int[8])]]: ...
def guarded_false() -> Tensor[[guarded_lt(Int[3], Int[2], Int[7], Int[8])]]: ...
def guarded_gradual() -> Tensor[[guarded_lt(Int, Int[3], Int[7], Int[8])]]: ...
def reflexive_gradual() -> Tensor[[reflexive_lt(Int, Int[7], Int[8])]]: ...
def min_concrete() -> Tensor[[int_min(Int[2], Int[3])]]: ...
def min_gradual() -> Tensor[[int_min(Int, Int[2])]]: ...
def guarded_symbolic[N: IntVar, M: IntVar](x: Tensor[[N]], y: Tensor[[M]]) -> Tensor[[guarded_lt(N, M, Int[7], Int[8])]]: ...
def unguarded_symbolic[N: IntVar, M: IntVar](x: Tensor[[N]], y: Tensor[[M]]) -> Tensor[[unguarded_lt(N, M, Int[7], Int[8])]]: ...
def same_symbolic[N: IntVar](x: Tensor[[N]]) -> Tensor[[unguarded_lt(N, N, Int[7], Int[8])]]: ...
def reflexive_symbolic[N: IntVar](x: Tensor[[N]]) -> Tensor[[reflexive_lt(N, Int[7], Int[8])]]: ...

def test() -> None:
    reveal_type(guarded_true())  # E: revealed type: Tensor[[7]]
    reveal_type(guarded_false())  # E: revealed type: Tensor[[8]]
    reveal_type(guarded_gradual())  # E: revealed type: Tensor[[8]]
    assert_type(reflexive_gradual(), Tensor[[8]])
    reveal_type(min_concrete())  # E: revealed type: Tensor[[2]]
    reveal_type(min_gradual())  # E: revealed type: Tensor[[int]]

def test_symbolic[N: IntVar, M: IntVar](x: Tensor[[N]], y: Tensor[[M]]) -> None:
    reveal_type(guarded_symbolic(x, y))  # E: revealed type: Tensor[[8]]
    reveal_type(unguarded_symbolic(x, y))  # E: revealed type: Tensor[[int]]
    assert_type(same_symbolic(x), Tensor[[8]])
    assert_type(reflexive_symbolic(x), Tensor[[8]])
"#,
);

testcase!(
    test_type_shape_dsl_invalid_is_concrete_int,
    type_shape_dsl_predicate_env(),
    r#"
import shape_extensions.dsl as dsl
from predicate_lookalike import is_concrete_int as lookalike
from shape_extensions import Int, IntTuple, type_shape_dsl_function

class Spoof:
    @staticmethod
    def is_concrete_int(value: object) -> bool: ...

@type_shape_dsl_function
def missing(x: Int) -> Int:
    if dsl.is_concrete_int():  # E: @type_shape_dsl_function `is_concrete_int` condition requires exactly one positional argument  # E: Missing argument `value`
        return x
    return x

@type_shape_dsl_function
def excess(x: Int) -> Int:
    if dsl.is_concrete_int(x, x):  # E: @type_shape_dsl_function `is_concrete_int` condition requires exactly one positional argument  # E: Expected 1 positional argument
        return x
    return x

@type_shape_dsl_function
def keyword(x: Int) -> Int:
    if dsl.is_concrete_int(value=x):  # E: @type_shape_dsl_function `is_concrete_int` condition requires exactly one positional argument
        return x
    return x

@type_shape_dsl_function
def starred(x: Int) -> Int:
    if dsl.is_concrete_int(*(x,)):  # E: @type_shape_dsl_function `is_concrete_int` condition requires exactly one positional argument
        return x
    return x

@type_shape_dsl_function
def keyword_starred(x: Int) -> Int:
    if dsl.is_concrete_int(**{"value": x}):  # E: @type_shape_dsl_function `is_concrete_int` condition requires exactly one positional argument
        return x
    return x

@type_shape_dsl_function
def wrong_domain(x: IntTuple) -> IntTuple:
    if dsl.is_concrete_int(x):  # E: condition operands must be annotated as `Int`
        return x
    return x

@type_shape_dsl_function
def builtin_isinstance(x: Int) -> Int:
    if isinstance(x, int):  # E: @type_shape_dsl_function condition may use only boolean Flag values
        return x
    return x

@type_shape_dsl_function
def imported_lookalike(x: Int) -> Int:
    if lookalike(x):  # E: @type_shape_dsl_function condition may use only boolean Flag values
        return x
    return x

@type_shape_dsl_function
def spoof(x: Int) -> Int:
    if Spoof.is_concrete_int(x):  # E: @type_shape_dsl_function condition may use only boolean Flag values
        return x
    return x

@type_shape_dsl_function
def shadowed(is_concrete_int: Int, x: Int) -> Int:
    if is_concrete_int(x):  # E: @type_shape_dsl_function condition may use only boolean Flag values  # E: Expected a callable
        return x
    return x

@type_shape_dsl_function
def boolean_or(x: Int, y: Int) -> Int:
    if dsl.is_concrete_int(x) or dsl.is_concrete_int(y):
        return x
    return y

@type_shape_dsl_function
def other_order(x: Int, y: Int) -> Int:
    if x <= y:  # E: `Int` comparisons support only `==`, `!=`, and `<`
        return x
    return y

@type_shape_dsl_function
def tuple_lt(x: IntTuple, y: IntTuple) -> IntTuple:
    if x < y:  # E: comparison operands must both be annotated as `Int` or both be `Flag[int]`
        return x
    return y
"#,
);

testcase!(
    test_type_shape_dsl_multi_parameter_call_errors,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, broadcast, type_shape_dsl_function
from torch import Tensor
from typing import reveal_type

@type_shape_dsl_function
def select_int(shape: IntTuple, dim: Int) -> Int:
    return dim

def missing() -> Tensor[[select_int(IntTuple[2])]]: ...  # E: Expected 2 arguments for `select_int`, got 1
def excess() -> Tensor[[select_int(IntTuple[2], Int[3], Int[4])]]: ...  # E: Expected 2 arguments for `select_int`, got 3
def keyword() -> Tensor[[select_int(IntTuple[2], dim=Int[3])]]: ...  # E: `select_int` does not accept keyword arguments
def keyword_starred() -> Tensor[[select_int(**dict[str, object])]]: ...  # E: `select_int` does not accept starred keyword arguments
def positional_starred() -> Tensor[[select_int(*tuple[IntTuple[2], Int[3]])]]: ...  # E: `select_int` does not accept starred arguments
def wrong_first() -> Tensor[[select_int(Int[2], Int[3])]]: ...  # E: Expected an `IntTuple` argument for parameter `shape` (position 1) of `select_int`, got `Int[2]`
def wrong_second() -> Tensor[[select_int(IntTuple[2], IntTuple[3])]]: ...  # E: Expected an `Int` argument for parameter `dim` (position 2) of `select_int`, got `IntTuple[3]`
def stop_first() -> Tensor[[select_int(Int[2], IntTuple[3])]]: ...  # E: Expected an `IntTuple` argument for parameter `shape` (position 1) of `select_int`, got `Int[2]`
def invalid_unused_nested() -> Tensor[[select_int(broadcast(IntTuple[2], IntTuple[3]), Int[1])]]: ...

def test() -> None:
    result = invalid_unused_nested()  # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[2] with dimension Int[3] at position 0
    reveal_type(result)  # E: revealed type: Tensor[[int]]
"#,
);

testcase!(
    test_type_shape_dsl_identity_call_errors_and_boundaries,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import Callable, Concatenate, TypeGuard, TypeIs, Union, reveal_type

@type_shape_dsl_function
def int_identity(x: Int) -> Int:
    return x

@type_shape_dsl_function
def shape_identity(x: IntTuple) -> IntTuple:
    return x

def missing[S: IntTuple](x: Tensor[S]) -> Tensor[shape_identity()]: ...  # E: Expected 1 argument for `shape_identity`, got 0
def extra[S: IntTuple](x: Tensor[S]) -> Tensor[shape_identity(S, S)]: ...  # E: Expected 1 argument for `shape_identity`, got 2
def keyword[S: IntTuple](x: Tensor[S]) -> Tensor[shape_identity(x=S)]: ...  # E: `shape_identity` does not accept keyword arguments
def wrong_shape_domain(x: Tensor[[2]]) -> Tensor[shape_identity(Int[2])]: ...  # E: Expected an `IntTuple` argument for parameter `x` (position 1) of `shape_identity`, got `Int[2]`
def wrong_int_domain(x: Tensor[[2]]) -> Tensor[[int_identity(IntTuple[2])]]: ...  # E: Expected an `Int` argument for parameter `x` (position 1) of `int_identity`, got `IntTuple[2]`
def wrong_dimension_result(x: Tensor[[2]]) -> Tensor[[shape_identity(IntTuple[2])]]: ...  # E: Expected a type-level shape DSL call with an `Int` result in a shape dimension, got an `IntTuple` result
def wrong_shape_result(x: Tensor[[2]]) -> Tensor[int_identity(Int[2])]: ...  # E: Expected a type-level shape DSL call with an `IntTuple` result in a shaped-array shape argument, got an `Int` result
def nested_wrong_domain(x: Tensor[[2]]) -> Tensor[shape_identity(int_identity(IntTuple[2]))]: ...  # E: Expected an `Int` argument for parameter `x` (position 1) of `int_identity`, got `IntTuple[2]`
def malformed_int(x: Tensor[[2]]) -> Tensor[[int_identity("x")]]: ...  # E: String literals are not valid tensor dimensions
def recovered_dimension[N: IntVar]() -> Tensor[[int_identity(Int[N + MissingDim])]]: ...  # E: Could not find name `MissingDim`
def recovered_ordinary() -> Tensor[[int_identity(list[MissingType])]]: ...  # E: Could not find name `MissingType`  # E: Expected an `Int` argument for parameter `x` (position 1) of `int_identity`, got `list[Unknown]`
def nonpositive_int(x: Tensor[[2]]) -> Tensor[[int_identity(-1)]]: ...  # E: Tensor shape dimension must be positive, got -1
def malformed_shape(x: Tensor[[2]]) -> Tensor[shape_identity(IntTuple["x"])]: ...  # E: String literals are not valid tensor dimensions
def unbound_shape(x: Tensor[[2]]) -> Tensor[shape_identity(MissingShape)]: ...  # E: Could not find name `MissingShape`

BadAlias = Tensor[shape_identity(IntTuple[2])]  # E: Function call cannot be used in annotations
bad_global: Tensor[shape_identity(IntTuple[2])]  # E: Function call cannot be used in annotations
def bad_parameter(x: Tensor[shape_identity(IntTuple[2])]) -> None: ...  # E: Function call cannot be used in annotations
def bad_composed_parameter[N: IntVar](x: Tensor[shape_identity(IntTuple[int_identity(N)])]) -> None: ...  # E: Function call cannot be used in annotations
def tuple_return[S: IntTuple](x: Tensor[S]) -> tuple[Tensor[shape_identity(S)], int]: ...
def bad_tuple_paramspec[**P]() -> tuple[P]: ...  # E: `P` is not allowed in this context
def bad_tuple_concatenate[**P]() -> tuple[Concatenate[int, P]]: ...  # E: `Concatenate[int, P]` is not allowed in this context
def nested_callable[S: IntTuple]() -> tuple[Callable[[], Tensor[shape_identity(S)]]]: ...  # E: Function call cannot be used in annotations
type Deferred[T] = Callable[[], T]
def nested_callable_alias[S: IntTuple]() -> Deferred[Tensor[shape_identity(S)]]: ...
type Mixed[T, U] = tuple[T, Callable[[], U]]
def mixed_alias[S: IntTuple]() -> Mixed[Tensor[shape_identity(S)], int]: ...
type NestedDeferred[T] = tuple[int, Deferred[T]]
def nested_alias[S: IntTuple]() -> NestedDeferred[Tensor[shape_identity(S)]]: ...
type Guard[T] = TypeGuard[T]
def guard_alias[S: IntTuple](x: object) -> Guard[Tensor[shape_identity(S)]]: ...
type Narrowed[T] = TypeIs[T]
def type_is_alias[S: IntTuple](x: object) -> Narrowed[Tensor[shape_identity(S)]]: ...
type Recursive[T] = T | list[Recursive[T]]
def recursive_alias[S: IntTuple]() -> Recursive[Tensor[shape_identity(S)]]: ...
type RecursiveTransform[T] = T | list[RecursiveTransform[list[T]]]
def recursive_transform_alias[S: IntTuple]() -> RecursiveTransform[Tensor[shape_identity(S)]]: ...
type RecursiveDeferred[T] = T | Callable[[], RecursiveDeferred[T]]
def recursive_deferred_alias[S: IntTuple]() -> RecursiveDeferred[Tensor[shape_identity(S)]]: ...
type RecursiveDeferredTransform[T] = T | Callable[[], RecursiveDeferredTransform[list[T]]]
def recursive_deferred_transform_alias[S: IntTuple]() -> RecursiveDeferredTransform[Tensor[shape_identity(S)]]: ...

def pep604_union[S: IntTuple](x: Tensor[S]) -> Tensor[shape_identity(S)] | None: ...
def typing_union[S: IntTuple](x: Tensor[S]) -> Union[Tensor[shape_identity(S)], None]: ...
def bad_union_parameter[S: IntTuple](x: Tensor[shape_identity(S)] | None) -> None: ...  # E: Function call cannot be used in annotations
def bad_nested_union_callable[S: IntTuple]() -> Callable[[], Tensor[shape_identity(S)] | None]: ...  # E: Function call cannot be used in annotations
type BadUnionAlias[S: IntTuple] = Tensor[shape_identity(S)] | None  # E: Function call cannot be used in annotations

def test_union(x: Tensor[[2, 3]]) -> None:
    reveal_type(pep604_union(x))  # E: revealed type: Tensor[[2, 3]] | None
    reveal_type(typing_union(x))  # E: revealed type: Tensor[[2, 3]] | None

def runtime(x: Int[2]) -> Int:
    return int_identity(x)
"#,
);

testcase!(
    test_type_level_dsl_broadcast_rejected_outside_return_annotation,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import IntTuple, broadcast
from torch import Tensor

BadAlias = Tensor[broadcast(IntTuple[2], IntTuple[3])]  # E: Function call cannot be used in annotations

bad_global: Tensor[broadcast(IntTuple[2], IntTuple[3])]  # E: Function call cannot be used in annotations

class C:
    bad_attr: Tensor[broadcast(IntTuple[2], IntTuple[3])]  # E: Function call cannot be used in annotations

def bad_parameter[S0: IntTuple](x: Tensor[broadcast(S0, S0)]) -> None: ...  # E: Function call cannot be used in annotations
"#,
);

testcase!(
    test_type_level_dsl_broadcast_annotation_boundaries,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import IntTuple, broadcast
from torch import Tensor
from typing import Annotated, Callable, TypeGuard, TypeIs, Union

type Wrapper[T] = tuple[T]
type Recursive[T] = T | list[Recursive[T]]
type Deferred[T] = Callable[[], T]
type RecursiveDeferred[T] = Callable[[], T | RecursiveDeferred[T]]
type Rotate[A, B, C] = A | list[Rotate[B, C, A]]
type Delayed[A, B, C] = tuple[A, Callable[[], Delayed[B, C, A]]]
type Grow[T] = T | list[Grow[list[T]]]

def wrapped[S: IntTuple]() -> Wrapper[Tensor[broadcast(S, S)]]: ...
def annotated[S: IntTuple]() -> Annotated[Tensor[broadcast(S, S)], "shape"]: ...
def pep604[S: IntTuple]() -> Tensor[broadcast(S, S)] | None: ...
def union[S: IntTuple]() -> Union[Tensor[broadcast(S, S)], None]: ...
def recursive[S: IntTuple]() -> Recursive[Tensor[broadcast(S, S)]]: ...
def rotating_alias[S: IntTuple]() -> Rotate[int, str, Tensor[broadcast(S, S)]]: ...
def growing_alias() -> Grow[int]: ...

def callable_boundary[S: IntTuple]() -> Callable[[], Tensor[broadcast(S, S)]]: ...  # E: Function call cannot be used in annotations
def alias_hidden_callable[S: IntTuple]() -> Deferred[Tensor[broadcast(S, S)]]: ...
def alias_hidden_recursive_callable[S: IntTuple]() -> RecursiveDeferred[Tensor[broadcast(S, S)]]: ...
def alias_hidden_delayed_callable[S: IntTuple]() -> Delayed[int, str, Tensor[broadcast(S, S)]]: ...
def type_guard_boundary[S: IntTuple](x: object) -> TypeGuard[Tensor[broadcast(S, S)]]: ...  # E: Function call cannot be used in annotations
def type_is_boundary[S: IntTuple](x: object) -> TypeIs[Tensor[broadcast(S, S)]]: ...  # E: Function call cannot be used in annotations

def bad_arity() -> Tensor[broadcast(IntTuple[2])]: ...  # E: Expected 2 arguments for `broadcast`, got 1
def bad_keyword() -> Tensor[broadcast(IntTuple[2], right=IntTuple[2])]: ...  # E: `broadcast` does not accept keyword arguments
def bad_domain() -> Tensor[broadcast(int, IntTuple[2])]: ...  # E: Expected an `IntTuple` argument to `broadcast`
def bad_dimension() -> Tensor[[broadcast(IntTuple[2], IntTuple[2])]]: ...  # E: Expected a type-level shape DSL call with an `Int` result in a shape dimension, got an `IntTuple` result
"#,
);

testcase!(
    test_type_level_dsl_broadcast_rejected_at_direct_type_roots,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import IntTuple, broadcast
from torch import Tensor
from typing import Generic, TypeVar, TypedDict, assert_type, cast
from typing_extensions import TypeForm

LegacyBound = TypeVar("LegacyBound", bound=Tensor[broadcast(IntTuple[2], IntTuple[2])])  # E: Function call cannot be used in annotations
LegacyConstraint = TypeVar("LegacyConstraint", Tensor[broadcast(IntTuple[2], IntTuple[2])], int)  # E: Function call cannot be used in annotations
LegacyDefault = TypeVar("LegacyDefault", default=Tensor[broadcast(IntTuple[2], IntTuple[2])])  # E: Function call cannot be used in annotations

def pep_bound[T: Tensor[broadcast(IntTuple[2], IntTuple[2])]]() -> None: ...  # E: Function call cannot be used in annotations
def pep_constraint[T: (Tensor[broadcast(IntTuple[2], IntTuple[2])], int)]() -> None: ...  # E: Function call cannot be used in annotations
def pep_default[T = Tensor[broadcast(IntTuple[2], IntTuple[2])]]() -> None: ...  # E: Function call cannot be used in annotations

class BadBase(list[Tensor[broadcast(IntTuple[2], IntTuple[2])]]): ...  # E: Function call cannot be used in annotations
class BadGeneric(Generic[broadcast(IntTuple[2], IntTuple[2])]): ...  # E: Function call cannot be used in annotations
class BadMetaclass(metaclass=Tensor[broadcast(IntTuple[2], IntTuple[2])]): ...  # E: Function call cannot be used in annotations
class BadExtraItems(TypedDict, extra_items=Tensor[broadcast(IntTuple[2], IntTuple[2])]): ...  # E: Function call cannot be used in annotations

assert_type(None, Tensor[broadcast(IntTuple[2], IntTuple[2])])  # E: Function call cannot be used in annotations
cast(Tensor[broadcast(IntTuple[2], IntTuple[2])], None)  # E: Function call cannot be used in annotations
TypeForm(Tensor[broadcast(IntTuple[2], IntTuple[2])])  # E: Function call cannot be used in annotations
"#,
);

testcase!(
    test_shaped_array_inttuple_non_shape_arg_does_not_reproject,
    shaped_array_env(),
    r#"
from shape_extensions import IntTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Array[Meta: IntTuple, Shape: IntTuple, DType]:
    shape: Shape
    def clone(self) -> Array[Meta, Shape, DType]: ...

def f[Shape: IntTuple](x: Array[IntTuple[1], Shape, int]) -> None:
    y = x.clone()
    reveal_type(y)  # E: revealed type: Array[IntTuple[1], Shape, int]
"#,
);

testcase!(
    test_shaped_array_inttuple_nonzero_shape_arg_display_projection_and_subset,
    shaped_array_env(),
    r#"
from shape_extensions import IntTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class DTypeFirstArray[DType, Shape: IntTuple]:
    shape: Shape
    def dtype(self) -> DType: ...

def want_2_3(x: DTypeFirstArray[int, [2, 3]]) -> None: ...

def f(
    x: DTypeFirstArray[int, [2, 3]],
    y: DTypeFirstArray[int, [2, 4]],
) -> None:
    reveal_type(x)  # E: revealed type: DTypeFirstArray[int, [2, 3]]
    reveal_type(x.shape)  # E: revealed type: IntTuple[2, 3]
    reveal_type(x.dtype())  # E: revealed type: int
    want_2_3(x)
    want_2_3(y)  # E: Argument `DTypeFirstArray[int, [2, 4]]` is not assignable to parameter `x` with type `DTypeFirstArray[int, [2, 3]]`
"#,
);

testcase!(
    test_symbolic_size_subset_delegates_to_symbolic_leaf,
    shaped_array_env(),
    r#"
from typing import Any, reveal_type
from shape_extensions import Elements, IntTuple, IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple = tuple[Any, ...], DType = Any]: ...

def append_dim[S: IntTuple, OUT: IntVar](
    source: Array[S, int],
    result: Array[[*Elements[S], OUT], int],
) -> Array[[*Elements[S], OUT], int]:
    return result

def f[M: IntVar, N: IntVar](
    source: Array[[M], int],
    result: Array[[M, N], int],
) -> None:
    reveal_type(append_dim(source, result))  # E: revealed type: Array[[M, N], int]
"#,
);

testcase!(
    test_tensor_shapes_inttuple_assignability,
    shaped_array_env(),
    r#"
from typing import Literal
from shape_extensions import Elements, Int, IntTuple, IntVar

def takes_int_tuple(x: IntTuple) -> None: ...
def takes_tuple_of_Ints(x: tuple[Int, ...]) -> None: ...
def takes_tuple_of_ints(x: tuple[int, ...]) -> None: ...
def takes_fixed_shape(x: IntTuple[2, 3]) -> None: ...
def takes_fixed_symbolic_shape[N: IntVar](x: IntTuple[2, N]) -> None: ...
def takes_fixed_int_tuple[N: IntVar](x: tuple[Int[2], Int[N]]) -> None: ...
def takes_legacy_literal_pair(x: tuple[Literal[2], Literal[3]]) -> None: ...
def takes_int_pair(x: tuple[int, int]) -> None: ...
def takes_unpacked_shape[S: IntTuple, N: IntVar](x: IntTuple[*Elements[S], N]) -> None: ...

def bare(shape: IntTuple, ints: tuple[int, ...], Ints: tuple[Int, ...]) -> None:
    takes_tuple_of_Ints(shape)
    takes_tuple_of_ints(shape)
    takes_int_tuple(ints)
    takes_int_tuple(Ints)

def fixed[N: IntVar](
    shape: IntTuple[2, N],
    shape_23: IntTuple[2, 3],
    tuple_of_ints: tuple[Int[2], Int[N]],
    legacy_23: tuple[Literal[2], Literal[3]],
) -> None:
    takes_fixed_int_tuple(shape)
    takes_fixed_symbolic_shape(tuple_of_ints)
    takes_fixed_shape(legacy_23)
    takes_legacy_literal_pair(shape_23)
    takes_int_pair(shape)

def unpacked[S: IntTuple, N: IntVar](
    shape: IntTuple[*Elements[S], N],
    whole_shape: IntTuple[*Elements[S]],
    carrier: S,
) -> None:
    takes_unpacked_shape(shape)
    carrier_from_whole_shape: S = whole_shape
    whole_shape_from_carrier: IntTuple[*Elements[S]] = carrier

def bad[S: IntTuple, N: IntVar](
    shape_24: IntTuple[2, 4],
    int_pair: tuple[int, int],
    ints: tuple[int, ...],
    Ints: tuple[Int, ...],
    literal_Ints: tuple[Int[5], ...],
    legacy_literals: tuple[Literal[5], ...],
) -> None:
    takes_fixed_shape(shape_24)  # E: Shape dimension mismatch
    takes_fixed_shape(int_pair)  # E: is not assignable
    takes_unpacked_shape(ints)  # E: is not assignable
    takes_unpacked_shape(Ints)
    takes_unpacked_shape(literal_Ints)  # E: is not assignable
    takes_unpacked_shape(legacy_literals)  # E: is not assignable
    takes_int_tuple(literal_Ints)
    takes_int_tuple(legacy_literals)
"#,
);

testcase!(
    test_tensor_shapes_inttuple_tuple_behaviors,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import IntTuple, IntVar

def fixed[N: IntVar](shape: IntTuple[2, N]) -> None:
    reveal_type(shape[0])  # E: revealed type: Int[2]
    reveal_type(shape[1])  # E: revealed type: Int[N]
    reveal_type(shape[-1])  # E: revealed type: Int[N]
    reveal_type(shape[:1])  # E: revealed type: tuple[Int[2]]
    reveal_type(shape.count(2))  # E: revealed type: int
    first, second = shape
    reveal_type(first)  # E: revealed type: Int[2]
    reveal_type(second)  # E: revealed type: Int[N]

def bare(shape: IntTuple) -> None:
    reveal_type(shape[0])  # E: revealed type: Int[int]
    for dim in shape:
        reveal_type(dim)  # E: revealed type: Int[int]
"#,
);

testcase!(
    test_tensor_shapes_inttuple_unpacked_tuple_behaviors,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import Elements, Int, IntTuple, IntVar

def suffix_shape[S: IntTuple, N: IntVar](
    shape: IntTuple[*Elements[S], N],
    i: int,
    dim: Int[N],
) -> None:
    reveal_type(shape[0])  # E: revealed type: Int[int]
    reveal_type(shape[-1])  # E: revealed type: Int[N]
    reveal_type(shape[i])  # E: revealed type: Int[int]
    reveal_type(shape.count(dim))  # E: revealed type: int
    for elem in shape:
        reveal_type(elem)  # E: revealed type: Int[int]
    first, *middle, last = shape
    reveal_type(first)  # E: revealed type: Int[int]
    reveal_type(middle)  # E: revealed type: list[Int[int]]
    reveal_type(last)  # E: revealed type: Int[N]

def prefix_shape[S: IntTuple, N: IntVar](
    shape: IntTuple[N, *Elements[S]],
    i: int,
) -> None:
    reveal_type(shape[0])  # E: revealed type: Int[N]
    reveal_type(shape[-1])  # E: revealed type: Int[int]
    reveal_type(shape[i])  # E: revealed type: Int[int]
    for elem in shape:
        reveal_type(elem)  # E: revealed type: Int[int]
    first, *middle, last = shape
    reveal_type(first)  # E: revealed type: Int[N]
    reveal_type(middle)  # E: revealed type: list[Int[int]]
    reveal_type(last)  # E: revealed type: Int[int]
"#,
);

testcase!(
    test_tensor_shapes_ordinary_unpacked_tuple_behavior_is_not_shape_specific,
    shaped_array_env(),
    r#"
from typing import assert_type, reveal_type
from shape_extensions import Int

def ordinary(x: tuple[str, *tuple[Int, ...]]) -> None:
    reveal_type(x[0])  # E: revealed type: str
    first, *rest = x
    assert_type(first, str)
    reveal_type(rest)  # E: revealed type: list[Int[int]]
    *head, last = x
    reveal_type(head)  # E: revealed type: list[str | Int[int]]
    reveal_type(last)  # E: revealed type: str | Int[int]
"#,
);

testcase!(
    test_ordinary_typevar_shape_dimension_is_rejected,
    shaped_array_env(),
    r#"
from typing import Any, Generic, TypeVar
from shape_extensions import Int, Elements, Int, IntTuple, IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple = tuple[Any, ...], DType = Any]: ...

class SymBox[N: IntVar]: ...

def invalid[N, Shape: IntTuple](
    dim: Int[N],  # E: `N` must be an `IntVar` to be used as a shape dimension
    size: Int[N],  # E: `N` must be an `IntVar` to be used as a shape dimension
    arithmetic_dim: Int[N + 1],  # E: `N` must be an `IntVar` to be used in shape arithmetic
    list_shape: Array[[N], int],  # E: `N` must be an `IntVar` to be used as a shape dimension
    int_tuple: Array[IntTuple[N], int],  # E: `N` must be an `IntVar` to be used as a shape dimension
    unpack_prefix: Array[IntTuple[N, *Elements[Shape]], int],  # E: `N` must be an `IntVar` to be used as a shape dimension
    class_arg: SymBox[N],  # E: `N` must be an `IntVar` to be used as a shape dimension
) -> None:
    pass

type Alias[N] = Int[N]  # E: `N` must be an `IntVar` to be used as a shape dimension

LegacyN = TypeVar("LegacyN")

class LegacyBox(Generic[LegacyN]):
    dim: Int[LegacyN]  # E: `LegacyN` must be an `IntVar` to be used as a shape dimension
    size: Int[LegacyN]  # E: `LegacyN` must be an `IntVar` to be used as a shape dimension
    arithmetic_dim: Int[LegacyN + 1]  # E: `LegacyN` must be an `IntVar` to be used in shape arithmetic
    shape: Array[[LegacyN], int]  # E: `LegacyN` must be an `IntVar` to be used as a shape dimension
"#,
);

testcase!(
    test_size_bounded_typevar_is_not_symbolic_dimension,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import Int

# `N` is an ordinary `TypeVar` whose upper bound normalizes to the gradual
# `Int` type. Symbolic-ness is determined by the explicit `IntVar` kind, so a
# `Int` upper bound must NOT make the arg be parsed as a shape dimension.
class Box[N: Int]: ...

def f(a: Box[5]) -> None:  # E: Expected a type form, got instance of `Literal[5]`
    reveal_type(a)  # E: revealed type: Box[Unknown]
"#,
);

testcase!(
    test_ordinary_typevar_not_assignable_to_size,
    shaped_array_env(),
    r#"
from shape_extensions import Int

def to_size[T](x: T) -> Int:
    return x  # E: Returned type `T` is not assignable to declared return type `Int[int]`
"#,
);

testcase!(
    test_size_not_assignable_to_ordinary_typevar,
    shaped_array_env(),
    r#"
from shape_extensions import Int

def from_size[T](s: Int) -> T:
    return s  # E: Returned type `Int[int]` is not assignable to declared return type `T`
"#,
);

testcase!(
    test_module_level_intvar_dimension_does_not_panic,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar

# A legacy module-level `IntVar` used raw as a dimension resolves to a raw
# `Type::TypeVar` of `IntVar` kind (not a scoped `Quantified`). This must be
# reported gracefully as an out-of-scope type variable rather than panicking the
# checker (previously `Int::from_type` returned `None` here, hitting an
# `unreachable!`).
N = IntVar("N")

class C:
    x: Int[N]  # E: Type variable `N` is not in scope
"#,
);

testcase!(
    test_tensor_shapes_explicit_int_int_display,
    shaped_array_env(),
    r#"
from shape_extensions import Int
from typing import assert_type, reveal_type

def f(bare: Int, explicit: Int[int]) -> None:
    reveal_type(bare)  # E: revealed type: Int[int]
    reveal_type(explicit)  # E: revealed type: Int[int]
    assert_type(bare, Int[int])
    assert_type(explicit, Int[int])
"#,
);

testcase!(
    test_tensor_shapes_size_annotations_parse_to_size,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import assert_type, reveal_type

def sizes[N: IntVar](
    literal: Int[3],
    symbolic: Int[N],
    arithmetic: Int[N + 1],
    dim: Int[N + 1],
) -> None:
    reveal_type(literal)  # E: revealed type: Int[3]
    reveal_type(symbolic)  # E: revealed type: Int[N]
    reveal_type(arithmetic)  # E: revealed type: Int[(1 + N)]
    assert_type(arithmetic, Int[N + 1])
    reveal_type(dim)  # E: revealed type: Int[(1 + N)]
"#,
);

testcase!(
    test_tensor_shapes_dim_annotations_parse_to_size,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import Any, reveal_type

def bare_dim(x: Int) -> None:
    reveal_type(x)  # E: revealed type: Int[int]

def dims[N: IntVar](
    literal: Int[3],
    symbolic: Int[N],
    arithmetic: Int[N + 1],
) -> None:
    reveal_type(literal)  # E: revealed type: Int[3]
    reveal_type(symbolic)  # E: revealed type: Int[N]
    reveal_type(arithmetic)  # E: revealed type: Int[(1 + N)]
    reveal_type(arithmetic + 1)  # E: revealed type: Int[(2 + N)]

def gradual(any_dim: Int[Any], int_dim: Int[int]) -> None:
    reveal_type(int_dim)  # E: revealed type: Int[int]
    take_size3(any_dim)
    take_dim3(any_dim)
    take_size3(int_dim)
    take_dim3(int_dim)

def take_size3(x: Int[3]) -> None: ...
def take_dim3(x: Int[3]) -> None: ...
def take_size4(x: Int[4]) -> None: ...

def exact(d3: Int[3], s3: Int[3], d4: Int[4]) -> None:
    take_size3(d3)
    take_dim3(s3)
    take_size4(d3)  # E: Argument `Int[3]` is not assignable to parameter `x` with type `Int[4]`
    take_dim3(d4)  # E: Argument `Int[4]` is not assignable to parameter `x` with type `Int[3]`
"#,
);

testcase!(
    test_tensor_shapes_symbolic_int_mismatch_diagnostics,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar

def same_int[N: IntVar](left: Int[N], right: Int[N]) -> None: ...

def f[N: IntVar](n: Int[N], next_n: Int[N + 1]) -> None:
    exact: Int[N] = n
    mismatched: Int[N] = next_n  # E: Shape dimension mismatch: expected Int[N], got Int[(1 + N)]
    same_int(n, n)
    same_int(n, next_n)  # E: Argument `Int[(1 + N)]` is not assignable to parameter `right` with type `Int[N]`
"#,
);

testcase!(
    test_tensor_shapes_int_annotation_rejects_non_size_arguments,
    shaped_array_env(),
    r#"
from shape_extensions import Int

def bad_str(x: Int[str]) -> None: ...  # E: Tensor shape dimensions must be integer literals or type variables, got `type[str]`
def bad_object(x: Int[object]) -> None: ...  # E: Tensor shape dimensions must be integer literals or type variables, got `type[object]`
def bad_float(x: Int[1.5]) -> None: ...  # E: Tensor shape dimensions must be integers, not floats or complex numbers
def bad_complex(x: Int[1j]) -> None: ...  # E: Tensor shape dimensions must be integers, not floats or complex numbers
"#,
);

testcase!(
    test_tensor_shapes_int_class_and_dataclass_field_defaults,
    shaped_array_env(),
    r#"
from dataclasses import dataclass
from shape_extensions import Int
from typing import assert_type

class Config:
    d: Int = 768
    d2: Int[768] = 768

@dataclass
class DataConfig:
    d: Int = 768
    d2: Int[768] = 768

def f(config: Config, data_config: DataConfig) -> None:
    assert_type(config.d, Int[int])
    assert_type(config.d2, Int[768])
    assert_type(data_config.d, Int[int])
    assert_type(data_config.d2, Int[768])
    assert_type(DataConfig().d, Int[int])
    assert_type(DataConfig().d2, Int[768])
"#,
);

testcase!(
    test_tensor_shapes_int_annotation_pow_exponents,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import reveal_type

# The sign of symbolic forms like -M and 0 - M is not provable here, so keep
# them consistent and reject only exponents proven negative.
def valid[N: IntVar, M: IntVar](
    literal: Int[N ** 2],
    symbolic: Int[N ** M],
    symbolic_base: Int[2 ** N],
    sum_expr: Int[N ** (M + 1)],
    symbolic_negative: Int[N ** -M],
    symbolic_sub: Int[N ** (0 - M)],
) -> None:
    pass

def canonicalized[N: IntVar](
    half_power: Int[N ** (1 // 2)],
    neg_zero: Int[N ** -0],
    neg_zero_expr: Int[N ** -(1 - 1)],
) -> None:
    reveal_type(half_power)  # E: revealed type: Int[1]
    reveal_type(neg_zero)  # E: revealed type: Int[1]
    reveal_type(neg_zero_expr)  # E: revealed type: Int[1]

def negative_literal[N: IntVar](x: Int[N ** -1]) -> None:  # E: Tensor shape exponent must not be negative
    pass

def negative_floor_div_left[N: IntVar](x: Int[N ** (-1 // 2)]) -> None:  # E: Tensor shape exponent must not be negative
    pass

def negative_floor_div_expr[N: IntVar](x: Int[N ** ((1 - 2) // 2)]) -> None:  # E: Tensor shape exponent must not be negative
    pass

def negative_floor_div_right[N: IntVar](x: Int[N ** (1 // -2)]) -> None:  # E: Tensor shape exponent must not be negative
    pass

def ordinary_typevar[T](x: Int[2 ** T]) -> None:  # E: `T` must be an `IntVar` to be used in shape arithmetic
    pass
"#,
);

testcase!(
    test_tensor_shapes_generic_pow_overflow_is_gradual,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar

def accepts_overflow[N: IntVar](exponent: Int[N], value: Int[2 ** N]) -> None: ...

def test(exponent: Int[63], concrete: Int[7]) -> None:
    accepts_overflow(exponent, concrete)
"#,
);

testcase!(
    test_tensor_shapes_internal_dim_carrier_flows_to_size,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntTuple, IntVar, shaped_array
from typing import Any, reveal_type

@shaped_array(shape="Shape")
class Array[Shape: IntTuple = tuple[Any, ...], DType = Any]:
    shape: Shape

def take_size[N: IntVar](x: Int[N]) -> None: ...
def take_size4(x: Int[4]) -> None: ...

def shape_carrier_uses_canonical_size[N: IntVar](symbolic: Array[[N], int]) -> None:
    reveal_type(symbolic.shape[0])  # E: revealed type: Int[N]
    take_size(symbolic.shape[0])
    take_size4(symbolic.shape[0])  # E: Argument `Int[N]` is not assignable to parameter `x` with type `Int[4]`
"#,
);

testcase!(
    test_shaped_array_overload_impl_accepts_symbolic_size_return,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar, shaped_array
from typing import overload

@shaped_array(shape="Shape")
class Tensor[Shape]: ...

class Layer: ...

@overload
def dense_chain[B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    x: Tensor[[B, C, H, W]],
    layer: Layer,
    depth: Int[1],
) -> Tensor[[B, C + 32, H, W]]: ...

@overload
def dense_chain[I: IntVar, B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    x: Tensor[[B, C, H, W]],
    layer: Layer,
    depth: Int[I],
) -> Tensor[[B, C + I * 32, H, W]]: ...

def dense_chain[I: IntVar, B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    x: Tensor[[B, C, H, W]],
    layer: Layer,
    depth: Int[I],
) -> Tensor[[B, C + 32, H, W]] | Tensor[[B, C + I * 32, H, W]]: ...
"#,
);

testcase!(
    test_shaped_array_overload_impl_accepts_symbolic_size_return_with_generic_block,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar, shaped_array
from typing import Any, overload

@shaped_array(shape="Shape")
class Tensor[Shape]: ...

class Block[C: IntVar, GR: IntVar, BnC: IntVar]: ...

@overload
def dense_chain[GR: IntVar, B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    block: Block[Any, GR, Any],
    x: Tensor[[B, C, H, W]],
    depth: Int[1],
) -> Tensor[[B, C + GR, H, W]]: ...

@overload
def dense_chain[I: IntVar, GR: IntVar, B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    block: Block[Any, GR, Any],
    x: Tensor[[B, C, H, W]],
    depth: Int[I],
) -> Tensor[[B, C + I * GR, H, W]]: ...

def dense_chain[I: IntVar, GR: IntVar, B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    block: Block[Any, GR, Any],
    x: Tensor[[B, C, H, W]],
    depth: Int[I],
) -> Tensor[[B, C + GR, H, W]] | Tensor[[B, C + I * GR, H, W]]: ...
"#,
);

testcase!(
    test_tensor_shapes_nested_symbolic_size_matches_itself,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar

def with_derived[N: IntVar](first: Int[N], second: Int[N // 2]) -> None: ...

def f[N: IntVar](n: Int[N], half: Int[N // 2]) -> None:
    with_derived(n, half)
"#,
);

testcase!(
    test_tensor_shapes_nested_floor_div_negative_outer_divisor,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import reveal_type

def f[N: IntVar, M: IntVar, I: IntVar](
    positive_outer: Int[(N // 2) // 3],
    negative_outer: Int[(N // 2) // -1],
    unknown_outer: Int[(N // 2) // M],
    negative_inner_positive_outer: Int[(N // -2) // 3],
    risky_power_outer: Int[(N // 2) // (2 ** (I - 1))],
) -> None:
    reveal_type(positive_outer)  # E: revealed type: Int[(N // 6)]
    reveal_type(negative_outer)  # E: revealed type: Int[((N // 2) // -1)]
    reveal_type(unknown_outer)  # E: revealed type: Int[((N // 2) // M)]
    reveal_type(negative_inner_positive_outer)  # E: revealed type: Int[(N // -6)]
    reveal_type(risky_power_outer)  # E: revealed type: Int[((N // 2) // (2 ** (-1 + I)))]
"#,
);

testcase!(
    test_tensor_shapes_size_numeric_tower_and_literal_equivalence,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import Literal, reveal_type

def take_int(x: int) -> None: ...
def take_float(x: float) -> None: ...
def take_complex(x: complex) -> None: ...
def take_str(x: str) -> None: ...
def take_size3(x: Int[3]) -> None: ...
def take_literal3(x: Literal[3]) -> None: ...
def take_literal4(x: Literal[4]) -> None: ...
def take_huge_literal(x: Literal[100000000000000000000000000000000]) -> None: ...

def use(s: Int[3]) -> None:
    take_int(s)
    take_float(s)
    take_complex(s)  # E: Argument `Int[3]` is not assignable to parameter `x` with type `complex`
    take_str(s)  # E: Argument `Int[3]` is not assignable to parameter `x` with type `str`
    take_size3(3)
    take_size3(4)  # E: Argument `Literal[4]` is not assignable to parameter `x` with type `Int[3]`
    take_size3(True)  # E: Argument `Literal[True]` is not assignable to parameter `x` with type `Int[3]`
    take_size3(-3)  # E: Argument `Literal[-3]` is not assignable to parameter `x` with type `Int[3]`
    take_size3(1.0)  # E: Argument `float` is not assignable to parameter `x` with type `Int[3]`
    take_literal3(s)
    take_literal4(s)  # E: Argument `Int[3]` is not assignable to parameter `x` with type `Literal[4]`
    reveal_type(s * 1.5)  # E: revealed type: float

def use_symbolic[N: IntVar](s: Int[N]) -> None:
    take_int(s)
    take_float(s)
    take_complex(s)  # E: Argument `Int[N]` is not assignable to parameter `x` with type `complex`
    take_literal3(s)  # E: Argument `Int[N]` is not assignable to parameter `x` with type `Literal[3]`

def use_int(n: int) -> None:
    take_size3(n)  # E: Argument `int` is not assignable to parameter `x` with type `Int[3]`

def use_huge(s: Int[1]) -> None:
    take_size3(100000000000000000000000000000000)  # E: Argument `Literal[100000000000000000000000000000000]` is not assignable to parameter `x` with type `Int[3]`
    take_huge_literal(s - 1)  # E: Argument `Int[0]` is not assignable to parameter `x` with type `Literal[100000000000000000000000000000000]`
"#,
);

testcase!(
    test_tensor_shapes_size_annotations_reject_multiple_arguments,
    shaped_array_env(),
    r#"
from shape_extensions import Int

def bad_size(x: Int[3, 4]) -> None:  # E: Expected 1 type argument for `Int`, got 2
    pass
"#,
);

testcase!(
    test_shaped_array_unbounded_tuple_carrier_rejected,
    shaped_array_env(),
    r#"
from typing import Any, Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

@shaped_array(shape="Shape")
class DTypeFirstArray[DType, Shape]:
    def dtype(self) -> DType: ...

@shaped_array(shape="Shape")
class ArrayWithDefault[Shape, DType = int]: ...

# Unbounded tuple carriers have no concrete rank, so they cannot serve as a
# shaped-array shape carrier. Each form is rejected at the shape argument with a
# source-aware diagnostic; internally the slot degrades to an error type so that
# solving never panics or cascades.
def f_int(x: Array[tuple[int, ...], int]) -> None: ...  # E: Unbounded tuple types cannot be used as shaped-array shape carriers
def f_any(x: Array[tuple[Any, ...], int]) -> None: ...  # E: Unbounded tuple types cannot be used as shaped-array shape carriers
def f_object(x: Array[tuple[object, ...], int]) -> None: ...  # E: Unbounded tuple types cannot be used as shaped-array shape carriers
def f_unpacked_middle(x: Array[tuple[Literal[2], *tuple[int, ...]], int]) -> None: ...  # E: Unbounded tuple types cannot be used as shaped-array shape carriers
def f_nonfirst_shape(x: DTypeFirstArray[int, tuple[int, ...]]) -> None: ...  # E: Unbounded tuple types cannot be used as shaped-array shape carriers
def f_defaulted_dtype(x: ArrayWithDefault[tuple[int, ...]]) -> None: ...  # E: Unbounded tuple types cannot be used as shaped-array shape carriers

# The check is scoped to the registered shape slot. Unbounded tuple types remain
# ordinary type arguments in non-shape positions.
def non_shape_arg(x: DTypeFirstArray[tuple[int, ...], [2, 3]]) -> None:
    reveal_type(x.dtype())  # E: revealed type: tuple[int, ...]

# Wrong-arity annotations keep the ordinary arity diagnostic rather than adding
# a shape-carrier diagnostic.
def wrong_arity(x: Array[tuple[int, ...], int, str]) -> None: ...  # E: Expected 2 type arguments for `Array`, got 3
"#,
);

testcase!(
    test_shaped_array_fixed_tuple_carriers_still_accepted,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

# Fixed PEP-484 tuple carriers remain valid: only unbounded tuples are rejected.
def f(x: Array[tuple[Literal[2], Literal[3]], int]) -> None:
    reveal_type(x)  # E: revealed type: Array[[2, 3], int]

# Tuple-carrier shapes with a bounded variadic middle remain valid: only
# rank-indefinite unbounded tuple middles are rejected.
def with_typevartuple_middle[*Ts](x: Array[tuple[Literal[2], *Ts], int]) -> None: ...

# Raw generic carriers (a bare type variable in the shape slot) remain valid.
def g[S](x: Array[S, int]) -> None: ...
"#,
);

testcase!(
    test_shaped_array_compact_list_arity_error,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

# Extra args are an ordinary arity error, not compact tuple syntax.
def f(bad: Array[2, 3, int]) -> None: ...  # E: Expected a type form, got instance of `Literal[2]`  # E: Expected a type form, got instance of `Literal[3]`  # E: Expected 2 type arguments for `Array`, got 3
"#,
);

testcase!(
    test_shaped_array_compact_tuple_rejected,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(bad: Array[(2, 3), int]) -> None: ...  # E: Expected a type form, got instance of `tuple[Literal[2], Literal[3]]`
"#,
);

testcase!(
    test_shaped_array_compact_list_invalid_dim,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

# Invalid compact dims report the unresolved name without cascading to a
# non-integer dimension error.
def f(bad: Array[["rows", 3], int]) -> None: ...  # E: Could not find name `rows`
"#,
);

testcase!(
    test_shaped_array_rejects_invalid_tuple_carrier_for_inttuple_bound,
    shaped_array_env(),
    r#"
from typing import Literal
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

def f(bad: Array[tuple[str], int]) -> None: ...  # E: Invalid shaped-array shape carrier `tuple[str]`
def g(bad: Array[tuple[Literal[2], str, Literal[4]], int]) -> None: ...  # E: Invalid shaped-array shape carrier `tuple[Literal[2], str, Literal[4]]`
def h(bad: Array[tuple[Literal[1], *tuple[str], Literal[2]], int]) -> None: ...  # E: Invalid shaped-array shape carrier `tuple[Literal[1], str, Literal[2]]`
"#,
);

testcase!(
    test_shaped_array_recovers_invalid_solved_unpacked_middle,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def make[*S](shape: tuple[*S]) -> Array[tuple[Literal[2], *S, Literal[4]], int]: ...

def f(shape: tuple[str, str]) -> None:
    x = make(shape)
    reveal_type(x)  # E: revealed type: Array[[2, int, int, 4], int]
    reveal_type(x[0])  # E: revealed type: Array[[int, int, 4], int]
"#,
);

testcase!(
    test_shaped_array_renormalizes_solved_concrete_unpacked_middle,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def make[*S](shape: tuple[*S]) -> Array[tuple[Literal[1], *S, Literal[4]], int]: ...

def f(shape: tuple[Literal[2], Literal[3]]) -> None:
    x = make(shape)
    reveal_type(x)  # E: revealed type: Array[[1, 2, 3, 4], int]
    reveal_type(x[0])  # E: revealed type: Array[[2, 3, 4], int]
"#,
);

testcase!(
    test_shaped_array_compact_list_rejects_unbounded_tuple_unpack,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(bad: Array[[2, *tuple[int, ...]], int]) -> None: ...  # E: Unpacked type in `IntTuple` must use `Elements[...]`, got `tuple[int, ...]`
"#,
);

testcase!(
    test_shaped_array_compact_list_elements_rejects_non_inttuple_carrier,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(bad: Array[[2, *Elements[int]], int]) -> None: ...  # E: `Elements[...]` requires an `IntTuple` carrier, got `int`
"#,
);

testcase!(
    test_shaped_array_compact_list_requires_elements_for_inttuple_unpack,
    shaped_array_env(),
    r#"
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f[S: IntTuple](bad: Array[[2, *S], int]) -> None: ...  # E: Unpacked type in `IntTuple` must use `Elements[...]`, got `S`
"#,
);

testcase!(
    test_shaped_array_compact_list_rejects_multiple_unpacked_carriers,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f[S: IntTuple, T: IntTuple](bad: Array[[*Elements[S], *Elements[T]], int]) -> None: ...  # E: `IntTuple` can have at most one unpacked shape carrier
"#,
);

testcase!(
    test_shaped_array_elements_rejects_multiple_args,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f[S: IntTuple, T: IntTuple](bad: Array[[*Elements[S, T]], int]) -> None: ...  # E: Expected 1 type argument for `Elements`, got 2
"#,
);

testcase!(
    test_shaped_array_elements_accepts_legacy_typevar_carrier,
    shaped_array_env(),
    r#"
from typing import TypeVar
from shape_extensions import Elements, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

S = TypeVar("S", bound=IntTuple)

def f(x: Array[[*Elements[S], 3], int]) -> None: ...
"#,
);

testcase!(
    test_shaped_array_annotation_parsing,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, IntTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]:
    def __init__(self) -> None: ...
    def dtype(self) -> DType: ...

class Cpu: ...
class Gpu: ...

@shaped_array(shape="Shape")
class ArrayWithDevice[Shape: IntTuple, DType, Device: (Gpu, Cpu)]:
    def dtype(self) -> DType: ...
    def device(self) -> Device: ...

@shaped_array(shape="Shape")
class DTypeFirstArray[DType, Shape: IntTuple]:
    def dtype(self) -> DType: ...

def f(
    x: Array[[2, 3], int],
    y: Array[[], int],
    z: Array[[2, *Elements[IntTuple]], int],
    w: ArrayWithDevice[[2, 3], str, Cpu],
    w_scalar: ArrayWithDevice[[], str, Gpu],
    dtype_first: DTypeFirstArray[str, [2, 3]],
    dtype_first_scalar: DTypeFirstArray[str, []],
) -> None:
    reveal_type(x)  # E: revealed type: Array[[2, 3], int]
    reveal_type(x.dtype())  # E: revealed type: int
    reveal_type(y)  # E: revealed type: Array[[], int]
    reveal_type(y.dtype())  # E: revealed type: int
    reveal_type(z)  # E: revealed type: Array[[2, *tuple[int, ...]], int]
    reveal_type(z.dtype())  # E: revealed type: int
    reveal_type(w)  # E: revealed type: ArrayWithDevice[[2, 3], str, Cpu]
    reveal_type(w.dtype())  # E: revealed type: str
    reveal_type(w.device())  # E: revealed type: Cpu
    reveal_type(w_scalar)  # E: revealed type: ArrayWithDevice[[], str, Gpu]
    reveal_type(w_scalar.dtype())  # E: revealed type: str
    reveal_type(w_scalar.device())  # E: revealed type: Gpu
    reveal_type(dtype_first)  # E: revealed type: DTypeFirstArray[str, [2, 3]]
    reveal_type(dtype_first.dtype())  # E: revealed type: str
    reveal_type(dtype_first_scalar)  # E: revealed type: DTypeFirstArray[str, []]
    reveal_type(dtype_first_scalar.dtype())  # E: revealed type: str

def g(x: Array) -> None:
    reveal_type(x)  # E: revealed type: Array

def bad_arg_count(x: ArrayWithDevice[[2, 3], int]) -> None:  # E: Expected 3 type arguments for `ArrayWithDevice`, got 2
    pass
"#,
);

testcase!(
    test_shaped_array_indexing_and_bare_values,
    shaped_array_env(),
    r#"
from shape_extensions import IntTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]:
    def __init__(self) -> None: ...
    def dtype(self) -> DType: ...

def annotations(concrete: Array[[2, 3], int], scalar: Array[[], int], shapeless: Array) -> None:
    reveal_type(concrete[0])  # E: revealed type: Array[[3], int]
    reveal_type(concrete[:])  # E: revealed type: Array[[2, 3], int]
    reveal_type(concrete[0].dtype())  # E: revealed type: int
    scalar[0]  # E: Cannot index scalar tensor (rank 0)
    reveal_type(shapeless)  # E: revealed type: Array
    reveal_type(shapeless[0])  # E: revealed type: Array[tuple[Unknown, ...], Unknown]
    reveal_type(shapeless[None])  # E: revealed type: Array[[1, *tuple[int, ...]], Unknown]
    reveal_type(shapeless[None, ...])  # E: revealed type: Array[[1, *tuple[int, ...]], Unknown]

def accepts_precise(x: Array[[2, 3], int]) -> None:
    pass

def shapeless_is_gradual(shapeless: Array) -> None:
    accepts_precise(shapeless)

def values() -> None:
    value = Array()
    reveal_type(value)  # E: revealed type: Array[Unknown, Unknown]
    reveal_type(value[0])  # E: revealed type: Array[tuple[Unknown, ...], Unknown]

def index_preserves_dtype(concrete: Array[[2, 3], int]) -> Array[[3], int]:
    return concrete[0]
"#,
);

testcase!(
    test_shaped_array_slice_bound_kind_recovery,
    shaped_array_env(),
    r#"
from typing import assert_type, reveal_type
from shape_extensions import Int, IntTuple, IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

def ordinary_typevar[T](x: Array[[10], int], t: T) -> None:
    reveal_type(x[t:])  # E: revealed type: Array[[int], int]
    reveal_type(x[:t])  # E: revealed type: Array[[int], int]
    reveal_type(x[::t])  # E: revealed type: Array[[int], int]

def ordinary_paramspec[**P](x: Array[[10], int]) -> None:
    reveal_type(x[P:])  # E: revealed type: Array[[int], int]
    reveal_type(x[:P])  # E: revealed type: Array[[int], int]
    reveal_type(x[::P])  # E: revealed type: Array[[int], int]

def ordinary_typevartuple[*Ts](x: Array[[10], int]) -> None:
    reveal_type(x[Ts:])  # E: revealed type: Array[[int], int]
    reveal_type(x[:Ts])  # E: revealed type: Array[[int], int]
    reveal_type(x[::Ts])  # E: revealed type: Array[[int], int]

def intvar[N: IntVar](x: Array[[10], int], n: Int[N]) -> None:
    start: Array[[10 - N], int] = x[n:]
    stop: Array[[N], int] = x[:n]
    step: Array[[(10 + N - 1) // N], int] = x[::n]
    negative: Array[[N + 1], int] = x[-(n + 1):]
    assert_type(x[::- (n + 1)], Array[[(8 - N) // (-1 * (N + 1))], int])
"#,
);

testcase!(
    test_shaped_array_advanced_index_broadcast_and_placement,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

def f(
    x: Array[[10, 20, 30, 40], int],
    row: Array[[3], int],
    grid: Array[[2, 1], int],
    bad: Array[[4], int],
    scalar: Array[[], int],
    one_dimensional: Array[[10], int],
    pair: tuple[int, int],
    tuple_key: tuple[None, Array[[3], int]],
    unbounded: tuple[int, ...],
    gradual_list: list[int],
) -> None:
    reveal_type(x[pair])  # E: revealed type: Array[[30, 40], int]
    reveal_type(x[pair, grid])  # E: revealed type: Array[[2, 2, 30, 40], int]
    reveal_type(x[(pair,)])  # E: revealed type: Array[[2, 20, 30, 40], int]
    reveal_type(x[()])  # E: revealed type: Array[[10, 20, 30, 40], int]
    reveal_type(x[[0, 1]])  # E: revealed type: Array[[2, 20, 30, 40], int]
    reveal_type(x[[]])  # E: revealed type: Array[[0, 20, 30, 40], int]
    reveal_type(x[gradual_list])  # E: revealed type: Array[[int, 20, 30, 40], int]
    reveal_type(x[[*gradual_list]])  # E: revealed type: Array[[int, 20, 30, 40], int]
    reveal_type(x[tuple_key])  # E: revealed type: Array[[1, 3, 20, 30, 40], int]
    reveal_type(x[unbounded])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[(unbounded,)])  # E: revealed type: Array[[int, 20, 30, 40], int]
    reveal_type(x[gradual_list, grid])  # E: revealed type: Array[[2, int, 30, 40], int]
    reveal_type(x[unbounded, grid])  # E: revealed type: Array[[2, int, 30, 40], int]
    reveal_type(x[row, grid])  # E: revealed type: Array[[2, 3, 30, 40], int]
    reveal_type(x[row, :, grid])  # E: revealed type: Array[[2, 3, 20, 40], int]
    reveal_type(x[row, 0, grid])  # E: revealed type: Array[[2, 3, 40], int]
    reveal_type(x[0, row])  # E: revealed type: Array[[3, 30, 40], int]
    reveal_type(x[:, 0, row])  # E: revealed type: Array[[10, 3, 40], int]
    reveal_type(x[0, :, row])  # E: revealed type: Array[[20, 3, 40], int]
    reveal_type(x[0, ..., row])  # E: revealed type: Array[[20, 30, 3], int]
    reveal_type(x[:, row, :, 0])  # E: revealed type: Array[[10, 3, 30], int]
    reveal_type(x[:, row, ..., grid, :])  # E: revealed type: Array[[2, 3, 10, 40], int]
    reveal_type(x[row, ..., grid])  # E: revealed type: Array[[2, 3, 20, 30], int]
    reveal_type(x[(0, 1, 2), grid])  # E: revealed type: Array[[2, 3, 30, 40], int]
    reveal_type(x[scalar, scalar])  # E: revealed type: Array[[30, 40], int]
    x[(0, 1, 2), bad]  # E: Cannot broadcast dimension Int[3] with dimension Int[4] at position 0
    one_dimensional[(0, 1), bad]  # E: Too many indices for tensor: got 2, expected at most 1
"#,
);

testcase!(
    test_shaped_array_advanced_index_frontend_fallbacks,
    shaped_array_env(),
    r#"
from typing import Any, Literal, reveal_type
from types import EllipsisType
from shape_extensions import Int, IntTuple, IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

@shaped_array(shape="Shape")
class ArrayWithDevice[Shape: IntTuple, DType, Device]: ...

class Unsupported: ...

def fallbacks[T, *Ts](
    x: Array[[10, 20, 30, 40], int],
    integer_index: Array[[3], int],
    index_with_device: ArrayWithDevice[[3], int, str],
    bool_index: Array[[3], bool],
    float_index: Array[[3], float],
    str_index: Array[[3], str],
    any_dtype_index: Array[[3], Any],
    unsupported_index: Array[[3], Unsupported],
    any_index: Any,
    mixed: int | str,
    strings: list[str],
    anys: list[Any],
    bools: list[bool],
    raw: list,
    nested: list[list[int]],
    bool_literal: Literal[True],
    unpacked: tuple[*Ts],
    stored_slice: slice,
    stored_ellipsis: EllipsisType,
    slice_key: tuple[int, slice],
    ellipsis_key: tuple[int, EllipsisType],
    unconstrained: T,
    none_index: None,
) -> None:
    reveal_type(x[integer_index])  # E: revealed type: Array[[3, 20, 30, 40], int]
    reveal_type(x[index_with_device])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[bool_index])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[float_index])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[str_index])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[any_dtype_index])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[unsupported_index])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[any_index])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[mixed])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[strings])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[[*strings]])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[anys])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[bools])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[raw])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[nested])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[True])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[bool_literal])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[unpacked])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[(unpacked,)])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[unconstrained])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[stored_slice])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[stored_ellipsis])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[0, stored_slice])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[0, stored_ellipsis])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[slice_key])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[ellipsis_key])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[none_index])  # E: revealed type: Array[[1, 10, 20, 30, 40], int]

def int_sequence[N: IntVar](
    x: Array[[10, 20, 30, 40], int],
    pair: tuple[Int[N], int],
) -> None:
    reveal_type(x[pair])  # E: revealed type: Array[[30, 40], int]
    reveal_type(x[(pair,)])  # E: revealed type: Array[[2, 20, 30, 40], int]
"#,
);

testcase!(
    test_shaped_array_multi_axis_slice_bound_kind_recovery,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import Int, IntTuple, IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

def ordinary_typevar[T](x: Array[[10, 20], int], t: T) -> None:
    reveal_type(x[t:, :])  # E: revealed type: Array[[int, 20], int]

def ordinary_paramspec[**P](x: Array[[10, 20], int]) -> None:
    reveal_type(x[:, P:])  # E: revealed type: Array[[10, int], int]

def intvar[N: IntVar](x: Array[[10, 20], int], n: Int[N]) -> None:
    start: Array[[10 - N, 20], int] = x[n:, :]
    step: Array[[(10 + N - 1) // N, 20], int] = x[::n, :]

def unclassifiable_step(x: Array[[10, 20], int], bad_step: str) -> None:
    # A supplied invalid step is gradual; unlike an omitted step, it is not identity.
    reveal_type(x[::bad_step, :])  # E: revealed type: Array[[int, 20], int]
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_indexing_keeps_shape_coherent,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]:
    shape: Shape
    def dtype(self) -> DType: ...

@shaped_array(shape="Shape")
class DTypeFirstArray[DType, Shape]:
    shape: Shape
    def dtype(self) -> DType: ...

def f(x: Array[[2, 3, 4], int], dtype_first: DTypeFirstArray[int, [2, 3, 4]]) -> None:
    # Integer index drops the leading dim, and `.shape` stays coherent with the
    # normal class shape field.
    reveal_type(x[0])  # E: revealed type: Array[[3, 4], int]
    reveal_type(x[0].shape)  # E: revealed type: IntTuple[3, 4]
    reveal_type(x[0].dtype())  # E: revealed type: int

    # Mixed tuple index (slice + int) and `None`/newaxis stay coherent too.
    reveal_type(x[:, 0])  # E: revealed type: Array[[2, 4], int]
    reveal_type(x[:, 0].shape)  # E: revealed type: IntTuple[2, 4]
    reveal_type(x[None])  # E: revealed type: Array[[1, 2, 3, 4], int]
    reveal_type(x[None].shape)  # E: revealed type: IntTuple[1, 2, 3, 4]

    # The shape update follows the registered shape parameter, even when it is
    # not the first type argument.
    reveal_type(dtype_first[0])  # E: revealed type: DTypeFirstArray[int, [3, 4]]
    reveal_type(dtype_first[0].shape)  # E: revealed type: IntTuple[3, 4]
    reveal_type(dtype_first[0].dtype())  # E: revealed type: int

def scalar(s: Array[[], int]) -> None:
    s[0]  # E: Cannot index scalar tensor (rank 0)
"#,
);

testcase!(
    test_shaped_array_unknown_rank_carrier_indexing_not_stale,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]:
    shape: Shape

# A raw carrier `S` has unknown rank: indexing/slicing degrade to a shapeless
# array (no diagnostic), and crucially `.shape` must NOT stale-read `S` after the
# operation -- the carrier is rewritten to the shapeless form.
def g[S](x: Array[S, int]) -> None:
    reveal_type(x[0])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[0].shape)  # E: revealed type: IntTuple
    reveal_type(x[:])  # E: revealed type: Array[tuple[Unknown, ...], int]
    reveal_type(x[:].shape)  # E: revealed type: IntTuple
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_broadcast_keeps_shape_coherent,
    shaped_array_env(),
    r#"
from typing import Any, reveal_type
from shape_extensions import broadcast, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]:
    shape: Shape
    def dtype(self) -> DType: ...
    def __add__[OtherShape: IntTuple](self, other: Array[OtherShape, DType]) -> Array[broadcast(Shape, OtherShape), DType]: ...

def f(
    x: Array[[2, 3], int],
    y: Array[[1, 3], int],
    any_dim: Array[[Any, 3], int],
    gradual_dim: Array[[int, 3], int],
) -> None:
    z = x + y
    # Broadcasting `(2, 3)` with `(1, 3)` yields `(2, 3)`, and the shape
    # parameter is rewritten so `.shape` stays coherent. DType is preserved.
    reveal_type(z)  # E: revealed type: Array[[2, 3], int]
    reveal_type(z.shape)  # E: revealed type: IntTuple[2, 3]
    reveal_type(z.dtype())  # E: revealed type: int

    z_any = x + any_dim
    reveal_type(z_any)  # E: revealed type: Array[[2, 3], int]

    z_gradual = x + gradual_dim
    reveal_type(z_gradual)  # E: revealed type: Array[[2, 3], int]
"#,
);

testcase!(
    test_shaped_array_broadcast_gradual_size_keeps_precise_dimension,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import broadcast, Int, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]:
    shape: Shape
    def __add__[OtherShape: IntTuple](self, other: Array[OtherShape, DType]) -> Array[broadcast(Shape, OtherShape), DType]: ...

def f(
    known: Array[[5, 5], int],
    gradual: Array[tuple[Int[int], Int[int]], int],
    one: Array[[1, 5], int],
    gradual_then_mismatch: Array[tuple[Int[int], Literal[4]], int],
    mismatch: Array[[5, 4], int],
) -> None:
    z = known + gradual
    reveal_type(z.shape)  # E: revealed type: IntTuple[5, 5]
    z_reverse = gradual + known
    reveal_type(z_reverse.shape)  # E: revealed type: IntTuple[5, 5]

    z_one = one + gradual
    reveal_type(z_one.shape)  # E: revealed type: IntTuple[int, 5]
    z_one_reverse = gradual + one
    reveal_type(z_one_reverse.shape)  # E: revealed type: IntTuple[int, 5]

    known + gradual_then_mismatch  # E: Cannot broadcast dimension Int[5] with dimension Int[4] at position 1
    gradual_then_mismatch + known  # E: Cannot broadcast dimension Int[4] with dimension Int[5] at position 1
    known + mismatch  # E: Cannot broadcast dimension Int[5] with dimension Int[4] at position 1
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_binds_generic,
    shaped_array_env(),
    r#"
from typing import Literal
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def use_shape[S](x: Array[S, int], shape: S) -> None: ...
def get_shape[S](x: Array[S, int]) -> S: ...

def f(
    compact_2_3: Array[[2, 3], int],
    pep484_2_3: Array[tuple[Literal[2], Literal[3]], int],
) -> None:
    shape_2_3: tuple[Literal[2], Literal[3]] = (2, 3)
    shape_2_4: tuple[Literal[2], Literal[4]] = (2, 4)
    use_shape(compact_2_3, shape_2_3)
    use_shape(pep484_2_3, shape_2_3)
    use_shape(compact_2_3, shape_2_4)  # E: Argument `tuple[Literal[2], Literal[4]]` is not assignable to parameter `shape` with type `IntTuple[2, 3]`
    out: tuple[Literal[2], Literal[3]] = get_shape(compact_2_3)
    bad: tuple[Literal[2], Literal[4]] = get_shape(compact_2_3)  # E: `IntTuple[2, 3]` is not assignable to `tuple[Literal[2], Literal[4]]`
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_generic_return_reprojection,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def make_array[S](shape: S) -> Array[S, float]: ...

def f() -> None:
    shape_2_3: tuple[Literal[2], Literal[3]] = (2, 3)
    scalar_shape: tuple[()] = ()
    reveal_type(make_array(shape_2_3))  # E: revealed type: Array[[2, 3], float]
    reveal_type(make_array(scalar_shape))  # E: revealed type: Array[[], float]
"#,
);

testcase!(
    bug = "tuple literals passed to generic shape carriers are widened before return reprojection",
    test_shaped_array_tuple_carrier_generic_return_literal_tuple_widens,
    shaped_array_env(),
    r#"
from typing import assert_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def make_array[S](shape: S) -> Array[S, float]: ...

def f() -> None:
    assert_type(make_array((2, 3)), Array[[int, int], float])
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_generic_identity_preserves_shape_and_dtype,
    shaped_array_env(),
    r#"
from typing import reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]:
    def dtype(self) -> DType: ...

def identity[S, D](x: Array[S, D]) -> Array[S, D]: ...

def f(x_2_3_int: Array[[2, 3], int]) -> None:
    reveal_type(identity(x_2_3_int))  # E: revealed type: Array[[2, 3], int]
    reveal_type(identity(x_2_3_int).dtype())  # E: revealed type: int
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_generic_preserves_unpacked_prefix,
    shaped_array_env(),
    r#"
from typing import Literal
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def get_shape[S](x: Array[S, int]) -> S: ...

def f[*Ts](x: Array[tuple[Literal[2], *Ts], int]) -> None:
    good: tuple[Literal[2], *Ts] = get_shape(x)
    bad: tuple[Literal[3], *Ts] = get_shape(x)  # E: `IntTuple[2, *Ts]` is not assignable to `tuple[Literal[3], *Ts]`
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_unpacked_middle_is_invariant,
    shaped_array_env(),
    r#"
from typing import Literal
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def use_shape[S](x: Array[S, int], shape: S) -> None: ...

def f[*Ts](
    x: Array[tuple[Literal[2], *Ts], int],
    shape_2: tuple[Literal[2], *Ts],
    shape_3: tuple[Literal[3], *Ts],
) -> None:
    use_shape(x, shape_2)
    use_shape(x, shape_3)  # E: Argument `tuple[Literal[3], *Ts]` is not assignable to parameter `shape` with type `IntTuple[2, *Ts]`
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_shape_attr_preserves_generic_carrier,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def carrier[S](x: Array[S, float]) -> None:
    reveal_type(x.shape)  # E: revealed type: S

def concrete[M: IntVar](x: Array[[2, 4, M], float]) -> None:
    reveal_type(x.shape)  # E: revealed type: tuple[Literal[2], Literal[4], Int[M]]

def unpacked_prefix[*Ts](x: Array[tuple[Literal[2], *Ts], float]) -> None:
    reveal_type(x.shape)  # E: revealed type: tuple[Literal[2], *Ts]

def typevartuple[*Shape](x: Array[tuple[*Shape], float]) -> None:
    reveal_type(x.shape)  # E: revealed type: tuple[*Shape]
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_does_not_erase_dtype,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def want_int(x: Array[[2, 3], int]) -> None: ...

def f(x_str: Array[[2, 3], str]) -> None:
    want_int(x_str)  # E: Argument `Array[[2, 3], str]` is not assignable to parameter `x` with type `Array[[2, 3], int]`
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_closed_shapes_still_check_dimensions,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def want_2_4(x: Array[[2, 4], int]) -> None: ...

def f(x_2_3: Array[[2, 3], int]) -> None:
    want_2_4(x_2_3)  # E: Argument `Array[[2, 3], int]` is not assignable to parameter `x` with type `Array[[2, 4], int]`
"#,
);

testcase!(
    bug = "closed-carrier diagnostic wording/placement is provisional until tuple<->IntTuple assignability lands",
    test_shaped_array_invalid_closed_carrier,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def want_2_3(x: Array[[2, 3], int]) -> None: ...
def want_bad(x: Array[tuple[str, str], int]) -> None: ...  # E: Invalid shaped-array shape carrier `tuple[str, str]`

# `tuple[str, str]` is not a valid shape carrier. It projects to a shapeless
# array internally; a source-aware diagnostic rejecting this form is deferred.
def f(x_bad: Array[tuple[str, str], int]) -> None:  # E: Invalid shaped-array shape carrier `tuple[str, str]`
    want_2_3(x_bad)
    want_bad(x_bad)

def g(x_2_3: Array[[2, 3], int]) -> None:
    want_bad(x_2_3)
"#,
);

testcase!(
    test_undecorated_torch_tensor_stays_ordinary,
    shaped_array_env_with_plain_torch(),
    r#"
from typing import reveal_type
from torch import Tensor

def f(x: Tensor[2, 3], y: Tensor) -> None:  # E: Expected a type form, got instance of `Literal[2]`  # E: Expected a type form, got instance of `Literal[3]`
    reveal_type(x)  # E: revealed type: Tensor[Unknown, Unknown]
    reveal_type(x[0])  # E: revealed type: Tensor[Unknown, Unknown]
    reveal_type(y)  # E: revealed type: Tensor[*tuple[Unknown, ...]]
"#,
);

testcase!(
    test_tensor_shapes_keeps_integer_type_arguments_ordinary,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntTuple, IntVar, shaped_array
from typing import TypeVar, reveal_type

T = TypeVar("T")
DefaultT = TypeVar("DefaultT", default=3)  # E: Expected a type form, got instance of `Literal[3]`

class Box[T]: ...
class DefaultBox[T = 3]: ...  # E: Expected a type form, got instance of `Literal[3]`

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType, Device]: ...

@shaped_array(shape="Shape")
class DTypeFirstArray[DType, Shape: IntTuple]: ...

class Cpu: ...
class Gpu: ...

type Image = Array[[2, 3], int, Cpu]

def ordinary_type_arguments(x: Box[3]) -> None:  # E: Expected a type form, got instance of `Literal[3]`
    pass

def shaped_array_segments(
    good: Array[[2, 3], int, Cpu],
    bad_dtype: Array[[2, 3], 3, Cpu],  # E: Expected a type form, got instance of `Literal[3]`
    bad_device: Array[[2, 3], int, 3],  # E: Expected a type form, got instance of `Literal[3]`
    bad_dtype_first: DTypeFirstArray[3, [2, 3]],  # E: Expected a type form, got instance of `Literal[3]`
    alias: Image,
) -> None:
    reveal_type(good)  # E: revealed type: Array[[2, 3], int, Cpu]
    reveal_type(alias)  # E: revealed type: Array[[2, 3], int, Cpu]

def dims[N: IntVar](concrete: Int[3], symbolic: Int[N + 1]) -> None:
    pass
"#,
);

testcase!(
    test_tensor_shapes_gradual_size,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntTuple, shaped_array
from typing import Any, assert_type, overload, reveal_type

@shaped_array(shape="Shape")
class Array[Shape: IntTuple]: ...

def take_int(x: int) -> None: ...
def take_gradual(x: Int) -> None: ...
def take_gradual_int(x: Int[int]) -> None: ...
def take_size3(x: Int[3]) -> None: ...
def take_size4(x: Int[4]) -> None: ...

@overload
def choose_size(x: Int) -> int: ...
@overload
def choose_size(x: Int[3]) -> str: ...
def choose_size(x: object) -> int | str: ...

def f(bare: Int, gint: Int[int], s3: Int[3], s4: Int[4], i: int, a: Any) -> None:
    take_gradual(s3)
    take_gradual_int(s3)
    take_size3(bare)
    take_size3(gint)
    take_gradual(i)
    take_gradual_int(i)
    take_gradual(True)  # E: Argument `Literal[True]` is not assignable to parameter `x` with type `Int[int]`
    take_gradual(MyInt())  # E: Argument `MyInt` is not assignable to parameter `x` with type `Int[int]`
    take_size3(i)  # E: Argument `int` is not assignable to parameter `x` with type `Int[3]`
    take_int(bare)
    take_size4(s3)  # E: Argument `Int[3]` is not assignable to parameter `x` with type `Int[4]`
    take_size3(s4)  # E: Argument `Int[4]` is not assignable to parameter `x` with type `Int[3]`
    # Overload pruning materializes `Any`; this proves materialization is consistent
    # with the gradual `Int` type.
    assert_type(choose_size(a), int)

class MyInt(int): ...

def shape_any(x: Array[[Any, 3]]) -> None:
    pass

def shape_int(x: Array[[int, 3]]) -> None:
    pass

def size_any(x: Int[Any]) -> None:
    pass

def size_bool(x: Int[bool]) -> None:  # E: Tensor shape dimensions must be integer literals or type variables, got `type[bool]`
    pass
"#,
);

testcase!(
    test_tensor_shapes_int_and_int_int_equivalence,
    shaped_array_env(),
    r#"
from shape_extensions import Int
from typing import Literal, assert_type, overload, reveal_type

def take_int(x: int) -> None: ...
def take_int_int(x: Int[int]) -> None: ...
def take_int3(x: Int[3]) -> None: ...

def returns_int_from_Int(x: Int[int]) -> int:
    return x

def returns_Int_from_int(x: int) -> Int[int]:
    return x

@overload
def choose_int(x: Int[3]) -> Literal["exact"]: ...
@overload
def choose_int(x: Int[int]) -> Literal["gradual"]: ...
def choose_int(x: int) -> str: ...

@overload
def choose_gradual_first(x: Int[int]) -> Literal["gradual"]: ...
@overload
def choose_gradual_first(x: Int[3]) -> Literal["exact"]: ...
def choose_gradual_first(x: int) -> str: ...

def use(cond: bool, i: int, s: Int[int], s3: Int[3], s4: Int[4], lit3: Literal[3]) -> None:
    int_from_Int: int = s
    Int_from_int: Int[int] = i
    take_int(s)
    take_int_int(i)
    # `int` and `Int[int]` are mutually assignable (above), but each keeps its own
    # representation. See `test_tensor_shapes_int_and_int_int_not_assert_type_equal`
    # for the `assert_type` distinction between them.
    assert_type(i, int)
    assert_type(s, Int[int])
    assert_type(choose_int(s3), Literal["exact"])
    # `Literal[3]` intentionally participates in the same exact-shape
    # equivalence class as `Int[3]`.
    assert_type(choose_int(lit3), Literal["exact"])
    assert_type(choose_int(i), Literal["gradual"])
    assert_type(choose_int(s4), Literal["gradual"])
    assert_type(choose_gradual_first(s), Literal["gradual"])

    int3_from_literal: Int[3] = lit3
    take_int3(lit3)
    assert_type(lit3, Int[3])

    int3_from_int: Int[3] = i  # E: `int` is not assignable to `Int[3]`
    take_int3(i)  # E: Argument `int` is not assignable to parameter `x` with type `Int[3]`

    inferred_union = i if cond else s
    assert_type(inferred_union, int | Int[int])
"#,
);

testcase!(
    test_tensor_shapes_int_and_int_int_not_assert_type_equal,
    shaped_array_env(),
    r#"
from shape_extensions import Int
from typing import assert_type

def f(i: int, s: Int[int]) -> None:
    # `int` and `Int[int]` are mutually assignable, but they are distinct type
    # representations. `assert_type` checks the representation, not just the
    # subtyping order, so it treats them as non-equivalent.
    assert_type(i, Int[int])  # E: assert_type
    assert_type(s, int)  # E: assert_type
    # Each is equivalent to its own representation.
    assert_type(i, int)
    assert_type(s, Int[int])
"#,
);

testcase!(
    test_tensor_shapes_int_satisfies_fresh_symbolic_size,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import reveal_type

def take_symbolic[N: IntVar](x: Int[N]) -> Int[N]: ...
def same_symbolic[N: IntVar](x: Int[N], y: Int[N]) -> Int[N]: ...
def take_size3(x: Int[3]) -> None: ...

def f(i: int, s3: Int[3]) -> None:
    reveal_type(take_symbolic(i))  # E: revealed type: Int[int]
    reveal_type(take_symbolic(3))  # E: revealed type: Int[3]
    reveal_type(take_symbolic(s3))  # E: revealed type: Int[3]
    take_size3(i)  # E: Argument `int` is not assignable to parameter `x` with type `Int[3]`
    take_size3(3)
    same_symbolic(s3, i)  # E: Argument `int` is not assignable to parameter `y` with type `Int[3]`
    # Two `int`s into a repeated symbolic dimension: the first pins N gradual, the
    # second matches that gradual bound (accepted).
    same_symbolic(i, i)
"#,
);

testcase!(
    test_tensor_shapes_gradual_size_satisfies_fresh_symbolic_size,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import assert_type

def take_symbolic[N: IntVar](x: Int[N]) -> Int[N]: ...

# A gradual `Int` (bare `Int` == `Int[int]`) flowing into a fresh symbolic
# `Int[N]` resolves to the gradual size: the unconstrained `IntVar` defaults
# to gradual rather than leaking an unsolved `Var`.
def f(s: Int) -> None:
    assert_type(take_symbolic(s), Int)
"#,
);

testcase!(
    bug = "int eagerly pins a repeated IntVar to gradual, so argument order flips accept/reject",
    test_tensor_shapes_symvar_inference_is_order_dependent,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar

def same_symbolic[N: IntVar](x: Int[N], y: Int[N]) -> Int[N]: ...

# An `int` argument eagerly pins the fresh `N` to the gradual size, so the later
# concrete `Int[3]` is accepted; the mirror-image call correctly rejects the
# `int`. The two orders should agree once `int` accumulates a gradual bound
# instead of pinning it (see the `IntVar` eager-pin note in solver/subset.rs).
def f(i: int, s3: Int[3]) -> None:
    same_symbolic(i, s3)
    same_symbolic(s3, i)  # E: Argument `int` is not assignable to parameter `y` with type `Int[3]`
"#,
);

testcase!(
    test_tensor_shapes_numpy_shaped_api_accepts_int_lengths,
    {
        let mut env = shaped_array_env();
        env.add_with_path(
            "numpy",
            "numpy.pyi",
            r#"
from shape_extensions import Int, IntTuple, IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType = int]: ...

def arange[N: IntVar](stop: Int[N]) -> Array[[N], int]: ...
def full[N: IntVar](shape: Int[N], fill_value: float) -> Array[[N], float]: ...
def take_size3(x: Int[3]) -> None: ...
"#,
        );
        env
    },
    r#"
import numpy as np

def f(targets: list[int], n_points: int) -> None:
    np.arange(len(targets))
    np.full(n_points - 1, 0.0)
    np.take_size3(n_points)  # E: Argument `int` is not assignable to parameter `x` with type `Int[3]`
"#,
);

testcase!(
    test_tensor_shapes_len_carries_first_dimension,
    {
        let mut env = shaped_array_env();
        env.add_with_path(
            "numpy",
            "numpy.pyi",
            r#"
from shape_extensions import Int, IntTuple, IntVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType = int]:
    def __len__[N: IntVar](self: Array[[N], DType]) -> Int[N]: ...

def arange[N: IntVar](stop: Int[N]) -> Array[[N], int]: ...
def zeros[N: IntVar](shape: Int[N]) -> Array[[N], int]: ...
"#,
        );
        env
    },
    r#"
import numpy as np
from shape_extensions import Int
from typing import assert_type

def f(a: np.Array[[5], int], xs: list[int]) -> None:
    # `len()` returns `Array.__len__`'s `Int[N]` result (a subtype of `int`), so it
    # carries the first dimension and flows into shape-DSL arithmetic downstream.
    assert_type(len(a), Int[5])
    assert_type(np.arange(len(a)), np.Array[[5], int])
    # A plain `list.__len__` returns `int`, so `len()` stays gradual there.
    assert_type(len(xs), int)
"#,
);

testcase!(
    test_tensor_shapes_size_bound_defaults,
    shaped_array_env(),
    r#"
from shape_extensions import Int

class SizeDefault[N: Int = 3]: ...
class SizeIntDefault[N: Int[int] = 3]: ...
class SizeHuge[N: Int]: ...

def f() -> None:
    # `N: Size` is an ordinary `TypeVar`, so an integer literal is a value, not a
    # type form; it is no longer parsed as a symbolic shape dimension.
    size: SizeDefault[3] = SizeDefault()  # E: Expected a type form, got instance of `Literal[3]`
    size_int: SizeIntDefault[3] = SizeIntDefault()  # E: Expected a type form, got instance of `Literal[3]`
    huge: SizeHuge[100000000000000000000000000000000] = SizeHuge()  # E: Expected a type form, got instance of `Literal[100000000000000000000000000000000]`
"#,
);

testcase!(
    test_tensor_shapes_gradual_size_through_size_bound_typevar,
    shaped_array_env(),
    r#"
from shape_extensions import Int
from typing import reveal_type

def id_size[N: Int](x: N) -> N: ...
def takes_size_bound[N: Int](x: N) -> None: ...
def takes_size(x: Int) -> None: ...
def takes_size3(x: Int[3]) -> None: ...

def pass_size_bound_to_gradual[N: Int](x: N) -> None:
    takes_size(x)

def f(s: Int, s3: Int[3]) -> None:
    reveal_type(id_size(s))  # E: revealed type: Int[int]
    reveal_type(id_size(s3))  # E: revealed type: Int[3]
    takes_size_bound(s)
    takes_size_bound(s3)
    takes_size(id_size(s3))
    takes_size3(id_size(s3))
"#,
);

testcase!(
    test_tensor_shapes_recanonicalizes_expanded_dimension_roots,
    shaped_array_env(),
    r#"
from collections.abc import Callable
from shape_extensions import Int, IntVar

def take_product[X: IntVar, Y: IntVar](
    left: Int[X],
    right: Int[Y],
    value: Int[X * Y],
) -> None: ...

def make_product[X: IntVar, Y: IntVar](left: Int[X], right: Int[Y]) -> Int[X * Y]: ...

def f[A: IntVar, B: IntVar, C: IntVar, D: IntVar](
    left: Int[A + B],
    right: Int[C + D],
    expanded: Int[A * C + A * D + B * C + B * D],
) -> None:
    # The want root needs another pass after X and Y expand to sums.
    take_product(left, right, expanded)
    # Callable parameter matching solves X and Y before comparing the return type.
    check: Callable[
        [Int[A + B], Int[C + D]],
        Int[A * C + A * D + B * C + B * D],
    ] = make_product
"#,
);

testcase!(
    test_tensor_shapes_recanonicalizes_mixed_dimension_roots,
    shaped_array_env(),
    r#"
from collections.abc import Callable
from shape_extensions import Int, IntVar

class Box[N: IntVar]:
    def get(self) -> Int[N]: ...

def take_box[X: IntVar, Y: IntVar](
    left: Int[X],
    right: Int[Y],
    value: Box[X * Y],
) -> None: ...

def make_box[X: IntVar, Y: IntVar](left: Int[X], right: Int[Y]) -> Box[X * Y]: ...

def f[A: IntVar, B: IntVar, C: IntVar, D: IntVar, Q: IntVar](
    left: Int[A + B],
    right: Int[C + D],
    box: Box[Q],
) -> None:
    quantified_want: Callable[[Int[A + B], Int[C + D]], Box[Q]] = make_box  # E: Shape dimension mismatch: expected Int[Q], got Int[((((A * C) + (A * D)) + (B * C)) + (B * D))]
    take_box(left, right, box)  # E: Shape dimension mismatch: expected Int[((((A * C) + (A * D)) + (B * C)) + (B * D))], got Int[Q]
"#,
);

testcase!(
    test_tensor_shapes_size_int_is_canonical_when_inferred,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar

def take_size[N: IntVar](x: Int[N]) -> None: ...
def take_size3(x: Int[3]) -> None: ...

def f[M: IntVar](x: int | Int[M]) -> None:
    take_size(x)

def g(x: int) -> None:
    take_size(x)
    take_size3(3)
    take_size3(x)  # E: Argument `int` is not assignable to parameter `x` with type `Int[3]`

class C[N: IntVar]:
    def __init__(self, x: Int[N]) -> None: ...

def h(x: int) -> None:
    C(x)
    C(int(x))  # E: Unnecessary `int()` call; argument is already of type `int`
"#,
);

testcase!(
    test_tensor_shapes_keeps_ordinary_literal_arithmetic_int,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import reveal_type

def ordinary_literals() -> None:
    reveal_type(1 + 2)  # E: revealed type: int
    reveal_type(1 - 2)  # E: revealed type: int
    reveal_type(2 * 3)  # E: revealed type: int
    reveal_type(5 // 2)  # E: revealed type: int
    reveal_type(2 ** 3)  # E: revealed type: int
    total = 1
    total += 2
    reveal_type(total)  # E: revealed type: int

def dim_literals[N: IntVar](x: Int[N]) -> None:
    reveal_type(x + 1)  # E: revealed type: Int[(1 + N)]
    reveal_type(1 + x)  # E: revealed type: Int[(1 + N)]

def ordinary_typevar_value[T: int](x: T) -> None:
    reveal_type(x + 1)  # E: revealed type: int

def ordinary_unrestricted_typevar_value[T](x: T) -> None:
    x + 1  # E: `+` is not supported between `T` and `Literal[1]`
"#,
);

testcase!(
    test_tensor_shapes_int_falls_back_to_int_behavior,
    shaped_array_env(),
    r#"
from shape_extensions import Int, IntVar
from typing import Any, SupportsIndex, assert_type, reveal_type

def take_index(x: SupportsIndex) -> None: ...
def keep_symbolic[M: IntVar](value: Int[M]) -> Int[M]: ...

def use[N: IntVar, M: IntVar](x: Int[N], y: Int[3], e3: Int[3], m: Int[M], i: int, f: float) -> None:
    reveal_type(x + 1)  # E: revealed type: Int[(1 + N)]
    reveal_type(x - 1)  # E: revealed type: Int[(-1 + N)]
    reveal_type(x * 2)  # E: revealed type: Int[(2 * N)]
    reveal_type(x // 2)  # E: revealed type: Int[(N // 2)]

    reveal_type(x + f)  # E: revealed type: float
    reveal_type(f + x)  # E: revealed type: float
    reveal_type(x / 2)  # E: revealed type: float
    reveal_type(x % 2)  # E: revealed type: int

    reveal_type(x ** 0)  # E: revealed type: Int[1]
    reveal_type(x ** 1)  # E: revealed type: Int[N]
    reveal_type(x ** 2)  # E: revealed type: Int[(N ** 2)]
    reveal_type(x ** e3)  # E: revealed type: Int[(N ** 3)]
    reveal_type(y ** 2)  # E: revealed type: Int[9]
    reveal_type(y ** e3)  # E: revealed type: Int[27]
    reveal_type(x ** -1)  # E: revealed type: float
    neg = y - 4
    reveal_type(neg)  # E: revealed type: Int[-1]
    reveal_type(x ** neg)  # E: revealed type: float
    reveal_type(x ** f)  # E: revealed type: float
    reveal_type(x ** i)  # E: revealed type: Unknown
    assert_type(x ** m, Any)
    assert_type(2 ** x, Any)
    reveal_type(2 ** y)  # E: revealed type: Int[8]
    reveal_type(x ** 100000000000000000000000000000000)  # E: revealed type: int
    flowed = keep_symbolic(neg)
    reveal_type(flowed)  # E: revealed type: Int[-1]
    reveal_type(2 ** flowed)  # E: revealed type: float
    reveal_type(flowed ** 0)  # E: revealed type: Int[1]

    reveal_type(x.bit_length())  # E: revealed type: int
    reveal_type(x.real)  # E: revealed type: int
    reveal_type(x.numerator)  # E: revealed type: int
    reveal_type(x.__index__())  # E: revealed type: int
    reveal_type(hash(x))  # E: revealed type: int

    reveal_type(x == i)  # E: revealed type: bool
    reveal_type(x < i)  # E: revealed type: bool
    reveal_type(x >= 0)  # E: revealed type: bool

    take_index(x)
    range(x)
    [1, 2, 3][x]

    reveal_type(+x)  # E: revealed type: int
    reveal_type(-x)  # E: revealed type: int
    reveal_type(~x)  # E: revealed type: int
"#,
);

testcase!(
    test_legacy_intvar_treated_as_intvar,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Int, IntVar
from torch import Tensor
from typing import Generic, assert_type, reveal_type

N = IntVar("N")
M = IntVar("M")

class Box(Generic[N]): ...

def f(n: Int[N], shifted: Int[N + 1], x: Tensor[[N, M]], shifted_x: Tensor[[N + 1, M]], y: Box[N]) -> None:
    reveal_type(n)  # E: revealed type: Int[N]
    assert_type(shifted, Int[N + 1])
    reveal_type(x)  # E: revealed type: Tensor[[N, M]]
    assert_type(shifted_x, Tensor[[N + 1, M]])
    reveal_type(y)  # E: revealed type: Box[N]
"#,
);

testcase!(
    test_intvar_type_parameter_bound,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Int, Elements, IntTuple, IntVar
from shape_extensions import IntVar as SV
import shape_extensions
import shape_extensions as se
from torch import Tensor
from typing import reveal_type

class SymBox[N: IntVar]: ...

def identity_alias[N: SV](x: Int[N]) -> Int[N]:
    return x

def identity_module[N: shape_extensions.IntVar](x: Int[N]) -> Int[N]:
    return x

def identity_module_alias[N: se.IntVar](x: Int[N]) -> Int[N]:
    return x

def shape[N: IntVar, M: IntVar, Shape: IntTuple](
    n: Int[N],
    x: Tensor[[N]],
    size: IntTuple[N, M],
    packed: Tensor[[*Elements[Shape], N]],
    boxed: SymBox[N],
) -> None:
    reveal_type(n)  # E: revealed type: Int[N]
    reveal_type(x)  # E: revealed type: Tensor[[N]]
    reveal_type(packed)  # E: revealed type: Tensor[[*Elements[Shape], N]]
    reveal_type(boxed)  # E: revealed type: SymBox[N]

def default_ok[N: IntVar, M: IntVar = N](x: Int[M]) -> None:
    pass

def default_expr_ok[N: IntVar, M: IntVar = N + 1](x: Int[M]) -> None:
    pass

type Shape[N: IntVar] = Tensor[[N]]
type Packed[Shape: IntTuple, N: IntVar] = Tensor[[*Elements[Shape], N]]
type OrdinaryAlias[T, N: IntVar] = tuple[T, Int[N]]

def alias_specialization[N: IntVar, ShapeT: IntTuple](
    x: Shape[N],
    packed: Packed[ShapeT, N],
    ordinary: OrdinaryAlias[int, N],
) -> None:
    reveal_type(x)  # E: revealed type: Tensor[[N]]
    reveal_type(packed)  # E: revealed type: Tensor[[*Elements[ShapeT], N]]
    reveal_type(ordinary)  # E: revealed type: tuple[int, Int[N]]
"#,
);

testcase!(
    test_intvar_type_parameter_bound_through_reexport,
    reexporting_shape_extensions_env(),
    r#"
from reexport import Int, IntVar
from torch import Tensor
from typing import assert_type

def bound[N: IntVar](n: Int[N], x: Tensor[[N]]) -> None:
    assert_type(n, Int[N])
    assert_type(x, Tensor[[N]])
"#,
);

testcase!(
    test_intvar_type_parameter_bound_through_reexport_alias,
    reexporting_shape_extensions_env(),
    r#"
from reexport import Int
from reexport import IntVar as SV
from torch import Tensor
from typing import assert_type

def bound[N: SV](n: Int[N], x: Tensor[[N]]) -> None:
    assert_type(x, Tensor[[N]])
"#,
);

testcase!(
    test_intvar_type_parameter_bound_through_assignment_alias,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Int, IntVar
from torch import Tensor
from typing import assert_type

MyIntVar = IntVar

def bound[N: MyIntVar](n: Int[N], x: Tensor[[N]]) -> None:
    assert_type(x, Tensor[[N]])
"#,
);

testcase!(
    test_reexported_intvar_still_rejected_as_typevar_bound,
    reexporting_shape_extensions_env(),
    r#"
from reexport import IntVar
from typing import TypeVar

Bad = TypeVar("Bad", bound=IntVar)  # E: `IntVar` cannot be used as a TypeVar bound
"#,
);

testcase!(
    test_intvar_rejected_in_ordinary_type_positions,
    shaped_array_env_with_shaped_torch(),
    r#"
from collections.abc import Callable
from shape_extensions import Int, IntVar
from torch import Tensor
from typing import Generic, Optional, TypeAlias, TypeAliasType, TypeVar

LegacyN = IntVar("LegacyN")
OrdinaryT = TypeVar("OrdinaryT")
OrdinaryDefault = TypeVar("OrdinaryDefault", default=LegacyN)  # E: `LegacyN` is an `IntVar` and cannot be used as an ordinary type
BadSymDefault = IntVar("BadSymDefault", default=OrdinaryT)  # E: `OrdinaryT` must be an `IntVar` to be used as a shape dimension
BadOperatorDefault = IntVar("BadOperatorDefault", default=1 | 2)  # E: Unsupported operator `|` in tensor shape dimension
IntDefault = IntVar("IntDefault", default=int)

class LegacyBox(Generic[LegacyN]): ...
class Box[T]: ...

def legacy_shape(n: Int[LegacyN], x: Tensor[[LegacyN]]) -> None:
    pass

def legacy_invalid(
    x: LegacyN,  # E: `LegacyN` is an `IntVar` and cannot be used as an ordinary type
    y: list[LegacyN],  # E: `LegacyN` is an `IntVar` and cannot be used as an ordinary type
    z: Box[LegacyN],  # E: `LegacyN` is an `IntVar` and cannot be used as an ordinary type
) -> None:
    pass

def invalid[N: IntVar](
    x: N,  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    y: list[N],  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    z: Box[N],  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    t: type[N],  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    u: N | int,  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    nested: int | (str | N),  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    optional: Optional[N],  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    c: Callable[[], N],  # E: `N` is an `IntVar` and cannot be used as an ordinary type
) -> None:
    pass

type Alias[N: IntVar] = N  # E: `N` is an `IntVar` and cannot be used as an ordinary type
type AliasUnion[N: IntVar] = N | int  # E: `N` is an `IntVar` and cannot be used as an ordinary type
LegacyAlias: TypeAlias = LegacyN | int  # E: `LegacyN` is an `IntVar` and cannot be used as an ordinary type
CallAlias = TypeAliasType("CallAlias", LegacyN | int, type_params=(LegacyN,))  # E: `LegacyN` is an `IntVar` and cannot be used as an ordinary type

def default_bad[T, N: IntVar = T](x: Int[N]) -> None:  # E: `T` must be an `IntVar` to be used as a shape dimension
    pass

def default_int[N: IntVar = int](x: Int[N]) -> None:
    pass
"#,
);

testcase!(
    test_ordinary_typevar_shape_arithmetic_is_rejected,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D, Int, IntTuple
from torch import Tensor
from typing import Generic, TypeVar

LegacyN = TypeVar("LegacyN")

class LegacyBox(Generic[LegacyN]):
    legacy_tensor: Tensor[[LegacyN + 1]]  # E: `LegacyN` must be an `IntVar` to be used in shape arithmetic

def invalid[N](
    dim: Int[N + 1],  # E: `N` must be an `IntVar` to be used in shape arithmetic
    tensor: Tensor[[N + 1]],  # E: `N` must be an `IntVar` to be used in shape arithmetic
    reversed_tensor: Tensor[[1 + N]],  # E: `N` must be an `IntVar` to be used in shape arithmetic
    tuple_shape: Tensor[IntTuple[N + 1]],  # E: `N` must be an `IntVar` to be used in shape arithmetic
    negated: Tensor[[-N]],  # E: `N` must be an `IntVar` to be used in shape arithmetic
    bracket_launder: Tensor[[D[N] + 1]],  # E: `N` must be an `IntVar` to be used in shape arithmetic
    call_launder: Tensor[[D(N) // 2]],  # E: `N` must be an `IntVar` to be used in shape arithmetic
    inner_launder: Tensor[[D[N + 1]]],  # E: `N` must be an `IntVar` to be used in shape arithmetic
) -> None:
    pass
"#,
);

testcase!(
    test_kind_errors_recover_with_gradual_components,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Int, IntVar
from torch import Tensor
from typing import Any, assert_type, reveal_type

def ordinary_type_recovery[N: IntVar](
    x: list[N],  # E: `N` is an `IntVar` and cannot be used as an ordinary type
    y: N | int,  # E: `N` is an `IntVar` and cannot be used as an ordinary type
) -> None:
    reveal_type(x)  # E: revealed type: list[Unknown]
    reveal_type(y)  # E: revealed type: int | Unknown

def symbolic_int_recovery[T](
    dim: Int[T],  # E: `T` must be an `IntVar` to be used as a shape dimension
    tensor: Tensor[[T, 3]],  # E: `T` must be an `IntVar` to be used as a shape dimension
) -> None:
    assert_type(dim, Int[Any])
    assert_type(tensor, Tensor[[Any, 3]])
"#,
);

testcase!(
    test_intvar_shape_arithmetic_is_accepted,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Int, IntVar
from torch import Tensor
from typing import assert_type

LegacyN = IntVar("LegacyN")

def pep695[N: IntVar](dim: Int[N + 1], tensor: Tensor[[N + 1]], negated: Tensor[[-N]]) -> None:
    pass

def legacy(dim: Int[LegacyN + 1], tensor: Tensor[[LegacyN + 1]], negated: Tensor[[-LegacyN]]) -> None:
    assert_type(dim, Int[LegacyN + 1])
    assert_type(tensor, Tensor[[LegacyN + 1]])
    assert_type(negated, Tensor[[-LegacyN]])
"#,
);

testcase!(
    test_intvar_special_form_is_only_a_kind_marker,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import IntVar
from typing import TypeVar

def ok[N: IntVar](x: object) -> None:
    pass

x: IntVar = 1  # E: `Literal[1]` is not assignable to `IntVar`
y: IntVar[int] = 1  # E: Expected 0 type arguments for `IntVar`, got 1  # E: `Literal[1]` is not assignable to `IntVar`
T = TypeVar("T", bound=IntVar)  # E: `IntVar` cannot be used as a TypeVar bound
U = TypeVar("U", IntVar, int)  # E: `IntVar` cannot be used as a TypeVar constraint
V = TypeVar("V", default=IntVar)  # E: `IntVar` cannot be used as a TypeVar default

def bad_constraint[T: (IntVar, int)](x: T) -> None:  # E: `IntVar` cannot be used as a TypeVar constraint
    pass
"#,
);

testcase!(
    test_intvar_class_type_parameter_accepts_dimension_expressions,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Int, IntVar
from typing import Generic, assert_type, reveal_type

class ExplicitBox[N: IntVar]: ...

N = IntVar("N")
M = IntVar("M")

class LegacyBox(Generic[N]): ...

def explicit[N: IntVar](x: ExplicitBox[N + 1]) -> None:
    assert_type(x, ExplicitBox[N + 1])

def legacy(x: LegacyBox[N + M]) -> None:
    reveal_type(x)  # E: revealed type: LegacyBox[Int[(N + M)]]

def explicit_literals[S: IntVar](literal: ExplicitBox[3], symbolic: ExplicitBox[S]) -> None:
    assert_type(literal, ExplicitBox[3])
    assert_type(symbolic, ExplicitBox[S])
"#,
);

testcase!(
    test_dim_field_requires_intvar_class_type_parameter,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Int

class FieldBox[N]:
    dim: Int[N]  # E: `N` must be an `IntVar` to be used as a shape dimension
"#,
);

testcase!(
    test_inttuple_elements_carrier_class_args_are_not_scalar_intvars,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Elements, IntTuple, IntVar
from typing import assert_type

class TupleBox[Shape: IntTuple]: ...
class PlainBox[N]: ...

def carrier[Bs: IntTuple, N: IntVar](
    x: TupleBox[[*Elements[Bs], N + 1]],
    y: TupleBox[IntTuple[*Elements[Bs], N + 1]],
) -> None:
    assert_type(x, TupleBox[[*Elements[Bs], N + 1]])
    assert_type(y, TupleBox[IntTuple[*Elements[Bs], N + 1]])

def scalar[N](x: PlainBox[N + 1]) -> None:  # E: `+` is not supported between `N` and `Literal[1]`  # E: Expected a type form, got instance of `int`
    pass
"#,
);

testcase!(
    test_tuple_bound_class_arg_does_not_enable_compact_shape_syntax,
    shaped_array_env_with_shaped_torch(),
    r#"
class TupleBoundBox[S: tuple[str, ...]]: ...

def f[N](x: TupleBoundBox[[N + 1]]) -> None:  # E: `ParamSpec` cannot be used for type parameter  # E: `+` is not supported between `N` and `Literal[1]`  # E: Expected a type form, got instance of `int`
    pass
"#,
);

testcase!(
    test_typevartuple_and_inttuple_class_args_parse_separately,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Elements, IntTuple, IntVar
from typing import assert_type

class Mixed[*Ts, Shape: IntTuple, N: IntVar]: ...

def f[*Ts, Shape: IntTuple, N: IntVar](
    x: Mixed[*Ts, [*Elements[Shape], N + 1], N + 2],
) -> None:
    assert_type(x, Mixed[*Ts, [*Elements[Shape], N + 1], N + 2])
"#,
);

testcase!(
    test_decorated_torch_tensor_parses_shapes,
    shaped_array_env_with_shaped_torch(),
    r#"
from typing import reveal_type
from torch import Tensor

def f(x: Tensor[[2, 3]], y: Tensor) -> None:
    reveal_type(x)  # E: revealed type: Tensor[[2, 3]]
    reveal_type(y)  # E: revealed type: Tensor
    reveal_type(x[0])  # E: revealed type: Tensor[[3]]
    reveal_type(y[0])  # E: revealed type: Tensor[tuple[Unknown, ...]]
"#,
);

testcase!(
    test_shape_arithmetic_wrapper_bracket_form,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D, IntVar
from typing import reveal_type
from torch import Tensor

def f[N: IntVar, M: IntVar](x: Tensor[[D[N] + D[M], D[N] * 2]]) -> None:
    reveal_type(x)  # E: revealed type: Tensor[[(N + M), (2 * N)]]
"#,
);

testcase!(
    test_shape_arithmetic_wrapper_call_form,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D, IntVar
from typing import reveal_type
from torch import Tensor

def f[N: IntVar, M: IntVar](x: Tensor[[D(N) // 2, D(N) ** D(M), -D(M)]]) -> None:
    reveal_type(x)  # E: revealed type: Tensor[[(N // 2), (N ** M), (-1 * M)]]
"#,
);

testcase!(
    test_shape_arithmetic_wrapper_rejects_invalid_forms,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D
from torch import Tensor

class Box[T]: ...
class Factory:
    def __init__(self, x: object) -> None: ...

def f[N, M](
    no_arg: Tensor[[D()]],  # E: Expected 1 positional argument for `D`, got 0
    too_many: Tensor[[D(N, M)]],  # E: Expected 1 positional argument for `D`, got 2
    keyword: Tensor[[D(N, dim=M)]],  # E: `D` accepts exactly 1 positional argument and no keyword arguments, got 1 positional and 1 keyword
    non_d_subscript: Tensor[[Box[N]]],  # E: Tensor shape dimensions must be positive integer literals, string literals, type variables, or expressions, got `type[Box[N]]`
    non_d_call: Tensor[[Factory(N)]],  # E: Tensor shape dimensions must be positive integer literals, string literals, type variables, or expressions, got `Factory`
) -> None:
    pass
"#,
);

testcase!(
    test_assert_shape_builtin,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D, IntVar, assert_shape
from typing import assert_type
from torch import Tensor

def f[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> None:
    assert_type(assert_shape(x, (D[N], D(M))), Tensor[[N, M]])
    assert_shape(x, (D[M], D[N]))  # E: assert_shape((N, M), (M, N)) failed
    assert_shape(x, [D[N], D(M)])  # E: Second argument to `assert_shape` must be a tuple of tensor dimensions
"#,
);

testcase!(
    test_assert_shape_preserves_registered_shape_arg,
    shaped_array_env(),
    r#"
from shape_extensions import D, IntTuple, IntVar, assert_shape, shaped_array
from typing import assert_type

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

def f[N: IntVar, M: IntVar](x: Array[[N, M], str]) -> None:
    assert_type(assert_shape(x, (D[N], D[M])), Array[[N, M], str])
    assert_shape(x, (D[M], D[N]))  # E: assert_shape((N, M), (M, N)) failed
"#,
);

testcase!(
    test_assert_shape_user_defined_helper,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import defines_assert_shape
from typing import Any, assert_type
from torch import Tensor

@defines_assert_shape
def check_shape(x: object, shape: tuple[Any, ...]) -> object: ...

def f(x: Tensor[[2, 3]]) -> None:
    assert_type(check_shape(x, (2, 3)), Tensor[[2, 3]])
    check_shape(x, (2, 4))  # E: assert_shape((2, 3), (2, 4)) failed
"#,
);

testcase!(
    test_assert_shape_rejects_non_shaped_array,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import assert_shape

assert_shape(0, (2, 3))  # E: First argument to `assert_shape` must be a shaped array, got `Literal[0]`
"#,
);

testcase!(
    test_tuple_carrier_shape_context_preserves_starred_inttuple,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, IntTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Tensor[Shape: IntTuple]: ...

class Foo[Shape: IntTuple]:
    x: Tensor[IntTuple[*Elements[Shape]]]

def f[Shape: IntTuple](x: Foo[Shape]) -> None:
    reveal_type(x)  # E: revealed type: Foo[Shape]
"#,
);

testcase!(
    test_jaxtyping_without_shape_stubs_uses_ordinary_type_args,
    shaped_array_env_with_plain_torch_and_jaxtyping(),
    r#"
from jaxtyping import Float
from torch import Tensor
from typing import reveal_type

def f(
    x: Float[Tensor, "batch channels"],
    y: Float[Tensor, 123],
    z: Float[Tensor, "shape metadata", 123],
) -> None:
    reveal_type(x)  # E: revealed type: Tensor[*tuple[Unknown, ...]]
    reveal_type(y)  # E: revealed type: Tensor[*tuple[Unknown, ...]]
    reveal_type(z)  # E: revealed type: Tensor[*tuple[Unknown, ...]]
"#,
);

#[test]
fn test_tensor_shapes_semantically_inert_without_shape_extensions() -> anyhow::Result<()> {
    let contents = r#"
from jaxtyping import Float
from torch import Tensor
from typing import Annotated, Literal, TypeVar, reveal_type

T = TypeVar("T")

class Box[T]: ...

def annotations(
    x: Tensor[Literal[2], Literal[3]],
    y: Float[Tensor, "batch channels"],
    z: Float[123, "batch"],  # E: Number literal cannot be used in annotations
    named: Float[Tensor, "batch"],
    box: Box[3],  # E: Expected a type form, got instance of `Literal[3]`
    annotated: Annotated[int, "metadata"],
) -> None:
    reveal_type(x)  # E: revealed type: Tensor[Literal[2], Literal[3]]
    reveal_type(x[0])  # E: revealed type: Tensor[Literal[2], Literal[3]]
    reveal_type(annotated)  # E: revealed type: int

def arithmetic(value: T) -> None:
    value + 1  # E: `+` is not supported between `T` and `Literal[1]`
"#;

    testcase_for_macro(plain_torch_and_jaxtyping_env(), contents, file!(), line!())?;
    Ok(())
}

testcase!(
    test_jaxtyping_accepts_decorated_torch_tensor,
    shaped_array_env_with_shaped_torch_and_jaxtyping(),
    r#"
from jaxtyping import Float
from jaxtyping import Float as F
from jaxtyping import Integer, Key, Real
import jaxtyping
import jaxtyping as jt
from torch import Tensor
from typing import assert_type, reveal_type

def f(
    x: Float[Tensor, "batch channels"],
    y: jaxtyping.Float[Tensor, "batch channels"],
    z: F[Tensor, "batch channels"],
    w: jt.Float[Tensor, "batch channels"],
    integer: Integer[Tensor, "batch channels"],
    key: Key[Tensor, "batch channels"],
    real: Real[Tensor, "batch channels"],
) -> None:
    reveal_type(x)  # E: revealed type: Shaped[Tensor, "batch channels"]
    reveal_type(y)  # E: revealed type: Shaped[Tensor, "batch channels"]
    reveal_type(z)  # E: revealed type: Shaped[Tensor, "batch channels"]
    reveal_type(w)  # E: revealed type: Shaped[Tensor, "batch channels"]
    reveal_type(integer)  # E: revealed type: Shaped[Tensor, "batch channels"]
    reveal_type(key)  # E: revealed type: Shaped[Tensor, "batch channels"]
    reveal_type(real)  # E: revealed type: Shaped[Tensor, "batch channels"]

def check_expected_type(x: Float[Tensor, "3 4"]) -> None:
    assert_type(x, jaxtyping.Shaped[Tensor, "3 4"])

def check_nontrivial_shape_syntax(
    variadic: Float[Tensor, "*batch h w"],
    arithmetic: Float[Tensor, "dim dim+1"],
) -> None:
    assert_type(variadic, jaxtyping.Shaped[Tensor, "*batch h w"])
    assert_type(arithmetic, jaxtyping.Shaped[Tensor, "dim dim+1"])

def bad_shape(x: Float[Tensor, 123]) -> None:  # E: Second argument to jaxtyping annotation must be a string literal
    pass
"#,
);

testcase!(
    test_non_jaxtyping_annotated_alias_keeps_vanilla_metadata,
    shaped_array_env_with_shaped_torch(),
    r#"
from torch import Tensor
from typing import Annotated as Float, reveal_type

def f(x: Float[Tensor, 123]) -> None:
    reveal_type(x)  # E: revealed type: Tensor
"#,
);

testcase!(
    test_jaxtyping_value_expression_keeps_vanilla_annotated_behavior,
    shaped_array_env_with_shaped_torch_and_jaxtyping(),
    r#"
from jaxtyping import Float
import jaxtyping
from torch import Tensor

alias: type[jaxtyping.Shaped[Tensor, "batch"]] = Float[Tensor, "batch"]  # E: `Annotated[Tensor]` is not assignable to `type[Shaped[Tensor, "batch"]]`
"#,
);

testcase!(
    test_shape_extensions_resolvability_enables_jaxtyping_shapes,
    {
        let mut env = shaped_array_env_with_shaped_torch();
        add_jaxtyping(&mut env);
        env
    },
    r#"
from jaxtyping import Float
from torch import Tensor
from typing import reveal_type

def f(x: Float[Tensor, "batch channels"]) -> None:
    reveal_type(x)  # E: revealed type: Shaped[Tensor, "batch channels"]
"#,
);

testcase!(
    test_numpy_shaped_array_fixture,
    shaped_array_env_with_numpy(),
    r#"
import numpy as np
from typing import reveal_type

def f(x: np.ndarray[[2, 3], float]) -> None:
    reveal_type(x)  # E: revealed type: ndarray[[2, 3], float]
    reveal_type(x.copy())  # E: revealed type: ndarray[[2, 3], float]
    reveal_type(x.item())  # E: revealed type: float
    reveal_type(x.shape)  # E: revealed type: IntTuple[2, 3]
    reveal_type(x[0])  # E: revealed type: ndarray[[3], float]
    reveal_type(np.add_leading_axis(x))  # E: revealed type: ndarray[[1, 2, 3], float]
"#,
);

testcase!(
    test_jaxtyping_inttuple_carrier_shapes,
    {
        let mut env = shaped_array_env();
        add_jaxtyping(&mut env);
        env.add_with_path(
            "tclib",
            "tclib.pyi",
            r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]:
    shape: Shape
"#,
        );
        env
    },
    r#"
from jaxtyping import Float
from tclib import Array
from typing import Literal, reveal_type

# Jaxtyping shape annotations work on a TypeVar (IntTuple) shape carrier, not just
# on torch's TypeVarTuple `*Shape`. The concrete case exercises the tuple-carrier
# sync path and the `*name` case exercises the synthesized shape-carrier TypeVar.
def concrete(x: Float[Array, "3 4"]) -> None:
    reveal_type(x)  # E: revealed type: Shaped[Array, "3 4"]

def named_variadic(x: Float[Array, "*batch channels"]) -> None:
    reveal_type(x)  # E: revealed type: Shaped[Array, "*batch channels"]
"#,
);

testcase!(
    test_numpy_tuple_carrier_meta_shape_keeps_shape_coherent,
    shaped_array_env_with_numpy(),
    r#"
import numpy as np
from typing import Literal, reveal_type

def f(x: np.tcarray[[2, 3], int]) -> None:
    y = np.tc_add_leading_axis(x)
    # The meta-shape DSL adds a leading axis. The result's shape parameter is
    # re-synced to the computed shape, so both the displayed shape and `.shape`
    # stay coherent.
    reveal_type(y)  # E: revealed type: tcarray[[1, 2, 3]]
    reveal_type(y.shape)  # E: revealed type: IntTuple[1, 2, 3]
    reveal_type(y.dtype())  # E: revealed type: int
"#,
);

testcase!(
    test_tuple_carrier_generic_return_feeds_meta_shape,
    shaped_array_env_with_numpy(),
    r#"
import numpy as np
from typing import reveal_type

def f(x: np.tcarray[[2, 3], int]) -> None:
    z = np.tc_identity(np.tc_identity(x))
    reveal_type(z)  # E: revealed type: tcarray[[2, 3]]
    y = np.tc_add_leading_axis(np.tc_identity(x))
    reveal_type(y)  # E: revealed type: tcarray[[1, 2, 3]]
"#,
);

fn shape_dsl_env() -> TestEnv {
    let mut env = shape_dsl_base_env();
    env.add_with_path(
        "my_shapes",
        "my_shapes.pyi",
        r#"
from typing import Any
from shape_extensions.dsl import ShapedArray, shape_dsl_function
import shape_extensions.dsl

class symint:
    def __mul__(self, other: symint) -> symint: ...
class Error(Exception): ...
Unknown: Any = ...

@shape_dsl_function
def identity_ir(x: int) -> int:
    return x

@shape_dsl_function
def times_two(x: int) -> int:
    return x + x

@shape_dsl_function
def double_ir(x: int) -> int:
    return times_two(x)

@shape_dsl_function
def scalar_kernel_ir(x: int) -> int:
    # Equivalent to x == 3 for the test input. The verbose spelling forces the
    # DSL evaluator through scalar arithmetic, comparison, unary, and boolean
    # operators while leaving the traced value precise.
    if not (((x + 2 == 5) and (x - 1 != 1) and (x * 2 > 5) and (x // 2 >= 1) and (x % 2 < 2) and (-x <= -3)) or False):
        raise Error("unreachable")
    return x

@shape_dsl_function
def string_guard_ir(x: int, label: str = "n") -> str:
    text = label + str(x)
    if text != "n3":
        raise Error(text)
    return "ok" if x == 3 else "bad"

@shape_dsl_function
def list_kernel_ir(x: list[int]) -> int:
    # For the test input, this sums the first four entries and adds 4 from the
    # retained indices. The deliberately indirect spelling covers indexing,
    # negative indexing, slicing, len/range, comprehensions, and in/not in.
    pair = (x[0], x[-1])
    middle = x[1:3]
    kept = [i for i in range(len(x)) if i in [1, 3] and i not in (0,)]
    return pair[0] + pair[-1] + middle[0] + middle[-1] + kept[0] + kept[1]

@shape_dsl_function
def iterator_kernel_ir(x: list[int], y: list[int]) -> int:
    indexed = [i * d for i, d in enumerate(x)]
    paired = [a + b for a, b in zip(x, y)]
    return indexed[2] + paired[1]

@shape_dsl_function
def reductions_ir(x: list[int | symint]) -> int | symint:
    return shape_extensions.dsl.prod(x) + shape_extensions.dsl.sum(x)  # E: in function `shape_extensions.dsl.prod`  # E: in function `shape_extensions.dsl.sum`

@shape_dsl_function
def identity_int_ir(x: symint) -> symint:
    return x

@shape_dsl_function
def product_int_ir(x: symint, y: symint) -> symint:
    return x * y

@shape_dsl_function
def same_int_or_one_ir(x: symint, y: symint) -> int | symint:
    if x == y:
        return x
    return 1

@shape_dsl_function
def int_min(a: int | symint, b: int | symint) -> int | symint:
    if a == b:
        return a
    if isinstance(a, int) and isinstance(b, int):
        if a < b:
            return a
        return b
    return Unknown

@shape_dsl_function
def svd_reduced_2d_ir(
    a: ShapedArray,
    full_matrices: bool,
    compute_uv: bool = True,
    hermitian: bool = False,
) -> list[ShapedArray]:
    if len(a.shape) != 2:
        raise Error("svd expects 2-D arrays")
    if full_matrices:
        raise Error("only reduced svd shapes are modeled")
    if not compute_uv:
        raise Error("svd without singular vectors is not modeled")
    if hermitian:
        raise Error("hermitian svd shapes are not modeled")
    k = int_min(a.shape[0], a.shape[1])
    return [
        ShapedArray(shape=[a.shape[0], k]),
        ShapedArray(shape=[k]),
        ShapedArray(shape=[k, a.shape[1]]),
    ]

@shape_dsl_function
def abs_int(k: int) -> int:
    if k < 0:
        return 0 - k
    return k

@shape_dsl_function
def diag_1d_ir(v: ShapedArray, k: int = 0) -> ShapedArray:
    if len(v.shape) != 1:
        raise Error("diag expects a 1-D array")
    n = v.shape[0] + abs_int(k)
    return ShapedArray(shape=[n, n])

@shape_dsl_function
def einsum_kernel_ir() -> int:
    parsed = shape_extensions.dsl.parse_einsum_equation("ab,bc->ac")
    output_map = parsed[0]
    checks = parsed[1]
    first = output_map[0]
    second = output_map[1]
    return first[0] + first[1] + second[0] + second[1] + len(checks)

def not_a_dsl_fn(x: int) -> int: ...

@shape_dsl_function
def bad_syntax_ir(x: int) -> int:
    while x > 0:  # E: @shape_dsl_function: unexpected statement in DSL body
        x = x - 1
    return x

@shape_dsl_function
def kwargs_ir(x: int, **kwargs) -> int:  # E: @shape_dsl_function: **kwargs parameters are not supported
    return x

@shape_dsl_function
def calls_undefined(x: int) -> int:  # E: @shape_dsl_function type error: undefined function: nonexistent
    return nonexistent(x)  # E: Could not find name `nonexistent`

@shape_dsl_function
def bad_no_ret(x: int):  # E: @shape_dsl_function type error: DSL function bad_no_ret must have a return type
    return x

@shape_dsl_function
def returns_wrong_type_ir(x: int) -> bool:  # E: @shape_dsl_function type error: return expression type int is not compatible with declared return type bool
    return x  # E: Returned type `int` is not assignable to declared return type `bool`

@shape_dsl_function
def dims_as_scalar_union_ir(x: list[int | symint]) -> int | symint:
    return [d for d in x]  # E: Returned type `list[int | symint]` is not assignable to declared return type `int | symint`

@shape_dsl_function
def unknown_fallback_ir(x: int) -> int:
    return Unknown

@shape_dsl_function
def helper_exact_one_ir(x: int) -> int:
    return x

@shape_dsl_function
def too_few_args_ir() -> int:  # E: @shape_dsl_function type error: 'helper_exact_one_ir' takes exactly 1 argument(s), got 0
    return helper_exact_one_ir()

@shape_dsl_function
def too_many_args_ir(x: int) -> int:  # E: @shape_dsl_function type error: 'helper_exact_one_ir' takes at most 1 argument(s), got 2
    return helper_exact_one_ir(x, x)

@shape_dsl_function
def two_errors_ir(x: int) -> int:  # E: @shape_dsl_function type error: undefined function: missing_one  # E: @shape_dsl_function type error: undefined function: missing_two
    return missing_one(x) + missing_two(x)  # E: Could not find name `missing_one`  # E: Could not find name `missing_two`
"#,
    );
    env.add_with_path(
        "my_lib",
        "my_lib.pyi",
        r#"
from typing import Any, Literal, overload
from shape_extensions import Int, IntVar, shaped_array, uses_shape_dsl
from my_shapes import identity_ir, double_ir, scalar_kernel_ir, string_guard_ir, list_kernel_ir, iterator_kernel_ir, reductions_ir, identity_int_ir, product_int_ir, same_int_or_one_ir, svd_reduced_2d_ir, diag_1d_ir, einsum_kernel_ir, not_a_dsl_fn, bad_syntax_ir, kwargs_ir, calls_undefined, bad_no_ret, two_errors_ir, returns_wrong_type_ir, dims_as_scalar_union_ir, unknown_fallback_ir, helper_exact_one_ir, too_few_args_ir, too_many_args_ir
import my_shapes

non_literal: Any

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

@uses_shape_dsl(identity_ir)
def plain_fn(x: int) -> int: ...

@overload
def overloaded_with_impl(x: int) -> int: ...
@overload
def overloaded_with_impl(x: str) -> str: ...
@uses_shape_dsl(identity_ir)
def overloaded_with_impl(x: int | str) -> int | str: ...

@uses_shape_dsl(identity_ir)
@overload
def overloaded_no_impl(x: int) -> int: ...
@overload
def overloaded_no_impl(x: str) -> str: ...

@uses_shape_dsl(double_ir)
def double_fn(x: int) -> int: ...

@uses_shape_dsl(scalar_kernel_ir)
def scalar_kernel_fn(x: int) -> int: ...

@uses_shape_dsl(string_guard_ir)
def string_guard_fn(x: int) -> str: ...

@uses_shape_dsl(list_kernel_ir)
def list_kernel_fn(x: tuple[int, ...]) -> int: ...

@uses_shape_dsl(iterator_kernel_ir)
def iterator_kernel_fn(x: tuple[int, ...], y: tuple[int, ...]) -> int: ...

@uses_shape_dsl(reductions_ir)
def reductions_fn(x: tuple[int, ...]) -> int: ...

@uses_shape_dsl(identity_int_ir)
def identity_int_fn[N: IntVar](x: Int[N]) -> int: ...

@uses_shape_dsl(product_int_ir)
def product_int_fn[N: IntVar, M: IntVar](x: Int[N], y: Int[M]) -> int: ...

@uses_shape_dsl(same_int_or_one_ir)
def same_int_or_one_fn[N: IntVar, M: IntVar](x: Int[N], y: Int[M]) -> int: ...

@uses_shape_dsl(svd_reduced_2d_ir)
def svd_fn[Shape, DType](
    a: Array[Shape, DType],
    full_matrices: Literal[False],
    compute_uv: Literal[True] = True,
    hermitian: Literal[False] = False,
) -> tuple[Array[Shape, DType], Array[Shape, DType], Array[Shape, DType]]: ...

@uses_shape_dsl(svd_reduced_2d_ir)
def svd_raw_flags_fn[Shape, DType](
    a: Array[Shape, DType],
    full_matrices: bool,
    compute_uv: bool = True,
    hermitian: bool = False,
) -> tuple[Array[Shape, DType], Array[Shape, DType], Array[Shape, DType]]: ...

@uses_shape_dsl(diag_1d_ir)
def diag_fn[Shape, DType](v: Array[Shape, DType], k: int = 0) -> Array[Shape, DType]: ...

@uses_shape_dsl(einsum_kernel_ir)
def einsum_kernel_fn() -> int: ...

@uses_shape_dsl(not_a_dsl_fn)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def bad_fn(x: int) -> int: ...

@uses_shape_dsl(bad_syntax_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def bad_syntax_fn(x: int) -> int: ...

@uses_shape_dsl(kwargs_ir)
def kwargs_fn(x: int) -> int: ...

@uses_shape_dsl(calls_undefined)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def calls_undefined_fn(x: int) -> int: ...

@uses_shape_dsl(bad_no_ret)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def no_ret_fn(x: int) -> int: ...

@uses_shape_dsl(two_errors_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def two_errors_fn(x: int) -> int: ...

@uses_shape_dsl(returns_wrong_type_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def returns_wrong_type_fn(x: int) -> bool: ...

@uses_shape_dsl(dims_as_scalar_union_ir)
def dims_as_scalar_union_fn(x: tuple[int, int]) -> tuple[int, int]: ...

@uses_shape_dsl(unknown_fallback_ir)
def unknown_fallback_fn(x: int) -> int: ...

@uses_shape_dsl(helper_exact_one_ir)
def helper_exact_one_fn(x: int) -> int: ...

@uses_shape_dsl(too_few_args_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def too_few_args_fn() -> int: ...

@uses_shape_dsl(too_many_args_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def too_many_args_fn(x: int) -> int: ...

class BadCaptureInit:
    @uses_shape_dsl(identity_ir, capture_init=["x", non_literal])  # E: `capture_init` entries must be string literals
    def forward(self, x: int) -> int: ...

@uses_shape_dsl(my_shapes.identity_ir)
def dotted_fn(x: int) -> int: ...

"#,
    );
    env
}

testcase!(
    test_uses_shape_dsl_preserves_type,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import plain_fn

# identity_ir returns its input unchanged. Because val_to_type synthesizes
# Literal[n] from the DSL's traced integer value (not the declared return
# type), the result is Literal[1], not int.
assert_type(plain_fn(1), Literal[1])
"#,
);

testcase!(
    test_uses_shape_dsl_overload_with_implementation,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import overloaded_with_impl

assert_type(overloaded_with_impl(1), Literal[1])
assert_type(overloaded_with_impl("a"), str)
"#,
);

testcase!(
    test_uses_shape_dsl_overload_no_implementation,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import overloaded_no_impl

assert_type(overloaded_no_impl(1), Literal[1])
assert_type(overloaded_no_impl("a"), str)
"#,
);

testcase!(
    test_uses_shape_dsl_cross_function_call,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import double_fn

assert_type(double_fn(3), Literal[6])
"#,
);

testcase!(
    test_shape_dsl_scalar_arithmetic_and_comparisons,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import scalar_kernel_fn

assert_type(scalar_kernel_fn(3), Literal[3])
"#,
);

testcase!(
    test_shape_dsl_strings_defaults_conditionals_and_raise,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import string_guard_fn

assert_type(string_guard_fn(3), str)
string_guard_fn(4)  # E: n4
"#,
);

testcase!(
    test_shape_dsl_list_primitives,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import list_kernel_fn

assert_type(list_kernel_fn((2, 3, 5, 7)), Literal[21])
"#,
);

testcase!(
    test_shape_dsl_iterator_builtins,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import iterator_kernel_fn

assert_type(iterator_kernel_fn((2, 3, 5), (7, 11, 13)), Literal[24])
"#,
);

testcase!(
    test_shape_dsl_reduction_builtins,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import reductions_fn

assert_type(reductions_fn((2, 3, 4)), Literal[33])
"#,
);

testcase!(
    test_shape_dsl_int_return_uses_canonical_size,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type, reveal_type
from shape_extensions import Int, IntVar
from my_lib import identity_int_fn, product_int_fn

def f[N: IntVar, M: IntVar](n: Int[N], m: Int[M]) -> None:
    reveal_type(identity_int_fn(n))  # E: revealed type: Int[N]
    reveal_type(product_int_fn(n, m))  # E: revealed type: Int[(N * M)]
    assert_type(identity_int_fn(n), Int[N])
    assert_type(product_int_fn(n, m), Int[N * M])
    assert_type(identity_int_fn(3), Literal[3])
    assert_type(product_int_fn(3, 4), Literal[12])
"#,
);

testcase!(
    test_shape_dsl_int_equality,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from shape_extensions import Int, IntVar
from my_lib import same_int_or_one_fn

def f[N: IntVar, M: IntVar](n: Int[N], m: Int[M]) -> None:
    assert_type(same_int_or_one_fn(n, n), Int[N])
    assert_type(same_int_or_one_fn(n, m), Literal[1])
"#,
);

testcase!(
    test_shape_dsl_svd_reduced_2d_shapes,
    shape_dsl_env(),
    r#"
from typing import Literal, reveal_type
from my_lib import Array, svd_fn

def f(tall: Array[[5, 3], float], wide: Array[[3, 5], float], square: Array[[4, 4], float]) -> None:
    tall_u, tall_s, tall_vt = svd_fn(tall, full_matrices=False)
    reveal_type(tall_u)  # E: revealed type: Array[[5, 3], float]
    reveal_type(tall_s)  # E: revealed type: Array[[3], float]
    reveal_type(tall_vt)  # E: revealed type: Array[[3, 3], float]

    wide_u, wide_s, wide_vt = svd_fn(wide, full_matrices=False)
    reveal_type(wide_u)  # E: revealed type: Array[[3, 3], float]
    reveal_type(wide_s)  # E: revealed type: Array[[3], float]
    reveal_type(wide_vt)  # E: revealed type: Array[[3, 5], float]

    square_u, square_s, square_vt = svd_fn(square, full_matrices=False)
    reveal_type(square_u)  # E: revealed type: Array[[4, 4], float]
    reveal_type(square_s)  # E: revealed type: Array[[4], float]
    reveal_type(square_vt)  # E: revealed type: Array[[4, 4], float]
"#,
);

testcase!(
    test_shape_dsl_svd_rejects_unsupported_modes,
    shape_dsl_env(),
    r#"
from my_lib import Array, svd_raw_flags_fn

def f(x: Array[[5, 3], float], vector: Array[[5], float]) -> None:
    svd_raw_flags_fn(vector, full_matrices=False)  # E: svd expects 2-D arrays
    svd_raw_flags_fn(x, full_matrices=True)  # E: only reduced svd shapes are modeled
    svd_raw_flags_fn(x, full_matrices=False, compute_uv=False)  # E: svd without singular vectors is not modeled
    svd_raw_flags_fn(x, full_matrices=False, hermitian=True)  # E: hermitian svd shapes are not modeled
"#,
);

testcase!(
    test_shape_dsl_diag_1d_shapes,
    shape_dsl_env(),
    r#"
from typing import reveal_type
from my_lib import Array, diag_fn

def f(vector: Array[[4], float], matrix: Array[[4, 4], float]) -> None:
    reveal_type(diag_fn(vector))  # E: revealed type: Array[[4, 4], float]
    reveal_type(diag_fn(vector, 1))  # E: revealed type: Array[[5, 5], float]
    reveal_type(diag_fn(vector, -1))  # E: revealed type: Array[[5, 5], float]
    diag_fn(matrix)  # E: diag expects a 1-D array
"#,
);

testcase!(
    test_shape_dsl_parse_einsum_equation_builtin,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import einsum_kernel_fn

assert_type(einsum_kernel_fn(), Literal[3])
"#,
);

testcase!(
    test_uses_shape_dsl_not_a_dsl_function,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import bad_fn

# The @uses_shape_dsl argument is not a @shape_dsl_function, so no shape
# transform is applied and the declared return type (int) is used instead.
assert_type(bad_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_unsupported_syntax,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import bad_syntax_fn

# bad_syntax_ir uses a while loop which is unsupported DSL syntax, so
# bad_syntax_fn falls back to the declared return type.
assert_type(bad_syntax_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_kwargs_warning,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import kwargs_fn

# kwargs_ir has **kwargs which triggers a warning but the DSL conversion
# still succeeds (kwargs are silently dropped), so shape inference works.
assert_type(kwargs_fn(1), Literal[1])
"#,
);

testcase!(
    test_shape_dsl_uses_failing_function,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import calls_undefined_fn

# calls_undefined is rejected because its body calls an undefined helper. The
# consumer also gets rejected as a DSL use-site and falls back to its declared
# return type.
assert_type(calls_undefined_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_function_requires_return_annotation,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import no_ret_fn

# bad_no_ret is not accepted as a DSL function without a return annotation, so
# no_ret_fn falls back to its declared return type.
assert_type(no_ret_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_reports_multiple_errors,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import two_errors_fn

# two_errors_ir reports both undefined helper names from the same DSL body, and
# the consumer falls back to the declared return type.
assert_type(two_errors_fn(1), int)
"#,
);

testcase!(
    bug = "dotted-name arguments to @uses_shape_dsl silent-noop; should emit a diagnostic",
    test_shape_dsl_dotted_name_silent_noop,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import dotted_fn

# Dotted-name arguments are currently ignored without a diagnostic, so no shape
# transform is applied and the declared return type is used.
assert_type(dotted_fn(1), int)
"#,
);

// ── Recursion-safety tests ────────────────────────────────────────────────────

fn shape_dsl_recursion_env() -> TestEnv {
    let mut env = shape_dsl_base_env();
    env.add_with_path(
        "recursive_shapes",
        "recursive_shapes.pyi",
        r#"
from shape_extensions.dsl import shape_dsl_function

# Direct self-recursion: should be rejected with a cycle diagnostic.
@shape_dsl_function
def self_recursive_ir(x: int) -> int:  # E: @shape_dsl_function type error: DSL function 'self_recursive_ir' is recursive
    return self_recursive_ir(x)

# Mutual recursion A → B → A: both should be rejected individually.
@shape_dsl_function
def mutual_a_ir(x: int) -> int:  # E: @shape_dsl_function type error: DSL function 'mutual_a_ir' is recursive
    return mutual_b_ir(x)

@shape_dsl_function
def mutual_b_ir(x: int) -> int:  # E: @shape_dsl_function type error: DSL function 'mutual_b_ir' is recursive
    return mutual_a_ir(x)

# Non-recursive depth-3 chain: triple_ir → triple_mid → triple_leaf.
# For input n, triple_leaf(n) = n+n+n = 3n, so triple_ir(4) = 12.
@shape_dsl_function
def triple_leaf(x: int) -> int:
    return x + x + x

@shape_dsl_function
def triple_mid(x: int) -> int:
    return triple_leaf(x)

@shape_dsl_function
def triple_ir(x: int) -> int:
    return triple_mid(x)
"#,
    );
    env.add_with_path(
        "recursive_lib",
        "recursive_lib.pyi",
        r#"
from shape_extensions import uses_shape_dsl
from recursive_shapes import self_recursive_ir, mutual_a_ir, triple_ir

@uses_shape_dsl(self_recursive_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def self_recursive_fn(x: int) -> int: ...

@uses_shape_dsl(mutual_a_ir)  # E: `@uses_shape_dsl` argument does not resolve to a `@shape_dsl_function`
def mutual_fn(x: int) -> int: ...

@uses_shape_dsl(triple_ir)
def triple_fn(x: int) -> int: ...
"#,
    );
    env
}

testcase!(
    test_shape_dsl_self_recursive_rejected,
    shape_dsl_recursion_env(),
    r#"
from typing import assert_type
from recursive_lib import self_recursive_fn

# self_recursive_ir is rejected as recursive, so self_recursive_fn falls
# back to its declared return type rather than crashing the evaluator.
assert_type(self_recursive_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_mutual_recursive_rejected,
    shape_dsl_recursion_env(),
    r#"
from typing import assert_type
from recursive_lib import mutual_fn

# mutual_a_ir / mutual_b_ir form a cycle; mutual_fn falls back to int.
assert_type(mutual_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_non_recursive_chain,
    shape_dsl_recursion_env(),
    r#"
from typing import Literal, assert_type
from recursive_lib import triple_fn

# triple_ir → triple_mid → triple_leaf is a valid depth-3 chain with no
# cycles.  triple_leaf(x) = x+x+x, so triple_fn(4) evaluates to Literal[12].
assert_type(triple_fn(4), Literal[12])
"#,
);

testcase!(
    test_shape_dsl_wrong_return_type,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import returns_wrong_type_fn

# returns_wrong_type_ir is declared `-> bool` but its body returns an `int`
# expression, so it fails the compile-time return-type check and
# returns_wrong_type_fn falls back to its declared bool return type.
assert_type(returns_wrong_type_fn(1), bool)
"#,
);

testcase!(
    test_shape_dsl_list_return_for_scalar_union,
    shape_dsl_env(),
    r#"
from typing import Literal, assert_type
from my_lib import dims_as_scalar_union_fn

# Tensor.size() uses this shape: the DSL annotation is the scalar dimension
# type `int | symint`, but returning a list of dimensions means "produce a
# concrete tuple of dimensions".
assert_type(dims_as_scalar_union_fn((1, 2)), tuple[Literal[1], Literal[2]])
"#,
);

testcase!(
    test_shape_dsl_unknown_return_fallback,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import unknown_fallback_fn

# Unknown is the DSL's explicit fixture fallback sentinel. It should not make
# the DSL function invalid just because it evaluates to Val::None internally.
assert_type(unknown_fallback_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_arg_count_too_few,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import too_few_args_fn

# too_few_args_ir calls helper_exact_one_ir() with 0 args but it needs 1,
# so the DSL compile-time check fires and the consumer falls back to int.
assert_type(too_few_args_fn(), int)
"#,
);

testcase!(
    test_shape_dsl_arg_count_too_many,
    shape_dsl_env(),
    r#"
from typing import assert_type
from my_lib import too_many_args_fn

# too_many_args_ir calls helper_exact_one_ir(x, x) with 2 args but it takes 1,
# so the DSL compile-time check fires and the consumer falls back to int.
assert_type(too_many_args_fn(1), int)
"#,
);

testcase!(
    test_shape_dsl_capture_init_requires_string_literals,
    shape_dsl_env(),
    r#"
from my_lib import BadCaptureInit

# capture_init is read during class binding. Non-literal entries are rejected
# instead of silently dropping them from the captured __init__ field list.
BadCaptureInit()
"#,
);

testcase!(
    test_shape_dsl_shape_specific_primitives,
    {
        let mut env = shape_dsl_tensor_env();
        env.add_with_path(
            "shape_ops",
            "shape_ops.pyi",
r#"
from shape_extensions import IntTuple, uses_shape_dsl
from shape_extensions.dsl import ShapedArray, shape_dsl_function
from torch import Tensor

class symint: ...

@shape_dsl_function
def replace_leading_dim_ir(x: ShapedArray, dim: int | symint) -> ShapedArray:
    dims = x.shape
    if isinstance(x, ShapedArray) and isinstance(dims, list) and isinstance(dims[0], int) and not isinstance(dim, symint):
        return ShapedArray(shape=[dim] + dims[1:])
    return ShapedArray(shape=dims)

@uses_shape_dsl(replace_leading_dim_ir)
def replace_leading_dim[Shape: IntTuple](x: Tensor[Shape], dim: int) -> Tensor[Shape]: ...
"#,
        );
        env
    },
    r#"
from shape_ops import replace_leading_dim
from torch import Tensor
from typing import Literal, assert_type

def f(x: Tensor[[2, 3]]) -> None:
    assert_type(x.shape, tuple[Literal[2], Literal[3]])
    assert_type(replace_leading_dim(x, 4), Tensor[[4, 3]])
"#,
);

testcase!(
    test_shape_dsl_numpy_matmul_2d_helper,
    {
        let mut env = shape_dsl_base_env();
        env.add_with_path(
            "numpy_like",
            "numpy_like.pyi",
            r#"
from shape_extensions import shaped_array, uses_shape_dsl
from shape_extensions.dsl import ShapedArray, shape_dsl_function

class Error(Exception): ...

@shape_dsl_function
def matmul_2d_ir(a: ShapedArray, b: ShapedArray) -> ShapedArray:
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise Error("matmul expects 2-D arrays")
    if isinstance(a.shape[1], int) and isinstance(b.shape[0], int) and a.shape[1] != b.shape[0]:
        raise Error("matmul inner dimensions must match")
    return ShapedArray(shape=[a.shape[0], b.shape[1]])

@shaped_array(shape="Shape")
class Array[Shape]: ...

@uses_shape_dsl(matmul_2d_ir)
def matmul(a: Array, b: Array) -> Array: ...
"#,
        );
        env
    },
    r#"
from numpy_like import Array, matmul
from typing import Literal, assert_type

def f(
    good_left: Array[tuple[Literal[3], Literal[4]]],
    good_right: Array[tuple[Literal[4], Literal[5]]],
    bad_right: Array[tuple[Literal[6], Literal[5]]],
    vector: Array[tuple[Literal[4]]],
) -> None:
    assert_type(matmul(good_left, good_right), Array[tuple[Literal[3], Literal[5]]])
    matmul(good_left, bad_right)  # E: matmul inner dimensions must match
    matmul(good_left, vector)  # E: matmul expects 2-D arrays
"#,
);

testcase!(
    test_assert_type_gradual_shape_not_equivalent_to_concrete,
    shaped_array_env(),
    r#"
from typing import Any, assert_type
from shape_extensions import Int, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def bare_dims(gradual: Int[int], concrete: Int[3]) -> None:
    # A gradual dimension is the shape analog of `Any`: not equivalent to a concrete size.
    assert_type(gradual, Int[3])  # E: assert_type
    assert_type(concrete, Int[int])  # E: assert_type
    # Sameness still holds.
    assert_type(gradual, Int[int])
    assert_type(concrete, Int[3])

def shapes(gradual: Array[[Any], int], concrete: Array[[3], int]) -> None:
    assert_type(gradual, Array[[3], int])  # E: assert_type
    assert_type(concrete, Array[[Any], int])  # E: assert_type
    assert_type(gradual, Array[[Any], int])
    assert_type(concrete, Array[[3], int])
"#,
);

testcase!(
    test_assert_type_shapeless_shape_not_equivalent_to_concrete,
    shaped_array_env(),
    r#"
from typing import assert_type
from shape_extensions import IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(shapeless: Array[IntTuple, int], concrete: Array[[3], int]) -> None:
    # A wholly shapeless array is the maximal gradual shape (unknown rank) — the
    # whole-tensor analog of `Any` — so it is non-equivalent to a concrete shape
    # under `assert_type`, matching the gradual-dimension case above.
    assert_type(shapeless, Array[[3], int])  # E: assert_type
    assert_type(concrete, Array[IntTuple, int])  # E: assert_type
    # Sameness and gradual assignability are unaffected.
    assert_type(shapeless, Array[IntTuple, int])
    assert_type(concrete, Array[[3], int])
"#,
);
testcase!(
    test_type_shape_dsl_reduction_flag_values,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import Literal, reveal_type

@type_shape_dsl_function
def reduction_shape(
    shape: IntTuple, axis: int | tuple[int, ...] | None,
) -> IntTuple:
    if axis is None:
        axes = range(len(shape))
    elif dsl.is_int_value(axis):
        normalized = axis % len(shape)
        axes = (normalized,)
    else:
        axes = axis
    if 0 not in axes and 1 not in axes:
        return shape
    if 0 in axes and 1 in axes:
        return dsl.IntTuple(())
    if 0 in axes:
        return dsl.IntTuple((shape[1],))
    return dsl.IntTuple((shape[0],))

@type_shape_dsl_function
def unused_flag(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    ignored = axis
    return shape

@type_shape_dsl_function
def choose_shape(left: IntTuple, right: IntTuple, choose: int) -> IntTuple:
    if choose < 0:
        result = left
    else:
        result = right
    return result

@type_shape_dsl_function
def choose_axis(
    shape: IntTuple,
    first: int | tuple[int, ...] | None,
    second: int | tuple[int, ...] | None,
    choose: int,
) -> IntTuple:
    if choose < 0:
        axis = first
    elif dsl.is_int_value(first):
        axis = first
    else:
        axis = second
    if dsl.is_int_value(axis):
        return dsl.IntTuple((shape[0],))
    return shape

def reduce[Shape: IntTuple, Axis: Flag[int | tuple[int, ...] | None]](
    x: Tensor[Shape], axis: Axis = None,
) -> Tensor[reduction_shape(Shape, Axis)]: ...

def default_axis(x: Tensor[[2, 3]]) -> None:
    reveal_type(reduce(x))  # E: revealed type: Tensor[[]]
    reveal_type(reduce(x, 0))  # E: revealed type: Tensor[[3]]
    reveal_type(reduce(x, -1))  # E: revealed type: Tensor[[2]]
    reveal_type(reduce(x, (0, 1)))  # E: revealed type: Tensor[[]]
    reveal_type(reduce(x, ()))  # E: revealed type: Tensor[[2, 3]]

def broad() -> Tensor[reduction_shape(IntTuple[2, 3], int)]: ...
def unused_broad() -> Tensor[unused_flag(IntTuple[2, 3], int)]: ...
def choose_left() -> Tensor[choose_shape(IntTuple[2], IntTuple[3], -1)]: ...
def choose_right() -> Tensor[choose_shape(IntTuple[2], IntTuple[3], 1)]: ...
def choose_first_axis() -> Tensor[choose_axis(IntTuple[2, 3], 0, tuple[Literal[1]], -1)]: ...
def choose_narrowed_axis() -> Tensor[choose_axis(IntTuple[2, 3], 1, tuple[Literal[0]], 0)]: ...
def choose_second_axis() -> Tensor[choose_axis(IntTuple[2, 3], tuple[Literal[1]], 0, 1)]: ...
def choose_second_sequence() -> Tensor[choose_axis(IntTuple[2, 3], tuple[Literal[1]], tuple[Literal[0]], 1)]: ...

def check_broad() -> None:
    reveal_type(broad())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(unused_broad())  # E: revealed type: Tensor[[2, 3]]
    reveal_type(choose_left())  # E: revealed type: Tensor[[2]]
    reveal_type(choose_right())  # E: revealed type: Tensor[[3]]
    reveal_type(choose_first_axis())  # E: revealed type: Tensor[[2]]
    reveal_type(choose_narrowed_axis())  # E: revealed type: Tensor[[2]]
    reveal_type(choose_second_axis())  # E: revealed type: Tensor[[2]]
    reveal_type(choose_second_sequence())  # E: revealed type: Tensor[[2, 3]]
"#,
);

testcase!(
    test_type_shape_dsl_invalid_locals_and_flag_values,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def reassigned(shape: IntTuple) -> IntTuple:
    rank = len(shape)
    rank = 2  # E: locals are immutable and cannot be reassigned
    return shape

@type_shape_dsl_function
def assigned_parameter(shape: IntTuple) -> IntTuple:
    shape = shape  # E: parameters are immutable and cannot be assigned
    return shape

@type_shape_dsl_function
def unnarrowed_flag_length(shape: IntTuple, axes: tuple[int, ...]) -> IntTuple:
    if len(axes) == 0:  # E: `len` of a Flag value requires control-flow narrowing to a sequence
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def branch_only(shape: IntTuple, choose: int) -> IntTuple:
    if choose < 0:
        axes = (0,)
    if 0 in axes:  # E: local value must be definitely assigned before use  # E: may be uninitialized
        return dsl.IntTuple((shape[0],))
    return shape

@type_shape_dsl_function
def mutable(shape: IntTuple) -> IntTuple:
    axes = [0]  # E: local assignment value is not supported
    return shape

@type_shape_dsl_function
def wrong_domain(shape: IntTuple) -> IntTuple:
    axes = (shape,)  # E: Flag operation requires a compatible Flag parameter
    return shape

@type_shape_dsl_function
def mutation(shape: IntTuple) -> IntTuple:
    shape[0] = 1  # E: local assignment requires exactly one bare name target  # E: Cannot set item
    return shape

@type_shape_dsl_function
def incompatible_branch_alias(left: Int, right: IntTuple, choose: int) -> IntTuple:
    if choose < 0:
        result = left
    else:
        result = right
    return result  # E: local alias return domain must match the declared result  # E: Returned type
"#,
);

testcase!(
    test_type_shape_dsl_flag_value_regressions,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, Int, IntTuple, broadcast, type_shape_dsl_function
from torch import Tensor
from typing import reveal_type

@type_shape_dsl_function
def alias_isinstance(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    local_axis = axis
    if dsl.is_int_value(local_axis):
        return dsl.IntTuple((shape[0],))
    return shape

@type_shape_dsl_function
def alias_broadcast(left: IntTuple, right: IntTuple) -> IntTuple:
    local_left = left
    local_right = right
    return broadcast(local_left, local_right)

@type_shape_dsl_function
def alias_dimension_compare(
    left: Int, right: Int, equal: IntTuple, less: IntTuple, greater: IntTuple,
) -> IntTuple:
    local_left = left
    local_right = right
    if local_left == local_right:
        return equal
    if local_left < local_right:
        return less
    return greater

@type_shape_dsl_function
def indexed_dimension_compare(
    shape: IntTuple, right: Int, equal: IntTuple, less: IntTuple, greater: IntTuple,
) -> IntTuple:
    left = shape[0]
    if left == right:
        return equal
    if left < right:
        return less
    return greater

@type_shape_dsl_function
def two_indexed_dimensions(shape: IntTuple, equal: IntTuple, unequal: IntTuple) -> IntTuple:
    left = shape[0]
    right = shape[1]
    if left == right:
        return equal
    return unequal

@type_shape_dsl_function
def dimension_and_flag(shape: IntTuple, right: int) -> IntTuple:
    left = shape[0]
    if left == right:  # E: comparison operands must both be annotated as `Int` or both be `Flag[int]`
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def merged_narrowing(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    local_axis = axis
    if dsl.is_int_value(local_axis):
        marker = local_axis + 1
    return shape

@type_shape_dsl_function
def sequence_length(shape: IntTuple) -> IntTuple:
    axes = (3, 2, 1, 0)
    if len(axes) == 4 and 0 in range(3, -1, -1):
        return dsl.IntTuple((shape[0],))
    return shape

@type_shape_dsl_function
def compare_flag_values(
    left: int, right: int, equal: IntTuple, less: IntTuple, greater: IntTuple,
) -> IntTuple:
    if left == right:
        return equal
    if left < right:
        return less
    return greater

@type_shape_dsl_function
def disjoint_local_domains(shape: IntTuple, a: Int, b: Int) -> IntTuple:
    if a == b:
        result = shape[0]
        return shape
    result = shape
    return result

def alias_axis[Shape: IntTuple, Axis: Flag[int | tuple[int, ...] | None]](
    x: Tensor[Shape], axis: Axis,
) -> Tensor[alias_isinstance(Shape, Axis)]: ...
def apply_broadcast[Left: IntTuple, Right: IntTuple](
    left: Tensor[Left], right: Tensor[Right],
) -> Tensor[alias_broadcast(Left, Right)]: ...
def alias_equal() -> Tensor[alias_dimension_compare(Int[2], Int[2], IntTuple[1], IntTuple[2], IntTuple[3])]: ...
def alias_less() -> Tensor[alias_dimension_compare(Int[1], Int[2], IntTuple[1], IntTuple[2], IntTuple[3])]: ...
def indexed_equal() -> Tensor[indexed_dimension_compare(IntTuple[2], Int[2], IntTuple[1], IntTuple[2], IntTuple[3])]: ...
def indexed_less() -> Tensor[indexed_dimension_compare(IntTuple[1], Int[2], IntTuple[1], IntTuple[2], IntTuple[3])]: ...
def indexed_pair_equal() -> Tensor[two_indexed_dimensions(IntTuple[2, 2], IntTuple[1], IntTuple[2])]: ...
def indexed_pair_unequal() -> Tensor[two_indexed_dimensions(IntTuple[2, 3], IntTuple[1], IntTuple[2])]: ...
def apply_merged[Shape: IntTuple, Axis: Flag[int | tuple[int, ...] | None]](
    x: Tensor[Shape], axis: Axis,
) -> Tensor[merged_narrowing(Shape, Axis)]: ...
def disjoint_equal() -> Tensor[disjoint_local_domains(IntTuple[4, 5], Int[1], Int[1])]: ...
def disjoint_unequal() -> Tensor[disjoint_local_domains(IntTuple[4, 5], Int[1], Int[2])]: ...
def flags_equal() -> Tensor[compare_flag_values(1, 1, IntTuple[1], IntTuple[2], IntTuple[3])]: ...
def flags_less() -> Tensor[compare_flag_values(1, 2, IntTuple[1], IntTuple[2], IntTuple[3])]: ...
def flags_greater() -> Tensor[compare_flag_values(2, 1, IntTuple[1], IntTuple[2], IntTuple[3])]: ...
def apply_length[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[sequence_length(Shape)]: ...

def test(x: Tensor[[2, 3]], left: Tensor[[2, 1]], right: Tensor[[1, 3]]) -> None:
    reveal_type(alias_axis(x, 1))  # E: revealed type: Tensor[[2]]
    reveal_type(alias_axis(x, None))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_broadcast(left, right))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(alias_equal())  # E: revealed type: Tensor[[1]]
    reveal_type(alias_less())  # E: revealed type: Tensor[[2]]
    reveal_type(indexed_equal())  # E: revealed type: Tensor[[1]]
    reveal_type(indexed_less())  # E: revealed type: Tensor[[2]]
    reveal_type(indexed_pair_equal())  # E: revealed type: Tensor[[1]]
    reveal_type(indexed_pair_unequal())  # E: revealed type: Tensor[[2]]
    reveal_type(apply_merged(x, 1))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_length(x))  # E: revealed type: Tensor[[2]]
    reveal_type(disjoint_equal())  # E: revealed type: Tensor[[4, 5]]
    reveal_type(disjoint_unequal())  # E: revealed type: Tensor[[4, 5]]
    reveal_type(flags_equal())  # E: revealed type: Tensor[[1]]
    reveal_type(flags_less())  # E: revealed type: Tensor[[2]]
    reveal_type(flags_greater())  # E: revealed type: Tensor[[3]]
"#,
);

testcase!(
    test_type_shape_dsl_conditional_expressions,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def choose_dimension(shape: IntTuple, axis: int) -> IntTuple:
    return dsl.IntTuple((shape[0] if axis == 0 else shape[1],))

@type_shape_dsl_function
def choose_flag(shape: IntTuple, axis: int) -> IntTuple:
    selected = 0 if axis == 0 else 1
    if selected == 0:
        return dsl.IntTuple((shape[0],))
    return dsl.IntTuple((shape[1],))

def apply_dimension[Axis: Flag[int]](x: Tensor[[2, 3]], axis: Axis) -> Tensor[choose_dimension(IntTuple[2, 3], Axis)]: ...
def apply_flag[Axis: Flag[int]](x: Tensor[[2, 3]], axis: Axis) -> Tensor[choose_flag(IntTuple[2, 3], Axis)]: ...
def broad() -> Tensor[choose_dimension(IntTuple[2, 3], int)]: ...

def test(x: Tensor[[2, 3]]) -> None:
    assert_type(apply_dimension(x, 0), Tensor[[2]])
    assert_type(apply_dimension(x, 1), Tensor[[3]])
    assert_type(apply_flag(x, 0), Tensor[[2]])
    assert_type(apply_flag(x, 1), Tensor[[3]])
    reveal_type(broad())  # E: revealed type: Tensor[tuple[Unknown, ...]]
"#,
);

testcase!(
    test_type_shape_dsl_invalid_flag_value_regressions,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import Tuple as TypingTuple, reveal_type

@type_shape_dsl_function
def maybe_reassigned(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    if axis is None:
        axes = (0,)
    axes = (1,)  # E: locals are immutable and cannot be reassigned
    return shape

@type_shape_dsl_function
def typing_tuple_is_not_dsl(shape: IntTuple) -> IntTuple:
    axes = TypingTuple((0,))  # E: local assignment value is not supported  # E: Expected a callable
    return shape

@type_shape_dsl_function
def zero_step(shape: IntTuple, axis: int) -> IntTuple:
    unused = range(axis, 3, 0)
    return shape

@type_shape_dsl_function
def zero_division(shape: IntTuple, axis: int) -> IntTuple:
    unused = axis // 0  # E: Cannot divide by zero
    return shape

@type_shape_dsl_function
def overflow(shape: IntTuple) -> IntTuple:
    unused = 9223372036854775807 + 1
    return shape

@type_shape_dsl_function
def overflow_subtract(shape: IntTuple) -> IntTuple:
    unused = -9223372036854775808 - 1
    return shape

@type_shape_dsl_function
def overflow_multiply(shape: IntTuple) -> IntTuple:
    unused = 9223372036854775807 * 2
    return shape

@type_shape_dsl_function
def overflow_floor_divide(shape: IntTuple) -> IntTuple:
    unused = -9223372036854775808 // -1
    return shape

@type_shape_dsl_function
def overflow_negative_literal(shape: IntTuple) -> IntTuple:
    unused = -9223372036854775809
    return shape

@type_shape_dsl_function
def exact_min_modulo_negative_one(shape: IntTuple) -> IntTuple:
    remainder = -9223372036854775808 % -1
    if remainder == 0:
        return shape
    return dsl.IntTuple(())

@type_shape_dsl_function
def used_overflow(shape: IntTuple) -> IntTuple:
    marker = 9223372036854775807 + 1
    if marker == 0:
        return shape
    return shape

@type_shape_dsl_function
def invalid_right_operand_after_overflow(shape: IntTuple) -> IntTuple:
    unused = (9223372036854775807 + 1) + (1 % 0)  # E: Cannot divide by zero
    return shape

@type_shape_dsl_function
def unknown_modulo_zero(shape: IntTuple, axis: int) -> IntTuple:
    unused = axis % 0  # E: Cannot divide by zero
    return shape

@type_shape_dsl_function
def nested_invalid(shape: IntTuple) -> IntTuple:
    unused = (1 % 0) // 0  # E: Cannot divide by zero  # E: Cannot divide by zero
    return shape

@type_shape_dsl_function
def invalid_comparison(shape: IntTuple, axis: int) -> IntTuple:
    if axis < 1 // 0:  # E: Cannot divide by zero
        return shape
    return shape

@type_shape_dsl_function
def invalid_membership(shape: IntTuple, axis: int) -> IntTuple:
    if axis in range(0, 1, 0):
        return shape
    return shape

@type_shape_dsl_function
def unknown_then_false(shape: IntTuple, axis: int, false_result: IntTuple) -> IntTuple:
    if axis < 0 and 1 == 1 and 0 == 1:
        return false_result
    return shape

@type_shape_dsl_function
def unknown_then_true(shape: IntTuple, axis: int, false_result: IntTuple) -> IntTuple:
    if axis < 0 or 0 == 1 or 1 == 1:
        return shape
    return false_result

@type_shape_dsl_function
def unknown_before_invalid(shape: IntTuple, axis: int) -> IntTuple:
    if axis < 0 and 1 % 0 == 0:  # E: Cannot divide by zero
        return shape
    return shape

@type_shape_dsl_function
def known_before_invalid(shape: IntTuple) -> IntTuple:
    if 1 == 1 and 1 % 0 == 0:  # E: Cannot divide by zero
        return shape
    return shape

@type_shape_dsl_function
def invalid_before_unknown(shape: IntTuple, axis: int) -> IntTuple:
    if 1 % 0 == 0 and axis < 0:  # E: Cannot divide by zero
        return shape
    return shape

@type_shape_dsl_function
def false_before_invalid(shape: IntTuple, true_result: IntTuple) -> IntTuple:
    if 0 == 1 and 1 % 0 == 0:
        return true_result
    return shape

def check_zero_step(x: Tensor[[2, 3]]) -> Tensor[zero_step(IntTuple[2, 3], int)]: ...
def check_zero_division(x: Tensor[[2, 3]]) -> Tensor[zero_division(IntTuple[2, 3], int)]: ...
def check_overflow(x: Tensor[[2, 3]]) -> Tensor[overflow(IntTuple[2, 3])]: ...
def check_overflow_subtract(x: Tensor[[2, 3]]) -> Tensor[overflow_subtract(IntTuple[2, 3])]: ...
def check_overflow_multiply(x: Tensor[[2, 3]]) -> Tensor[overflow_multiply(IntTuple[2, 3])]: ...
def check_overflow_floor_divide(x: Tensor[[2, 3]]) -> Tensor[overflow_floor_divide(IntTuple[2, 3])]: ...
def check_overflow_negative_literal(x: Tensor[[2, 3]]) -> Tensor[overflow_negative_literal(IntTuple[2, 3])]: ...
def check_exact_modulo(x: Tensor[[2, 3]]) -> Tensor[exact_min_modulo_negative_one(IntTuple[2, 3])]: ...
def check_used_overflow(x: Tensor[[2, 3]]) -> Tensor[used_overflow(IntTuple[2, 3])]: ...
def check_invalid_right_operand(x: Tensor[[2, 3]]) -> Tensor[invalid_right_operand_after_overflow(IntTuple[2, 3])]: ...
def check_unknown_modulo_zero(x: Tensor[[2, 3]]) -> Tensor[unknown_modulo_zero(IntTuple[2, 3], int)]: ...
def check_nested_invalid(x: Tensor[[2, 3]]) -> Tensor[nested_invalid(IntTuple[2, 3])]: ...
def check_comparison(x: Tensor[[2, 3]]) -> Tensor[invalid_comparison(IntTuple[2, 3], int)]: ...
def check_membership(x: Tensor[[2, 3]]) -> Tensor[invalid_membership(IntTuple[2, 3], int)]: ...
def check_unknown_then_false(x: Tensor[[2, 3]]) -> Tensor[unknown_then_false(IntTuple[2, 3], int, IntTuple[1])]: ...
def check_unknown_then_true(x: Tensor[[2, 3]]) -> Tensor[unknown_then_true(IntTuple[2, 3], int, IntTuple[1])]: ...
def check_unknown_before_invalid(x: Tensor[[2, 3]]) -> Tensor[unknown_before_invalid(IntTuple[2, 3], int)]: ...
def check_known_before_invalid(x: Tensor[[2, 3]]) -> Tensor[known_before_invalid(IntTuple[2, 3])]: ...
def check_invalid_before_unknown(x: Tensor[[2, 3]]) -> Tensor[invalid_before_unknown(IntTuple[2, 3], int)]: ...
def check_false_before_invalid(x: Tensor[[2, 3]]) -> Tensor[false_before_invalid(IntTuple[2, 3], IntTuple[1])]: ...

def test(x: Tensor[[2, 3]]) -> None:
    check_zero_step(x)  # E: range() arg 3 must not be zero
    check_zero_division(x)  # E: dimension integer division by zero
    reveal_type(check_overflow(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(check_overflow_subtract(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(check_overflow_multiply(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(check_overflow_floor_divide(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(check_overflow_negative_literal(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(check_exact_modulo(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(check_used_overflow(x))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    check_invalid_right_operand(x)  # E: Flag integer modulo by zero
    check_unknown_modulo_zero(x)  # E: dimension integer modulo by zero
    check_nested_invalid(x)  # E: Flag integer modulo by zero
    check_comparison(x)  # E: Flag integer division by zero
    check_membership(x)  # E: range() arg 3 must not be zero
    reveal_type(check_unknown_then_false(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(check_unknown_then_true(x))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(check_unknown_before_invalid(x))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    check_known_before_invalid(x)  # E: Flag integer modulo by zero
    check_invalid_before_unknown(x)  # E: Flag integer modulo by zero
    reveal_type(check_false_before_invalid(x))  # E: revealed type: Tensor[[2, 3]]
"#,
);

/// Pins the invariant `iterate_int_tuple` relies on: an unpacked shape's middle
/// is always gradual, because a concrete one flattens into the prefix.
#[test]
fn test_int_tuple_unpacked_middle_is_always_gradual() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import Elements, IntTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: IntTuple, DType]: ...

def variadic[S: IntTuple](x: Array[[2, *Elements[S], 3], int]) -> IntTuple[2, *Elements[S], 3]: ...
def make() -> Array[[2, 4, 5, 3], int]: ...

concrete = variadic(make())
symbolic: IntTuple[2, *Elements[IntTuple], 3]
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let solutions = state.transaction().get_solutions(&main).unwrap();
    for name in ["concrete", "symbolic"] {
        let ty = solutions.get(&KeyExport(Name::new(name)));
        let Type::IntTuple(shape) = ty else {
            panic!("expected `{name}` to solve to an `IntTuple`, got `{ty}`");
        };
        match shape.to_tuple() {
            // Flattened to a fixed length, so `iterate_int_tuple` never sees a middle.
            Tuple::Concrete(_) | Tuple::Unbounded(_) => {}
            Tuple::Unpacked(unpacked) => {
                let middle = unpacked.middle();
                assert!(
                    matches!(middle, Type::IntTuple(s) if s.is_shapeless())
                        || matches!(middle, Type::Tuple(Tuple::Unbounded(elt))
                            if elt.is_any() || is_gradual_size(elt)),
                    "`{name}` has non-gradual unpacked middle `{middle}`; folding the \
                     ends into the element type would now double-count them"
                );
            }
        }
    }
}

testcase!(
    test_type_shape_dsl_int_tuple_values,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Elements, Int, IntTuple, IntVar, type_shape_dsl_function
from shape_extensions.dsl import Invalid as invalid_alias
from shape_extensions.dsl import IntTuple as make_shape
from torch import Tensor
from typing import Literal, assert_type

@type_shape_dsl_function
def reorder(shape: IntTuple) -> IntTuple:
    if len(shape) == 3:
        return dsl.IntTuple((shape[-1], shape[0], 7))
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def imported_alias(shape: IntTuple) -> IntTuple:
    return make_shape((shape[0],))

@type_shape_dsl_function
def boundaries(shape: IntTuple) -> IntTuple:
    if len(shape) == 3:
        return dsl.IntTuple((shape[-3], shape[2], +7))
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def empty(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(())

@type_shape_dsl_function
def out_of_bounds(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[3],))

@type_shape_dsl_function
def negative_out_of_bounds(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[-4],))

@type_shape_dsl_function
def unknown_then_out_of_bounds(unknown: Int, shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((unknown, shape[3]))

@type_shape_dsl_function
def rank_two_prefix(shape: IntTuple) -> IntTuple:
    if len(shape) == 2:
        return dsl.IntTuple((shape[0],))
    return dsl.Invalid("expected rank two")

@type_shape_dsl_function
def first_dimension(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[0],))

@type_shape_dsl_function
def explicit_unknown(shape: IntTuple) -> IntTuple:
    if len(shape) == 1:
        return shape
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def explicit_invalid(shape: IntTuple) -> IntTuple:
    if len(shape) == 1:
        return shape
    return invalid_alias("expected rank one")

@type_shape_dsl_function
def always_invalid(dim: Int) -> Int:
    return dsl.Invalid("no integer result")

@type_shape_dsl_function
def identity(shape: IntTuple) -> IntTuple:
    return shape

@type_shape_dsl_function
def first(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[0],))

@type_shape_dsl_function
def require_rank_two(shape: IntTuple) -> IntTuple:
    if len(shape) == 2:
        return shape
    return dsl.Invalid("expected rank two")

def apply_reorder[S: IntTuple](x: Tensor[S]) -> Tensor[reorder(S)]: ...
def apply_alias[S: IntTuple](x: Tensor[S]) -> Tensor[imported_alias(S)]: ...
def apply_boundaries[S: IntTuple](x: Tensor[S]) -> Tensor[boundaries(S)]: ...
def apply_empty[S: IntTuple](x: Tensor[S]) -> Tensor[empty(S)]: ...
def apply_oob[S: IntTuple](x: Tensor[S]) -> Tensor[out_of_bounds(S)]: ...
def apply_negative_oob[S: IntTuple](x: Tensor[S]) -> Tensor[negative_out_of_bounds(S)]: ...
def apply_unknown_then_oob[S: IntTuple](x: Tensor[S]) -> Tensor[unknown_then_out_of_bounds(int, S)]: ...
def apply_rank_two_prefix[S: IntTuple](x: Tensor[S]) -> Tensor[rank_two_prefix(S)]: ...
def apply_first_dimension[S: IntTuple](x: Tensor[S]) -> Tensor[first_dimension(S)]: ...
def apply_unknown[S: IntTuple](x: Tensor[S]) -> Tensor[explicit_unknown(S)]: ...
def apply_invalid[S: IntTuple](x: Tensor[S]) -> Tensor[explicit_invalid(S)]: ...
def apply_invalid_int() -> Tensor[[always_invalid(Int[1])]]: ...
def apply_identity[S: IntTuple](x: Tensor[S]) -> Tensor[identity(S)]: ...
def apply_first[S: IntTuple](x: Tensor[S]) -> Tensor[first(S)]: ...
def apply_require_rank_two[S: IntTuple](x: Tensor[S]) -> Tensor[require_rank_two(S)]: ...

def test[N: IntVar, S: IntTuple](concrete: Tensor[[2, 3, 4]], symbolic: Tensor[[N, 3, 4]], gradual: Tensor[IntTuple], unpacked: Tensor[IntTuple[2, *Elements[S]]], tuple_carrier: Tensor[tuple[Literal[2], Literal[3]]]) -> None:
    assert_type(apply_reorder(concrete), Tensor[[4, 2, 7]])
    assert_type(apply_reorder(symbolic), Tensor[[4, N, 7]])
    assert_type(apply_alias(concrete), Tensor[[2]])
    assert_type(apply_boundaries(concrete), Tensor[[2, 4, 7]])
    assert_type(apply_empty(concrete), Tensor[[]])
    assert_type(apply_reorder(gradual), Tensor[IntTuple])
    assert_type(apply_rank_two_prefix(gradual), Tensor[IntTuple])
    assert_type(apply_first_dimension(gradual), Tensor[IntTuple])
    assert_type(apply_rank_two_prefix(unpacked), Tensor[IntTuple])
    assert_type(apply_first_dimension(unpacked), Tensor[[2]])
    assert_type(apply_unknown(concrete), Tensor[IntTuple])
    apply_oob(concrete)  # E: Cannot evaluate type-level shape DSL call: IntTuple index out of bounds
    apply_negative_oob(concrete)  # E: Cannot evaluate type-level shape DSL call: IntTuple index out of bounds
    apply_unknown_then_oob(concrete)  # E: Cannot evaluate type-level shape DSL call: IntTuple index out of bounds
    apply_invalid(concrete)  # E: Cannot evaluate type-level shape DSL call: expected rank one
    apply_invalid_int()  # E: Cannot evaluate type-level shape DSL call: no integer result
    assert_type(apply_identity(tuple_carrier), Tensor[[2, 3]])
    assert_type(apply_first(tuple_carrier), Tensor[[2]])
    assert_type(apply_require_rank_two(tuple_carrier), Tensor[[2, 3]])
"#,
);

testcase!(
    test_type_shape_dsl_lowers_tuple_carrier_parameters,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Elements, IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import assert_type

@type_shape_dsl_function
def identity(shape: IntTuple) -> IntTuple:
    return shape

def echo[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[Shape]: ...
def dsl_echo[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[identity(Shape)]: ...

def test[Batch: IntTuple](
    x: Tensor[IntTuple[2, *Elements[Batch], 3]],
    concrete: Tensor[[4, 5]],
) -> None:
    assert_type(echo(x), Tensor[[2, *Elements[Batch], 3]])
    assert_type(dsl_echo(x), Tensor[[2, *Elements[Batch], 3]])
    assert_type(dsl_echo(concrete), Tensor[[4, 5]])
"#,
);

testcase!(
    test_type_shape_dsl_invalid_int_tuple_values,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function
from shape_extensions.dsl import IntTuple as body_int_tuple
from torch import Tensor
from typing import Any

def make_shape(values: tuple[int, ...]) -> IntTuple: ...

body_int_tuple_alias = body_int_tuple

def Invalid(message: str) -> Any: ...

@type_shape_dsl_function
def local_lookalike(shape: IntTuple) -> IntTuple:
    return make_shape((shape[0],))  # E: @type_shape_dsl_function return value must be

@type_shape_dsl_function
def value_alias(shape: IntTuple) -> IntTuple:
    return body_int_tuple_alias((shape[0],))

@type_shape_dsl_function
def local_invalid(shape: IntTuple) -> IntTuple:
    return Invalid("bad")  # E: @type_shape_dsl_function return value must be

@type_shape_dsl_function
def shadowed_invalid(Invalid: IntTuple) -> IntTuple:
    return Invalid("bad")  # E: @type_shape_dsl_function return value must be  # E: Expected a callable

@type_shape_dsl_function
def list_argument(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple([shape[0]])  # E: @type_shape_dsl_function `dsl.IntTuple` argument must be a fixed tuple

@type_shape_dsl_function
def generator_argument(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(x for x in shape)

@type_shape_dsl_function
def mutation(shape: IntTuple) -> IntTuple:
    shape[0] = 1  # E: @type_shape_dsl_function body supports only `if` and `return`  # E: Cannot set item
    return shape

@type_shape_dsl_function
def nonliteral_index(shape: IntTuple, index: int) -> IntTuple:
    return dsl.IntTuple((shape[index],))

@type_shape_dsl_function
def wrong_flag_index(shape: IntTuple, choose: bool) -> IntTuple:
    return dsl.IntTuple((shape[choose],))  # E: Flag operation requires a compatible Flag parameter

@type_shape_dsl_function
def wrong_index_domain(dim: Int) -> IntTuple:
    return dsl.IntTuple((dim[0],))  # E: len and indexing require an `IntTuple` parameter  # E: not subscriptable

@type_shape_dsl_function
def wrong_len_domain(dim: Int) -> IntTuple:
    if len(dim) == 1:  # E: len and indexing require an `IntTuple` parameter  # E: not assignable
        return dsl.IntTuple((dim,))
    return dsl.IntTuple(())

@type_shape_dsl_function
def wrong_element_domain(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape,))  # E: IntTuple elements must be annotated as `Int`

@type_shape_dsl_function
def wrong_result(dim: Int) -> Int:
    return dsl.IntTuple((dim,))  # E: returned shape expression requires an `IntTuple` result  # E: Returned type

@type_shape_dsl_function
def invalid_message(shape: IntTuple, message: str) -> IntTuple:
    return dsl.Invalid(message)  # E: @type_shape_dsl_function `dsl.Invalid` requires exactly one positional string literal

@type_shape_dsl_function
def invalid_keyword(shape: IntTuple) -> IntTuple:
    return dsl.Invalid(message="bad")  # E: @type_shape_dsl_function `dsl.Invalid` requires exactly one positional string literal

@type_shape_dsl_function
def gradual_arguments(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple.gradual(1)  # E: @type_shape_dsl_function gradual return does not accept arguments  # E: Expected 0 positional arguments

@type_shape_dsl_function
def unsupported_unary(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((~1,))  # E: @type_shape_dsl_function dimension literal supports only unary `+` or `-`

def invalid_metadata() -> Tensor[local_lookalike(IntTuple[2])]: ...  # E: Expected a type-level DSL function
"#,
);

testcase!(
    test_type_shape_dsl_flag_sequence_count,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def tuple_count(shape: IntTuple, axes: tuple[int, ...]) -> IntTuple:
    if axes.count(0) == 2:
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def range_count(shape: IntTuple) -> IntTuple:
    axes = range(0, 5, 2)
    matches = axes.count(2)
    if matches == 1 and axes.count(1) == 0:
        return dsl.IntTuple(())
    return shape

def apply_tuple[S: IntTuple, A: Flag[tuple[int, ...]]](
    x: Tensor[S], axes: A,
) -> Tensor[tuple_count(S, A)]: ...
def apply_range[S: IntTuple](x: Tensor[S]) -> Tensor[range_count(S)]: ...
def apply_unknown(x: Tensor[[2, 3]]) -> Tensor[
    tuple_count(IntTuple[2, 3], tuple[int, ...])
]: ...

def test(x: Tensor[[2, 3]]) -> None:
    assert_type(apply_tuple(x, (0, 0)), Tensor[[]])
    assert_type(apply_tuple(x, (0, 1)), Tensor[[2, 3]])
    assert_type(apply_range(x), Tensor[[]])
    reveal_type(apply_unknown(x))  # E: revealed type: Tensor[tuple[Unknown, ...]]
"#,
);

testcase!(
    test_type_shape_dsl_invalid_flag_sequence_count,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def invalid_receiver(shape: IntTuple) -> IntTuple:
    axis = 0
    if axis.count(0) > 0:  # E: Flag value has the wrong domain for this operation  # E: has no attribute `count`
        return shape
    return shape

@type_shape_dsl_function
def invalid_arity(shape: IntTuple) -> IntTuple:
    axes = (0, 1)
    if axes.count() > 0:  # E: Flag sequence `.count` requires exactly one positional argument  # E: Missing positional argument
        return shape
    return shape
"#,
);

testcase!(
    test_type_shape_dsl_bounded_generators,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, Int, IntTuple, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def copy_shape(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(dim for dim in shape)

@type_shape_dsl_function
def reflexive_filter(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(dim for dim in shape if dim == dim)

@type_shape_dsl_function
def from_range(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(index if index > 0 else 7 for index in range(len(shape)) if index != 1)

@type_shape_dsl_function
def from_sequence(shape: IntTuple) -> IntTuple:
    values = (2, 3, 5)
    return dsl.IntTuple(value for value in values if value != 3)

@type_shape_dsl_function
def captured_filter(shape: IntTuple, axis: int) -> IntTuple:
    return dsl.IntTuple(index for index in range(len(shape)) if index != axis)

@type_shape_dsl_function
def captured_dimension(shape: IntTuple, dimension: Int) -> IntTuple:
    return dsl.IntTuple(dimension for index in range(1))

@type_shape_dsl_function
def flags(shape: IntTuple) -> IntTuple:
    axes = tuple(index for index in range(len(shape)) if index != 0)
    if 1 in axes:
        return dsl.IntTuple((shape[0],))
    return dsl.IntTuple(())

@type_shape_dsl_function
def dimension_flags(shape: IntTuple) -> IntTuple:
    axes = tuple(dim for dim in shape)
    if 3 in axes:
        return dsl.IntTuple((shape[0],))
    return dsl.IntTuple(())

@type_shape_dsl_function
def narrowed_flag_source(
    shape: IntTuple, axis: int | tuple[int, ...] | None,
) -> IntTuple:
    if axis is None:
        return dsl.IntTuple(())
    elif dsl.is_int_value(axis):
        return dsl.IntTuple(())
    return dsl.IntTuple(item for item in axis)

@type_shape_dsl_function
def empty(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(index for index in range(0))

@type_shape_dsl_function
def bounded_fallback(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(index for index in range(4097))

@type_shape_dsl_function
def lazy_unknown_filter(shape: IntTuple, axis: int) -> IntTuple:
    values = tuple(item // 0 for item in range(1) if item == axis)  # E: Cannot divide by zero
    if 0 in values:
        return shape
    return shape

@type_shape_dsl_function
def later_included_invalid(shape: IntTuple, axis: int) -> IntTuple:
    values = tuple(1 // (item - 1) for item in range(2) if item != 0 or item == axis)
    return shape

@type_shape_dsl_function
def bounded_prefix_error(shape: IntTuple) -> IntTuple:
    values = tuple(1 // item for item in range(4097))
    return shape

@type_shape_dsl_function
def shadowed(shape: IntTuple, index: Int) -> IntTuple:
    axes = tuple(index for index in range(2))
    if index == index:
        return shape
    return dsl.IntTuple(())

def apply_copy[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[copy_shape(Shape)]: ...
def apply_reflexive[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[reflexive_filter(Shape)]: ...
def apply_range[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[from_range(Shape)]: ...
def apply_sequence[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[from_sequence(Shape)]: ...
def apply_capture[Shape: IntTuple, Axis: Flag[int]](
    x: Tensor[Shape], axis: Axis,
) -> Tensor[captured_filter(Shape, Axis)]: ...
def apply_flags[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[flags(Shape)]: ...
def apply_dimension[Dimension: IntVar](x: Tensor[[Dimension]]) -> Tensor[captured_dimension(IntTuple[2], Dimension)]: ...
def apply_dimension_flags[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[dimension_flags(Shape)]: ...
def apply_narrowed_source[Axis: Flag[int | tuple[int, ...] | None]](
    axis: Axis,
) -> Tensor[narrowed_flag_source(IntTuple[2, 3], Axis)]: ...
def apply_empty[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[empty(Shape)]: ...
def apply_bounded[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[bounded_fallback(Shape)]: ...
def anonymous_gradual(x: Tensor[[int, 3]]) -> Tensor[copy_shape(IntTuple[int, 3])]: ...
def lazy_fallback(x: Tensor[[2]]) -> Tensor[lazy_unknown_filter(IntTuple[2], int)]: ...
def apply_later_invalid(x: Tensor[[2]]) -> Tensor[later_included_invalid(IntTuple[2], int)]: ...
def apply_prefix_error(x: Tensor[[2]]) -> Tensor[bounded_prefix_error(IntTuple[2])]: ...
def apply_shadowed[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[shadowed(Shape, Int[9])]: ...

def broad() -> Tensor[captured_filter(IntTuple[2, 3], int)]: ...
def broad_dimension(x: Tensor[[2]]) -> Tensor[captured_dimension(IntTuple[2], Int)]: ...

def test[N: IntVar](concrete: Tensor[[2, 3, 4]], symbolic: Tensor[[N, 3]], one_dim: Tensor[[N]], literal: Tensor[[2]], anonymous: Tensor[[int, 3]], gradual: Tensor[IntTuple]) -> None:
    assert_type(apply_copy(concrete), Tensor[[2, 3, 4]])
    assert_type(apply_copy(symbolic), Tensor[[N, 3]])
    assert_type(apply_reflexive(symbolic), Tensor[[N, 3]])
    reveal_type(apply_copy(gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_range(concrete), Tensor[[7, 2]])
    assert_type(apply_sequence(concrete), Tensor[[2, 5]])
    reveal_type(apply_capture(concrete, 1))  # E: revealed type: Tensor[[0, 2]]
    reveal_type(broad())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_flags(concrete), Tensor[[2]])
    assert_type(apply_dimension(one_dim), Tensor[[N]])
    assert_type(broad_dimension(literal), Tensor[[int]])
    assert_type(apply_dimension_flags(concrete), Tensor[[2]])
    assert_type(apply_narrowed_source((2, 3)), Tensor[[2, 3]])
    assert_type(apply_empty(concrete), Tensor[[]])
    reveal_type(apply_bounded(concrete))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(anonymous_gradual(anonymous), Tensor[[int, 3]])
    reveal_type(lazy_fallback(literal))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    apply_later_invalid(literal)  # E: Flag integer division by zero
    apply_prefix_error(literal)  # E: Flag integer division by zero
    assert_type(apply_shadowed(concrete), Tensor[[2, 3, 4]])
"#,
);

testcase!(
    test_type_shape_dsl_invalid_bounded_generators,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, type_shape_dsl_function
from torch import Tensor

@type_shape_dsl_function
def multiple(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(x for x in range(2) for y in range(2))  # E: generators require exactly one

@type_shape_dsl_function
def destructured(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(x for x, y in ((1, 2),))  # E: generator target must be exactly one bare name

@type_shape_dsl_function
def arbitrary_iterator(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(x for x in [1, 2])  # E: generator source must be an IntTuple

@type_shape_dsl_function
def multiple_filters(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(x for x in range(3) if x != 0 if x != 1)  # E: support at most one

@type_shape_dsl_function
def wrong_inttuple_element(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple(shape for x in range(1))  # E: IntTuple elements must be

@type_shape_dsl_function
def wrong_tuple_element(shape: IntTuple) -> IntTuple:
    values = tuple(shape for x in range(1))  # E: Flag operation requires a compatible Flag parameter
    return shape

@type_shape_dsl_function
def nested(shape: IntTuple) -> IntTuple:
    values = tuple(x for x in tuple(y for y in range(2)))  # E: nested generators are not supported
    return shape

async def async_values():
    yield 1

@type_shape_dsl_function
def async_generator(shape: IntTuple) -> IntTuple:
    values = tuple(item async for item in async_values())  # E: async generators are not supported  # E: not assignable to parameter `iterable`
    return shape

@type_shape_dsl_function
def mutation(shape: IntTuple) -> IntTuple:
    captured = 0
    values = tuple((captured := item) for item in range(2))  # E: Flag integer expression is not supported
    return shape

@type_shape_dsl_function
def escaped(shape: IntTuple) -> IntTuple:
    values = tuple(item for item in range(2))
    if item == 0:  # E: local value must be assigned before use  # E: Could not find name
        return shape
    return shape

@type_shape_dsl_function
def invalid_element(shape: IntTuple) -> IntTuple:
    values = tuple(item // 0 for item in range(2))  # E: Cannot divide by zero
    return shape

@type_shape_dsl_function
def invalid_source(shape: IntTuple) -> IntTuple:
    values = tuple(item for item in range(0, 2, 0))
    return shape

def apply_invalid_element(x: Tensor[[2, 3]]) -> Tensor[invalid_element(IntTuple[2, 3])]: ...
def apply_invalid_source(x: Tensor[[2, 3]]) -> Tensor[invalid_source(IntTuple[2, 3])]: ...

def test(x: Tensor[[2, 3]]) -> None:
    apply_invalid_element(x)  # E: Flag integer division by zero
    apply_invalid_source(x)  # E: range() arg 3 must not be zero
"#,
);

testcase!(
    test_type_shape_dsl_shared_generator_budget,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def exact_budget(shape: IntTuple) -> IntTuple:
    first = tuple(item for item in range(2048))
    return dsl.IntTuple(7 for item in range(2048) if item == 0)

@type_shape_dsl_function
def shared_overflow(shape: IntTuple) -> IntTuple:
    first = tuple(item for item in range(2048))
    return dsl.IntTuple(7 for item in range(2049) if item == 0)

@type_shape_dsl_function
def prefix_error(shape: IntTuple) -> IntTuple:
    first = tuple(item for item in range(4095))
    second = tuple(1 // item for item in range(2))
    return shape

@type_shape_dsl_function
def beyond_budget_error(shape: IntTuple) -> IntTuple:
    first = tuple(item for item in range(4096))
    second = tuple(1 // item for item in range(1))
    if 0 in second:
        return dsl.IntTuple((7,))
    return shape

@type_shape_dsl_function
def nested_budget(shape: IntTuple) -> IntTuple:
    values = tuple(
        outer for outer in range(3000) if outer in tuple(inner for inner in (1, 2))
    )
    if 0 in values:
        return dsl.IntTuple((7,))
    return shape

def apply_exact() -> Tensor[exact_budget(IntTuple[2])]: ...
def apply_overflow() -> Tensor[shared_overflow(IntTuple[2])]: ...
def apply_prefix_error() -> Tensor[prefix_error(IntTuple[2])]: ...
def apply_beyond_budget_error() -> Tensor[beyond_budget_error(IntTuple[2])]: ...
def apply_nested_budget() -> Tensor[nested_budget(IntTuple[2])]: ...

def test() -> None:
    assert_type(apply_exact(), Tensor[[7]])
    reveal_type(apply_overflow())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    apply_prefix_error()  # E: Flag integer division by zero
    reveal_type(apply_beyond_budget_error())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_nested_budget())  # E: revealed type: Tensor[tuple[Unknown, ...]]
"#,
);

testcase!(
    test_type_shape_dsl_any,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, IntTuple, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def from_flag_sequence(shape: IntTuple, axes: tuple[int, ...]) -> IntTuple:
    if any(axis == 1 for axis in axes):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def from_shape(shape: IntTuple) -> IntTuple:
    if any(dimension == 3 for dimension in shape):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def filtered(shape: IntTuple, axis: int) -> IntTuple:
    if any(item == 1 for item in range(3) if item == axis):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def lazy_true(shape: IntTuple) -> IntTuple:
    if any(item == 0 or 1 // (item - 1) > 0 for item in range(2)):
        return shape
    return dsl.IntTuple(())

@type_shape_dsl_function
def unknown_before_error(shape: IntTuple, axis: int) -> IntTuple:
    if any((item == 0 and item == axis) or (item != 0 and 1 // (item - 1) > 0) for item in range(2)):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def unknown_filter_then_true(shape: IntTuple, axis: int) -> IntTuple:
    if any(item == 1 for item in range(2) if item == axis or item == 1):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def unknown_filter_guards_error(shape: IntTuple, axis: int) -> IntTuple:
    if any(1 // item > 0 for item in range(2) if axis == 0 or item == 1):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def capped_false(shape: IntTuple) -> IntTuple:
    if any(item == 4096 for item in range(4097)):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def capped_prefix_true(shape: IntTuple) -> IntTuple:
    if any(item == 4095 for item in range(4097)):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def capped_prefix_error(shape: IntTuple) -> IntTuple:
    if any(1 // item > 0 for item in range(4097)):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def nested_precise(shape: IntTuple) -> IntTuple:
    if any(any(inner == outer for inner in range(2)) for outer in range(2)):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def nested_exhausted(shape: IntTuple) -> IntTuple:
    if any(any(inner == -1 for inner in range(4096)) for outer in range(2)):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def nested_guarded_error(shape: IntTuple, axis: int) -> IntTuple:
    if any(
        any(1 // inner > 0 for inner in range(2) if axis == 0 or inner == 1)
        or outer == 1
        for outer in range(2)
    ):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def budget_after_possible_error(shape: IntTuple, axis: int) -> IntTuple:
    if any(
        any(1 // inner < 0 for inner in range(4096) if axis == 0 or inner > 0)
        for outer in range(2)
    ) or 1 == 1:
        return dsl.IntTuple(())
    return shape

def apply_flags[Axes: Flag[tuple[int, ...]]](axes: Axes) -> Tensor[from_flag_sequence(IntTuple[2, 3], Axes)]: ...
def broad_flags() -> Tensor[from_flag_sequence(IntTuple[2, 3], tuple[int, ...])]: ...
def apply_shape[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[from_shape(Shape)]: ...
def apply_filtered[Axis: Flag[int]](axis: Axis) -> Tensor[filtered(IntTuple[2, 3], Axis)]: ...
def broad_filtered() -> Tensor[filtered(IntTuple[2, 3], int)]: ...
def apply_lazy() -> Tensor[lazy_true(IntTuple[2, 3])]: ...
def apply_unknown[Axis: Flag[int]](axis: Axis) -> Tensor[unknown_before_error(IntTuple[2, 3], Axis)]: ...
def broad_unknown() -> Tensor[unknown_before_error(IntTuple[2, 3], int)]: ...
def broad_filter_then_true() -> Tensor[unknown_filter_then_true(IntTuple[2, 3], int)]: ...
def apply_guarded_error[Axis: Flag[int]](axis: Axis) -> Tensor[unknown_filter_guards_error(IntTuple[2, 3], Axis)]: ...
def broad_guarded_error() -> Tensor[unknown_filter_guards_error(IntTuple[2, 3], int)]: ...
def apply_capped_false() -> Tensor[capped_false(IntTuple[2, 3])]: ...
def apply_capped_true() -> Tensor[capped_prefix_true(IntTuple[2, 3])]: ...
def apply_capped_error() -> Tensor[capped_prefix_error(IntTuple[2, 3])]: ...
def apply_nested_precise() -> Tensor[nested_precise(IntTuple[2, 3])]: ...
def apply_nested_exhausted() -> Tensor[nested_exhausted(IntTuple[2, 3])]: ...
def apply_nested_guarded_error[Axis: Flag[int]](axis: Axis) -> Tensor[nested_guarded_error(IntTuple[2, 3], Axis)]: ...
def broad_nested_guarded_error() -> Tensor[nested_guarded_error(IntTuple[2, 3], int)]: ...
def broad_budget_after_error() -> Tensor[budget_after_possible_error(IntTuple[2, 3], int)]: ...

def test[N: IntVar](symbolic: Tensor[[N, 3]], one_symbolic: Tensor[[N]]) -> None:
    assert_type(apply_flags((0, 1)), Tensor[[]])
    assert_type(apply_flags((0, 2)), Tensor[[2, 3]])
    reveal_type(broad_flags())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_shape(symbolic), Tensor[[]])
    reveal_type(apply_shape(one_symbolic))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_filtered(1), Tensor[[]])
    assert_type(apply_filtered(4), Tensor[[2, 3]])
    reveal_type(broad_filtered())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_lazy(), Tensor[[2, 3]])
    assert_type(apply_unknown(0), Tensor[[]])
    reveal_type(broad_unknown())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(broad_filter_then_true(), Tensor[[]])
    assert_type(apply_guarded_error(1), Tensor[[]])
    apply_guarded_error(0)  # E: Flag integer division by zero
    reveal_type(broad_guarded_error())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    apply_unknown(2)  # E: Flag integer division by zero
    reveal_type(apply_capped_false())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_capped_true(), Tensor[[]])
    apply_capped_error()  # E: Flag integer division by zero
    assert_type(apply_nested_precise(), Tensor[[]])
    reveal_type(apply_nested_exhausted())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_nested_guarded_error(1), Tensor[[]])
    apply_nested_guarded_error(0)  # E: Flag integer division by zero
    reveal_type(broad_nested_guarded_error())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(broad_budget_after_error())  # E: revealed type: Tensor[tuple[Unknown, ...]]
"#,
);

testcase!(
    test_type_shape_dsl_invalid_any,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def no_arguments(shape: IntTuple) -> IntTuple:
    if any():  # E: `any` requires exactly one positional boolean generator  # E: Missing positional argument
        return shape
    return shape

@type_shape_dsl_function
def two_arguments(shape: IntTuple) -> IntTuple:
    if any((item == 0 for item in range(1)), (item == 1 for item in range(1))):  # E: `any` requires exactly one positional boolean generator  # E: Expected 1 positional argument
        return shape
    return shape

@type_shape_dsl_function
def not_a_generator(shape: IntTuple) -> IntTuple:
    if any((True, False)):  # E: `any` argument must be a bounded boolean generator
        return shape
    return shape

@type_shape_dsl_function
def invalid_source(shape: IntTuple) -> IntTuple:
    if any(item == 0 for item in [0, 1]):  # E: generator source must be an IntTuple
        return shape
    return shape

@type_shape_dsl_function
def multiple_clauses(shape: IntTuple) -> IntTuple:
    if any(left == right for left in range(2) for right in range(2)):  # E: `any` generators require exactly one `for` clause
        return shape
    return shape

@type_shape_dsl_function
def multiple_filters(shape: IntTuple) -> IntTuple:
    if any(item == 0 for item in range(2) if item >= 0 if item <= 1):  # E: `any` generators support at most one `if` filter
        return shape
    return shape

@type_shape_dsl_function
def non_boolean_element(shape: IntTuple) -> IntTuple:
    if any(item for item in range(2)):  # E: a name used directly as a condition requires a `Flag[bool]` value
        return shape
    return shape
"#,
);

#[test]
fn test_type_shape_dsl_diamond_graph_is_flat_and_depth_bounded() {
    assert_eq!(MAX_HELPER_GRAPH_NODES, 4096);
    assert_eq!(MAX_HELPER_GRAPH_EDGES, 16384);
    let mut source = r#"
from shape_extensions import IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def helper_0(shape: IntTuple, choice: int) -> IntTuple:
    return shape
"#
    .to_owned();
    for level in 1..=33 {
        source.push_str(&format!(
            r#"
@type_shape_dsl_function
def helper_{level}(shape: IntTuple, choice: int) -> IntTuple:
    if choice < {level}:
        return helper_{previous}(shape, choice)
    return helper_{previous}(shape, choice)
"#,
            previous = level - 1,
        ));
    }
    let mut env = shaped_array_env();
    env.add("main", &source);
    let (state, handle) = env.to_state();
    let main = handle("main");
    let solutions = state
        .transaction()
        .get_solutions(&main)
        .expect("diamond helper module should solve");
    let helper_32 = solutions.get(&KeyExport(Name::new("helper_32")));
    assert!(
        matches!(helper_32, Type::Function(function)
            if matches!(&function.metadata.kind,
                FunctionKind::TypeShapeDsl(_, resolved)
                    if resolved.helper_graph_metrics() == (33, 64, 32))),
        "expected a flat 33-node/64-edge depth-32 graph, got `{helper_32}`",
    );
    let helper_33 = solutions.get(&KeyExport(Name::new("helper_33")));
    assert!(
        matches!(helper_33, Type::Function(function)
            if matches!(&function.metadata.kind, FunctionKind::Def(_))),
        "depth-33 helper should recover as an ordinary function, got `{helper_33}`",
    );
    let errors = state
        .transaction()
        .get_errors([&main])
        .collect_display_errors();
    assert!(
        errors
            .iter()
            .any(|error| error.msg().contains("DSL helper call depth exceeds 32")),
        "expected a depth-bound diagnostic, got {errors:?}",
    );
}

testcase!(
    test_type_shape_dsl_helpers,
    {
        let mut env = shape_dsl_tensor_env();
        env.add_with_path(
            "dsl_helpers",
            "dsl_helpers.pyi",
            r#"
import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def leaf(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[0],))

@type_shape_dsl_function
def identity(shape: IntTuple) -> IntTuple:
    return shape

@type_shape_dsl_function
def middle(shape: IntTuple) -> IntTuple:
    return leaf(shape)

@type_shape_dsl_function
def int_leaf(dimension: Int) -> Int:
    return dimension

@type_shape_dsl_function
def axis_helper(shape: IntTuple, axis: int) -> IntTuple:
    if axis == 0:
        return leaf(shape)
    return shape

@type_shape_dsl_function
def unknown(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def invalid(shape: IntTuple) -> IntTuple:
    return dsl.Invalid("helper rejected shape")
"#,
        );
        env
    },
    r#"
import dsl_helpers as qualified
import shape_extensions.dsl as dsl
from dsl_helpers import axis_helper, identity, int_leaf, invalid, leaf as imported_leaf, unknown
from shape_extensions import Int, IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

leaf_alias = imported_leaf

@type_shape_dsl_function
def imported(shape: IntTuple) -> IntTuple:
    return leaf_alias(shape)

@type_shape_dsl_function
def propagate_argument(shape: IntTuple) -> IntTuple:
    return identity(shape)

@type_shape_dsl_function
def helper_of_helper(shape: IntTuple) -> IntTuple:
    return qualified.middle(shape)

@type_shape_dsl_function
def local_argument(shape: IntTuple) -> Int:
    dimension = shape[0]
    return int_leaf(dimension)

@type_shape_dsl_function
def parameter_flag(shape: IntTuple, axis: int) -> IntTuple:
    return axis_helper(shape, axis)

@type_shape_dsl_function
def local_flag(shape: IntTuple) -> IntTuple:
    axis = 0
    return axis_helper(shape, axis)

@type_shape_dsl_function
def axis_without_none(shape: IntTuple, axis: int | tuple[int, ...]) -> IntTuple:
    if dsl.is_int_value(axis):
        return axis_helper(shape, axis)
    return shape

@type_shape_dsl_function
def narrowed_union(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    if axis is None:
        return shape
    return axis_without_none(shape, axis)

@type_shape_dsl_function
def diamond(shape: IntTuple, choice: int) -> IntTuple:
    if choice < 1:
        return imported(shape)
    return helper_of_helper(shape)

@type_shape_dsl_function
def joined_argument(first: IntTuple, second: IntTuple, choice: int) -> IntTuple:
    if choice < 1:
        selected = first
    else:
        selected = second
    return imported_leaf(selected)

@type_shape_dsl_function
def propagate_gradual(shape: IntTuple) -> IntTuple:
    return unknown(shape)

@type_shape_dsl_function
def propagate_invalid(shape: IntTuple) -> IntTuple:
    return invalid(shape)

def apply_imported(x: Tensor[[2, 3]]) -> Tensor[imported(IntTuple[2, 3])]: ...
def apply_gradual_argument() -> Tensor[propagate_argument(IntTuple)]: ...
def apply_nested(x: Tensor[[2, 3]]) -> Tensor[helper_of_helper(IntTuple[2, 3])]: ...
def apply_local(x: Tensor[[2, 3]]) -> Tensor[[local_argument(IntTuple[2, 3])]]: ...
def apply_parameter_flag(x: Tensor[[2, 3]]) -> Tensor[parameter_flag(IntTuple[2, 3], 0)]: ...
def apply_local_flag(x: Tensor[[2, 3]]) -> Tensor[local_flag(IntTuple[2, 3])]: ...
def apply_narrowed_union(x: Tensor[[2, 3]]) -> Tensor[narrowed_union(IntTuple[2, 3], 0)]: ...
def apply_diamond(x: Tensor[[2, 3]]) -> Tensor[diamond(IntTuple[2, 3], 0)]: ...
def apply_joined_first(x: Tensor[[2, 3]]) -> Tensor[joined_argument(IntTuple[2, 3], IntTuple[4, 5], 0)]: ...
def apply_joined_second(x: Tensor[[2, 3]]) -> Tensor[joined_argument(IntTuple[2, 3], IntTuple[4, 5], 1)]: ...
def apply_unknown(x: Tensor[[2, 3]]) -> Tensor[propagate_gradual(IntTuple[2, 3])]: ...
def apply_invalid(x: Tensor[[2, 3]]) -> Tensor[propagate_invalid(IntTuple[2, 3])]: ...

def test(x: Tensor[[2, 3]]) -> None:
    assert_type(apply_imported(x), Tensor[[2]])
    reveal_type(apply_gradual_argument())  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_nested(x), Tensor[[2]])
    assert_type(apply_local(x), Tensor[[2]])
    assert_type(apply_parameter_flag(x), Tensor[[2]])
    assert_type(apply_local_flag(x), Tensor[[2]])
    assert_type(apply_narrowed_union(x), Tensor[[2]])
    assert_type(apply_diamond(x), Tensor[[2]])
    assert_type(apply_joined_first(x), Tensor[[2]])
    assert_type(apply_joined_second(x), Tensor[[4]])
    reveal_type(apply_unknown(x))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    apply_invalid(x)  # E: helper rejected shape
"#,
);

testcase!(
    test_type_shape_dsl_invalid_helpers,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def shape_helper(shape: IntTuple) -> IntTuple:
    return shape

@type_shape_dsl_function
def int_helper(dimension: Int) -> Int:
    return dimension

@type_shape_dsl_function
def wrong_argument(shape: IntTuple) -> IntTuple:
    return shape_helper(shape, shape)  # E: DSL helper argument domains must exactly match  # E: Expected 1 positional

@type_shape_dsl_function
def wrong_domain(shape: IntTuple) -> IntTuple:
    return int_helper(shape)  # E: DSL helper argument domains must exactly match  # E: Returned type  # E: is not assignable to parameter

@type_shape_dsl_function
def wrong_result(dimension: Int) -> IntTuple:
    return int_helper(dimension)  # E: DSL helper result domain must match  # E: Returned type

def ordinary(shape: IntTuple) -> IntTuple: ...

@type_shape_dsl_function
def arbitrary(shape: IntTuple) -> IntTuple:
    return ordinary(shape)  # E: DSL helper callee must be a validated

@type_shape_dsl_function
def keyword(shape: IntTuple) -> IntTuple:
    return shape_helper(shape=shape)  # E: DSL helper calls accept only positional arguments

@type_shape_dsl_function
def direct_recursive(shape: IntTuple) -> IntTuple:
    return direct_recursive(shape)  # E: recursive DSL helper calls are not supported
"#,
);

testcase!(
    test_type_shape_dsl_helpers_share_generator_budget,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def large_leaf(shape: IntTuple) -> IntTuple:
    if any(item == -1 for item in range(2500)):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def large_root(shape: IntTuple) -> IntTuple:
    if any(item == -1 for item in range(2500)):
        return dsl.IntTuple(())
    return large_leaf(shape)

@type_shape_dsl_function
def small_leaf(shape: IntTuple) -> IntTuple:
    if any(item == -1 for item in range(2000)):
        return dsl.IntTuple(())
    return shape

@type_shape_dsl_function
def small_root(shape: IntTuple) -> IntTuple:
    if any(item == -1 for item in range(2000)):
        return dsl.IntTuple(())
    return small_leaf(shape)

def apply_large(x: Tensor[[2, 3]]) -> Tensor[large_root(IntTuple[2, 3])]: ...
def apply_small(x: Tensor[[2, 3]]) -> Tensor[small_root(IntTuple[2, 3])]: ...

def test(x: Tensor[[2, 3]]) -> None:
    reveal_type(apply_large(x))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_small(x), Tensor[[2, 3]])
"#,
);

testcase!(
    test_type_shape_dsl_boolean_flags,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Flag, IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import Literal, reveal_type

@type_shape_dsl_function
def choose(shape: IntTuple, keep: bool) -> IntTuple:
    alias = keep
    if not alias:
        return shape
    return dsl.IntTuple((1,))

@type_shape_dsl_function
def conjunction(shape: IntTuple, left: bool, right: bool) -> IntTuple:
    if left and right:
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def disjunction(shape: IntTuple, left: bool, right: bool) -> IntTuple:
    if left or right:
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def bool_helper(shape: IntTuple, keep: bool) -> IntTuple:
    if keep:
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def call_bool_helper(shape: IntTuple, keep: bool) -> IntTuple:
    return bool_helper(shape, keep)

@type_shape_dsl_function
def local_literal(shape: IntTuple) -> IntTuple:
    keep = True
    if keep:
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def conditional_local(shape: IntTuple, keep: bool, choose_branch: bool) -> IntTuple:
    local = keep if choose_branch else False
    if local:
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def conditional_helper(shape: IntTuple, keep: bool, choose_branch: bool) -> IntTuple:
    local = keep if choose_branch else False
    return bool_helper(shape, local)

@type_shape_dsl_function
def bool_local_is_not_none(shape: IntTuple) -> IntTuple:
    local = True
    if local is None:  # E: `is None` requires a `Flag[int | tuple[int, ...] | None]` value  # E: Identity comparison
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def mixed_flag_is_not_int(shape: IntTuple, choose_branch: bool) -> IntTuple:
    local = True if choose_branch else 0
    if dsl.is_int_value(local):  # E: `is_int_value` requires a `Flag[int | tuple[int, ...] | None]` value
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def joined_non_bool_is_not_bool(
    shape: IntTuple, candidate: tuple[int, ...], choose_branch: bool,
) -> IntTuple:
    local = candidate if choose_branch else False
    if local:  # E: a name used directly as a condition requires a `Flag[bool]` value
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def joined_bool_is_not_int_comparison(
    shape: IntTuple, candidate: bool, choose_branch: bool,
) -> IntTuple:
    local = candidate if choose_branch else 0
    zero = 0
    if local == zero:  # E: Flag operation requires a compatible Flag parameter
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def wrong_condition(shape: IntTuple, axis: int) -> IntTuple:
    if axis:  # E: a name used directly as a condition requires a `Flag[bool]` value
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def bool_is_not_int(shape: IntTuple) -> IntTuple:
    keep = True
    if dsl.is_int_value(keep):  # E: `is_int_value` requires a `Flag[int | tuple[int, ...] | None]` value
        return dsl.IntTuple((1,))
    return shape

# A parameter's Flag domain is checked after the function body has been validated.
@type_shape_dsl_function
def bool_parameter_is_not_int(shape: IntTuple, keep: bool) -> IntTuple:
    if dsl.is_int_value(keep):  # E: control-flow narrowing requires a Flag[int | tuple[int, ...] | None] value
        return dsl.IntTuple((1,))
    return shape

def apply[Shape: IntTuple, Keep: Flag[bool]](
    x: Tensor[Shape], keep: Keep = True,
) -> Tensor[choose(Shape, Keep)]: ...

def apply_and[Shape: IntTuple, Left: Flag[bool], Right: Flag[bool]](
    x: Tensor[Shape], left: Left, right: Right,
) -> Tensor[conjunction(Shape, Left, Right)]: ...

def apply_or[Shape: IntTuple, Left: Flag[bool], Right: Flag[bool]](
    x: Tensor[Shape], left: Left, right: Right,
) -> Tensor[disjunction(Shape, Left, Right)]: ...

def apply_helper[Shape: IntTuple, Keep: Flag[bool]](
    x: Tensor[Shape], keep: Keep,
) -> Tensor[call_bool_helper(Shape, Keep)]: ...

def apply_literal[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[local_literal(Shape)]: ...

def apply_conditional[Shape: IntTuple, Keep: Flag[bool], Choose: Flag[bool]](
    x: Tensor[Shape], keep: Keep, choose_branch: Choose,
) -> Tensor[conditional_local(Shape, Keep, Choose)]: ...

def apply_conditional_helper[Shape: IntTuple, Keep: Flag[bool], Choose: Flag[bool]](
    x: Tensor[Shape], keep: Keep, choose_branch: Choose,
) -> Tensor[conditional_helper(Shape, Keep, Choose)]: ...

def check(x: Tensor[[2, 3]]) -> None:
    reveal_type(apply(x))  # E: revealed type: Tensor[[1]]
    reveal_type(apply(x, True))  # E: revealed type: Tensor[[1]]
    reveal_type(apply(x, False))  # E: revealed type: Tensor[[2, 3]]

def bool_results(x: Tensor[[2, 3]], broad: bool) -> None:
    reveal_type(apply_and(x, True, True))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_and(x, False, broad))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_or(x, True, broad))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_or(x, False, False))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply(x, broad))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_helper(x, True))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_helper(x, broad))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_literal(x))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_conditional(x, True, True))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_conditional(x, True, False))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_conditional(x, broad, False))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_conditional_helper(x, True, True))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_conditional_helper(x, broad, False))  # E: revealed type: Tensor[[2, 3]]

def union_bool(x: Tensor[[2, 3]], keep: Literal[True, False]) -> None:
    reveal_type(apply(x, keep))  # E: revealed type: Tensor[tuple[Unknown, ...]]
"#,
);

testcase!(
    test_type_shape_dsl_dynamic_int_tuple_index,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Elements, Flag, IntTuple, IntVar, broadcast, type_shape_dsl_function
from torch import Tensor
from typing import reveal_type

@type_shape_dsl_function
def select(shape: IntTuple, index: int) -> IntTuple:
    return dsl.IntTuple((shape[index],))

@type_shape_dsl_function
def select_next(shape: IntTuple, index: int) -> IntTuple:
    next_index = index + 1
    return dsl.IntTuple((shape[next_index],))

@type_shape_dsl_function
def invalid_join_index(
    shape: IntTuple, candidate: tuple[int, ...], choose_branch: bool,
) -> IntTuple:
    index = candidate if choose_branch else 0
    return dsl.IntTuple((shape[index],))  # E: Cannot index into `IntTuple`  # E: Flag operation requires a compatible Flag parameter

@type_shape_dsl_function
def invalid_join_helper(
    shape: IntTuple, candidate: tuple[int, ...], choose_branch: bool,
) -> IntTuple:
    index = candidate if choose_branch else 0
    return select(shape, index)  # E: helper argument domains must exactly match  # E: not assignable to parameter `index`

@type_shape_dsl_function
def narrowed_int_comparison(
    shape: IntTuple, candidate: int | tuple[int, ...],
) -> IntTuple:
    if dsl.is_int_value(candidate):
        zero = 0
        if candidate == zero:
            return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def invalid_join_broadcast(
    shape: IntTuple, candidate: int, choose_branch: bool,
) -> IntTuple:
    right = candidate if choose_branch else 0
    return broadcast(shape, right)  # E: `broadcast` arguments must be IntTuple parameters

@type_shape_dsl_function
def copy_by_binder(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[index] for index in range(len(shape))))

@type_shape_dsl_function
def reverse_by_binder(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[index] for index in range(-1, -4, -1)))

@type_shape_dsl_function
def divide_index(shape: IntTuple, divisor: int) -> IntTuple:
    return dsl.IntTuple((shape[1 // divisor],))

@type_shape_dsl_function
def lazy_index(shape: IntTuple, choose: bool) -> IntTuple:
    if choose:
        return dsl.IntTuple((shape[1 // 0],))  # E: Cannot divide by zero
    return shape

@type_shape_dsl_function
def huge_index(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((shape[999999999999999999999999],))

def apply_select[Shape: IntTuple, Index: Flag[int]](
    x: Tensor[Shape], index: Index,
) -> Tensor[select(Shape, Index)]: ...

def apply_next[Shape: IntTuple, Index: Flag[int]](
    x: Tensor[Shape], index: Index,
) -> Tensor[select_next(Shape, Index)]: ...

def apply_copy[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[copy_by_binder(Shape)]: ...
def apply_reverse[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[reverse_by_binder(Shape)]: ...
def apply_huge[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[huge_index(Shape)]: ...

def apply_divide[Shape: IntTuple, Divisor: Flag[int]](
    x: Tensor[Shape], divisor: Divisor,
) -> Tensor[divide_index(Shape, Divisor)]: ...

def apply_lazy[Shape: IntTuple, Choose: Flag[bool]](
    x: Tensor[Shape], choose: Choose,
) -> Tensor[lazy_index(Shape, Choose)]: ...

def apply_narrowed_comparison[
    Shape: IntTuple, Candidate: Flag[int | tuple[int, ...]],
](x: Tensor[Shape], candidate: Candidate) -> Tensor[narrowed_int_comparison(Shape, Candidate)]: ...

def index_results[N: IntVar, Tail: IntTuple](
    symbolic: Tensor[[N, 3, 4]],
    empty: Tensor[[]],
    gradual: Tensor[IntTuple],
    unpacked: Tensor[IntTuple[2, *Elements[Tail]]],
    broad: int,
) -> None:
    reveal_type(apply_select(symbolic, 0))  # E: revealed type: Tensor[[N]]
    reveal_type(apply_select(symbolic, -1))  # E: revealed type: Tensor[[4]]
    reveal_type(apply_next(symbolic, 0))  # E: revealed type: Tensor[[3]]
    reveal_type(apply_copy(symbolic))  # E: revealed type: Tensor[[N, 3, 4]]
    reveal_type(apply_reverse(symbolic))  # E: revealed type: Tensor[[4, 3, N]]
    reveal_type(apply_select(symbolic, broad))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_select(gradual, 0))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_select(unpacked, 0))  # E: revealed type: Tensor[[2]]
    reveal_type(apply_huge(symbolic))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_lazy(symbolic, False))  # E: revealed type: Tensor[[N, 3, 4]]
    reveal_type(apply_narrowed_comparison(symbolic, 0))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_narrowed_comparison(symbolic, (0,)))  # E: revealed type: Tensor[[N, 3, 4]]
    apply_select(symbolic, 3)  # E: Cannot evaluate type-level shape DSL call: IntTuple index out of bounds
    apply_select(symbolic, -4)  # E: Cannot evaluate type-level shape DSL call: IntTuple index out of bounds
    apply_select(empty, 0)  # E: Cannot evaluate type-level shape DSL call: IntTuple index out of bounds
    apply_divide(gradual, 0)  # E: Cannot evaluate type-level shape DSL call: Flag integer division by zero
    apply_lazy(symbolic, True)  # E: Cannot evaluate type-level shape DSL call: Flag integer division by zero
"#,
);

testcase!(
    test_type_shape_dsl_int_tuple_length_minimum,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Elements, IntTuple, type_shape_dsl_function
from torch import Tensor
from typing import assert_type, reveal_type

@type_shape_dsl_function
def rank_zero(shape: IntTuple) -> IntTuple:
    if len(shape) == 0:
        return dsl.IntTuple((0,))
    return dsl.IntTuple((10,))

@type_shape_dsl_function
def rank_two(shape: IntTuple) -> IntTuple:
    if len(shape) == 2:
        return dsl.IntTuple((2,))
    return dsl.IntTuple((12,))

@type_shape_dsl_function
def rank_three(shape: IntTuple) -> IntTuple:
    if len(shape) == 3:
        return dsl.IntTuple((3,))
    return dsl.IntTuple((13,))

@type_shape_dsl_function
def rank_negative(shape: IntTuple) -> IntTuple:
    if len(shape) == -1:
        return dsl.IntTuple((9,))
    return dsl.IntTuple((19,))

def apply_rank_zero[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[rank_zero(Shape)]: ...
def apply_rank_two[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[rank_two(Shape)]: ...
def apply_rank_three[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[rank_three(Shape)]: ...
def apply_rank_negative[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[rank_negative(Shape)]: ...

def test[B: IntTuple](
    concrete_two: Tensor[[4, 5]],
    gradual: Tensor[IntTuple],
    unpacked: Tensor[IntTuple[2, *Elements[B], 3]],
) -> None:
    assert_type(apply_rank_zero(concrete_two), Tensor[[10]])
    assert_type(apply_rank_two(concrete_two), Tensor[[2]])
    assert_type(apply_rank_three(concrete_two), Tensor[[13]])
    reveal_type(apply_rank_zero(gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    assert_type(apply_rank_negative(gradual), Tensor[[19]])
    assert_type(apply_rank_zero(unpacked), Tensor[[10]])
    reveal_type(apply_rank_two(unpacked))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_rank_three(unpacked))  # E: revealed type: Tensor[tuple[Unknown, ...]]
"#,
);

testcase!(
    test_type_shape_dsl_concat_and_slice,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Elements, Flag, Int, IntTuple, IntVar, type_shape_dsl_function
from shape_extensions.dsl import concat as imported_concat
from torch import Tensor
from typing import reveal_type

concat_alias = imported_concat

@type_shape_dsl_function
def qualified(left: IntTuple, right: IntTuple) -> IntTuple:
    return dsl.concat(left, right)

@type_shape_dsl_function
def imported(left: IntTuple, right: IntTuple) -> IntTuple:
    return imported_concat(left, right)

@type_shape_dsl_function
def aliased(left: IntTuple, right: IntTuple) -> IntTuple:
    left_alias = left
    joined = concat_alias(left_alias, right)
    return joined

@type_shape_dsl_function
def shape_identity(shape: IntTuple) -> IntTuple:
    return shape

@type_shape_dsl_function
def helper_local(shape: IntTuple) -> IntTuple:
    prefix = shape[:1]
    return shape_identity(prefix)

@type_shape_dsl_function
def empty_prefix(shape: IntTuple) -> IntTuple:
    return shape[:0]

@type_shape_dsl_function
def first_two(shape: IntTuple) -> IntTuple:
    return shape[:2]

@type_shape_dsl_function
def first_three(shape: IntTuple) -> IntTuple:
    return shape[:3]

@type_shape_dsl_function
def clamped(shape: IntTuple) -> IntTuple:
    return shape[:99]

@type_shape_dsl_function
def without_last(shape: IntTuple) -> IntTuple:
    return shape[:-1]

@type_shape_dsl_function
def without_three(shape: IntTuple) -> IntTuple:
    return shape[:-3]

@type_shape_dsl_function
def keep_last(shape: IntTuple) -> IntTuple:
    prefix = shape[:-1]
    return dsl.concat(prefix, dsl.IntTuple((1,)))

@type_shape_dsl_function
def nested(shape: IntTuple) -> IntTuple:
    return dsl.concat(shape[:1], dsl.concat(dsl.IntTuple((7,)), shape[:-1]))

@type_shape_dsl_function
def concat_then_slice(shape: IntTuple) -> IntTuple:
    return dsl.concat(dsl.IntTuple((7,)), shape)[:2]

@type_shape_dsl_function
def minimum_stop(shape: IntTuple) -> IntTuple:
    return shape[:-9223372036854775808]

@type_shape_dsl_function
def full_slice(shape: IntTuple) -> IntTuple:
    return shape[:]

@type_shape_dsl_function
def bounded(shape: IntTuple, start: int, stop: int) -> IntTuple:
    start_alias = start
    computed_stop = stop - 1
    return shape[start_alias:computed_stop]

@type_shape_dsl_function
def suffix(shape: IntTuple, start: int) -> IntTuple:
    return shape[start:]

@type_shape_dsl_function
def helper_slice(shape: IntTuple, start: int, stop: int) -> IntTuple:
    return shape[start:stop]

@type_shape_dsl_function
def call_helper_slice(shape: IntTuple, start: int, stop: int) -> IntTuple:
    return helper_slice(shape, start, stop)

@type_shape_dsl_function
def extreme_stop(shape: IntTuple) -> IntTuple:
    return shape[:999999999999999999999999]

@type_shape_dsl_function
def exact_extreme_bounds(shape: IntTuple) -> IntTuple:
    return shape[-9223372036854775808:9223372036854775807]

@type_shape_dsl_function
def invalid_bound(shape: IntTuple, divisor: int) -> IntTuple:
    stop = 1 // divisor
    return shape[:stop]

@type_shape_dsl_function
def invalid_bound_after_unknown(
    shape: IntTuple, unknown_stop: int, divisor: int,
) -> IntTuple:
    return shape[:unknown_stop][:1 // divisor]

@type_shape_dsl_function
def unused_shape_expression(shape: IntTuple, dimension: Int) -> Int:
    prefix = shape[:1]
    joined = dsl.concat(prefix, dsl.IntTuple((7,)))
    return dimension

@type_shape_dsl_function
def branch_join(shape: IntTuple, keep: bool) -> IntTuple:
    if keep:
        result = shape
    else:
        result = shape[:1]
    return result

@type_shape_dsl_function
def mixed_branch_join(shape: IntTuple, keep: bool) -> IntTuple:
    if keep:
        result = shape[:1]
    else:
        result = dsl.IntTuple((1,))
    return result

@type_shape_dsl_function
def distinct_branch_join(left: IntTuple, right: IntTuple, keep: bool) -> IntTuple:
    if keep:
        result = left[:1]
    else:
        result = right[:1]
    return result

@type_shape_dsl_function
def invalid_before_unknown(left: IntTuple, right: IntTuple) -> IntTuple:
    return dsl.concat(dsl.IntTuple((left[99],)), right[:1])

def apply_qualified[L: IntTuple, R: IntTuple](left: Tensor[L], right: Tensor[R]) -> Tensor[qualified(L, R)]: ...
def apply_imported[L: IntTuple, R: IntTuple](left: Tensor[L], right: Tensor[R]) -> Tensor[imported(L, R)]: ...
def apply_aliased[L: IntTuple, R: IntTuple](left: Tensor[L], right: Tensor[R]) -> Tensor[aliased(L, R)]: ...
def apply_helper_local[S: IntTuple](x: Tensor[S]) -> Tensor[helper_local(S)]: ...
def apply_empty[S: IntTuple](x: Tensor[S]) -> Tensor[empty_prefix(S)]: ...
def apply_first_two[S: IntTuple](x: Tensor[S]) -> Tensor[first_two(S)]: ...
def apply_first_three[S: IntTuple](x: Tensor[S]) -> Tensor[first_three(S)]: ...
def apply_clamped[S: IntTuple](x: Tensor[S]) -> Tensor[clamped(S)]: ...
def apply_without_last[S: IntTuple](x: Tensor[S]) -> Tensor[without_last(S)]: ...
def apply_without_three[S: IntTuple](x: Tensor[S]) -> Tensor[without_three(S)]: ...
def apply_keep_last[S: IntTuple](x: Tensor[S]) -> Tensor[keep_last(S)]: ...
def apply_nested[S: IntTuple](x: Tensor[S]) -> Tensor[nested(S)]: ...
def apply_concat_then_slice[S: IntTuple](x: Tensor[S]) -> Tensor[concat_then_slice(S)]: ...
def apply_minimum_stop[S: IntTuple](x: Tensor[S]) -> Tensor[minimum_stop(S)]: ...
def apply_full_slice[S: IntTuple](x: Tensor[S]) -> Tensor[full_slice(S)]: ...
def apply_bounded[S: IntTuple, Start: Flag[int], Stop: Flag[int]](
    x: Tensor[S], start: Start, stop: Stop,
) -> Tensor[bounded(S, Start, Stop)]: ...
def apply_suffix[S: IntTuple, Start: Flag[int]](
    x: Tensor[S], start: Start,
) -> Tensor[suffix(S, Start)]: ...
def apply_helper_slice[S: IntTuple, Start: Flag[int], Stop: Flag[int]](
    x: Tensor[S], start: Start, stop: Stop,
) -> Tensor[call_helper_slice(S, Start, Stop)]: ...
def apply_extreme_stop[S: IntTuple](x: Tensor[S]) -> Tensor[extreme_stop(S)]: ...
def apply_exact_extreme_bounds[S: IntTuple](
    x: Tensor[S],
) -> Tensor[exact_extreme_bounds(S)]: ...
def apply_invalid_bound[S: IntTuple, Divisor: Flag[int]](
    x: Tensor[S], divisor: Divisor,
) -> Tensor[invalid_bound(S, Divisor)]: ...
def apply_invalid_bound_after_unknown[
    S: IntTuple, Stop: Flag[int], Divisor: Flag[int]
](x: Tensor[S], unknown_stop: Stop, divisor: Divisor) -> Tensor[
    invalid_bound_after_unknown(S, Stop, Divisor)
]: ...
def apply_unused_shape[S: IntTuple, N: IntVar](x: Tensor[S], dimension: Int[N]) -> Tensor[[unused_shape_expression(S, N)]]: ...
def apply_branch[S: IntTuple, Keep: Flag[bool]](x: Tensor[S], keep: Keep) -> Tensor[branch_join(S, Keep)]: ...
def apply_mixed_branch[S: IntTuple, Keep: Flag[bool]](x: Tensor[S], keep: Keep) -> Tensor[mixed_branch_join(S, Keep)]: ...
def apply_distinct_branch[L: IntTuple, R: IntTuple, Keep: Flag[bool]](
    left: Tensor[L], right: Tensor[R], keep: Keep,
) -> Tensor[distinct_branch_join(L, R, Keep)]: ...
def apply_invalid_before_unknown[L: IntTuple, R: IntTuple](left: Tensor[L], right: Tensor[R]) -> Tensor[invalid_before_unknown(L, R)]: ...

def test[S: IntTuple, T: IntTuple, N: IntVar](
    left: Tensor[[2, 3]],
    right: Tensor[[5]],
    unpacked: Tensor[[10, 20, *Elements[S], 30, 40]],
    another: Tensor[[50, *Elements[T], 60]],
    gradual: Tensor[IntTuple],
    dimension: Int[N],
    flag_value: int,
) -> None:
    reveal_type(apply_qualified(left, right))  # E: revealed type: Tensor[[2, 3, 5]]
    reveal_type(apply_imported(left, right))  # E: revealed type: Tensor[[2, 3, 5]]
    reveal_type(apply_aliased(left, right))  # E: revealed type: Tensor[[2, 3, 5]]
    reveal_type(apply_helper_local(left))  # E: revealed type: Tensor[[2]]
    reveal_type(apply_empty(left))  # E: revealed type: Tensor[[]]
    reveal_type(apply_empty(unpacked))  # E: revealed type: Tensor[[]]
    reveal_type(apply_empty(gradual))  # E: revealed type: Tensor[[]]
    reveal_type(apply_first_two(right))  # E: revealed type: Tensor[[5]]
    reveal_type(apply_clamped(left))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_first_two(unpacked))  # E: revealed type: Tensor[[10, 20]]
    reveal_type(apply_first_three(unpacked))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_without_last(unpacked))  # E: revealed type: Tensor[[10, 20, *Elements[S], 30]]
    reveal_type(apply_without_three(unpacked))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_keep_last(unpacked))  # E: revealed type: Tensor[[10, 20, *Elements[S], 30, 1]]
    reveal_type(apply_nested(left))  # E: revealed type: Tensor[[2, 7, 2]]
    reveal_type(apply_concat_then_slice(left))  # E: revealed type: Tensor[[7, 2]]
    reveal_type(apply_minimum_stop(left))  # E: revealed type: Tensor[[]]
    reveal_type(apply_minimum_stop(unpacked))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_full_slice(left))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_full_slice(unpacked))  # E: revealed type: Tensor[[10, 20, *Elements[S], 30, 40]]
    reveal_type(apply_full_slice(gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_bounded(left, 0, 2))  # E: revealed type: Tensor[[2]]
    reveal_type(apply_bounded(left, -2, 2))  # E: revealed type: Tensor[[2]]
    reveal_type(apply_bounded(left, -99, 99))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_bounded(left, 2, 1))  # E: revealed type: Tensor[[]]
    reveal_type(apply_suffix(left, 1))  # E: revealed type: Tensor[[3]]
    reveal_type(apply_suffix(left, 99))  # E: revealed type: Tensor[[]]
    reveal_type(apply_suffix(left, -1))  # E: revealed type: Tensor[[3]]
    reveal_type(apply_suffix(unpacked, 1))  # E: revealed type: Tensor[[20, *Elements[S], 30, 40]]
    reveal_type(apply_suffix(unpacked, -1))  # E: revealed type: Tensor[[40]]
    reveal_type(apply_helper_slice(left, 1, 2))  # E: revealed type: Tensor[[3]]
    reveal_type(apply_helper_slice(unpacked, 1, -1))  # E: revealed type: Tensor[[20, *Elements[S], 30]]
    reveal_type(apply_helper_slice(unpacked, 1, 2))  # E: revealed type: Tensor[[20]]
    reveal_type(apply_helper_slice(unpacked, -2, -1))  # E: revealed type: Tensor[[30]]
    reveal_type(apply_helper_slice(unpacked, -2, 1))  # E: revealed type: Tensor[[]]
    reveal_type(apply_helper_slice(unpacked, 2, -2))  # E: revealed type: Tensor[S]
    reveal_type(apply_helper_slice(unpacked, 99, 100))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_helper_slice(gradual, 1, 1))  # E: revealed type: Tensor[[]]
    reveal_type(apply_helper_slice(gradual, -1, 0))  # E: revealed type: Tensor[[]]
    reveal_type(apply_bounded(left, 0, flag_value))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_extreme_stop(left))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_exact_extreme_bounds(left))  # E: revealed type: Tensor[[2, 3]]
    apply_invalid_bound(left, 0)  # E: division by zero
    apply_invalid_bound(gradual, 0)  # E: division by zero
    apply_invalid_bound_after_unknown(left, flag_value, 0)  # E: division by zero
    reveal_type(apply_unused_shape(left, dimension))  # E: revealed type: Tensor[[N]]
    reveal_type(apply_branch(left, True))  # E: revealed type: Tensor[[2, 3]]
    reveal_type(apply_branch(left, False))  # E: revealed type: Tensor[[2]]
    reveal_type(apply_mixed_branch(left, True))  # E: revealed type: Tensor[[2]]
    reveal_type(apply_mixed_branch(left, False))  # E: revealed type: Tensor[[1]]
    reveal_type(apply_distinct_branch(left, right, True))  # E: revealed type: Tensor[[2]]
    reveal_type(apply_distinct_branch(left, right, False))  # E: revealed type: Tensor[[5]]
    reveal_type(apply_qualified(left, unpacked))  # E: revealed type: Tensor[[2, 3, 10, 20, *Elements[S], 30, 40]]
    reveal_type(apply_qualified(unpacked, right))  # E: revealed type: Tensor[[10, 20, *Elements[S], 30, 40, 5]]
    reveal_type(apply_first_two(gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_without_last(gradual))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_qualified(unpacked, another))  # E: revealed type: Tensor[tuple[Unknown, ...]]
    reveal_type(apply_qualified(gradual, right))  # E: revealed type: Tensor[[*tuple[int, ...], 5]]
    reveal_type(apply_qualified(left, gradual))  # E: revealed type: Tensor[[2, 3, *tuple[int, ...]]]
    apply_invalid_before_unknown(left, gradual)  # E: IntTuple index out of bounds
"#,
);

testcase!(
    test_type_shape_dsl_symbolic_suffix_index,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Elements, Int, IntTuple, IntVar, type_shape_dsl_function
from torch import Tensor
from typing import assert_type

@type_shape_dsl_function
def last(shape: IntTuple) -> Int:
    result = shape[-1]
    return result

def apply[Shape: IntTuple](x: Tensor[Shape]) -> Tensor[[last(Shape)]]: ...

def test[Batch: IntTuple, N: IntVar](x: Tensor[[*Elements[Batch], N]]) -> None:
    assert_type(apply(x), Tensor[[N]])
"#,
);

testcase!(
    test_type_shape_dsl_invalid_concat_and_slice,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def missing(shape: IntTuple) -> IntTuple:
    return dsl.concat(shape)  # E: `dsl.concat` requires exactly two positional arguments  # E: Missing positional argument `right`

@type_shape_dsl_function
def extra(shape: IntTuple) -> IntTuple:
    return dsl.concat(shape, shape, shape)  # E: `dsl.concat` requires exactly two positional arguments  # E: Expected 2 positional arguments

@type_shape_dsl_function
def keyword(shape: IntTuple) -> IntTuple:
    return dsl.concat(left=shape, right=shape)  # E: `dsl.concat` requires exactly two positional arguments  # E: Expected argument `left` to be positional  # E: Expected argument `right` to be positional

@type_shape_dsl_function
def wrong_operand(left: Int, right: IntTuple) -> IntTuple:
    return dsl.concat(left, right)  # E: shape expression operands must be annotated as `IntTuple`  # E: is not assignable to parameter

@type_shape_dsl_function
def wrong_result(left: IntTuple, right: IntTuple) -> Int:
    return dsl.concat(left, right)  # E: returned shape expression requires an `IntTuple` result  # E: Returned type

@type_shape_dsl_function
def incompatible_local_return(shape: IntTuple, dimension: Int, choose_shape: bool) -> IntTuple:
    if choose_shape:
        result = shape[:1]
    else:
        result = dimension
    return result  # E: local return requires contributing parameters to use the `IntTuple` domain  # E: Returned type

@type_shape_dsl_function
def shape_equality(shape: IntTuple) -> IntTuple:
    if shape[:1] == shape:  # E: Flag integer expression is not supported
        return shape
    return shape

@type_shape_dsl_function
def local_shape_is_not_int(shape: IntTuple) -> IntTuple:
    local = dsl.concat(dsl.IntTuple((1,)), dsl.IntTuple((2,)))
    if dsl.is_int_value(local):  # E: `is_int_value` requires a `Flag[int | tuple[int, ...] | None]` value
        return local
    return shape

@type_shape_dsl_function
def indexed_return(shape: IntTuple) -> Int:
    return shape[0]  # E: return value must be a bare parameter name

@type_shape_dsl_function
def step(shape: IntTuple) -> IntTuple:
    return shape[:2:1]  # E: IntTuple slices do not support steps

@type_shape_dsl_function
def dimension_bound(shape: IntTuple, start: Int) -> IntTuple:
    return shape[start:]  # E: Flag operation requires a compatible Flag parameter

@type_shape_dsl_function
def bool_bound(shape: IntTuple, stop: bool) -> IntTuple:
    return shape[:stop]  # E: Flag operation requires a compatible Flag parameter

def concat(left: IntTuple, right: IntTuple) -> IntTuple: ...

@type_shape_dsl_function
def shadowed(left: IntTuple, right: IntTuple) -> IntTuple:
    return concat(left, right)  # E: DSL helper callee must be a validated
"#,
);

testcase!(
    test_type_shape_dsl_prod,
    shape_dsl_tensor_env(),
    r#"
import shape_extensions.dsl
import shape_extensions.dsl as qualified_dsl
from shape_extensions import Elements, Int, IntTuple, IntVar, type_shape_dsl_function
from shape_extensions.dsl import IntTuple as DslIntTuple
from shape_extensions.dsl import prod as imported_prod
from torch import Tensor
from typing import assert_type, reveal_type

prod_alias = imported_prod

@type_shape_dsl_function
def qualified(shape: IntTuple) -> Int:
    return qualified_dsl.prod(shape)

@type_shape_dsl_function
def module_qualified(shape: IntTuple) -> Int:
    return shape_extensions.dsl.prod(shape)

@type_shape_dsl_function
def imported(shape: IntTuple) -> Int:
    return imported_prod(shape)

@type_shape_dsl_function
def aliased(shape: IntTuple) -> Int:
    return prod_alias(shape)

@type_shape_dsl_function
def local(shape: IntTuple) -> Int:
    result = imported_prod(shape)
    return result

@type_shape_dsl_function
def prefix(shape: IntTuple) -> Int:
    shape_alias = shape
    return imported_prod(shape_alias[:2])

@type_shape_dsl_function
def empty(shape: IntTuple) -> Int:
    return imported_prod(DslIntTuple(()))

@type_shape_dsl_function
def wrapped(shape: IntTuple) -> IntTuple:
    return DslIntTuple((prod_alias(shape),))

@type_shape_dsl_function
def zero_prefix(shape: IntTuple) -> Int:
    return imported_prod(qualified_dsl.concat(DslIntTuple((0,)), shape))

@type_shape_dsl_function
def zero_suffix(shape: IntTuple) -> Int:
    return imported_prod(qualified_dsl.concat(shape, DslIntTuple((0,))))

@type_shape_dsl_function
def zero_gradual_dimension(dimension: Int) -> Int:
    return imported_prod(DslIntTuple((0, dimension)))

@type_shape_dsl_function
def identity_padded(shape: IntTuple) -> Int:
    return imported_prod(DslIntTuple((1, shape[0], 1)))

@type_shape_dsl_function
def all_ones(shape: IntTuple) -> Int:
    return imported_prod(DslIntTuple((1, 1, 1)))

@type_shape_dsl_function
def zero_overflow(shape: IntTuple) -> Int:
    return imported_prod(DslIntTuple((0, 9223372036854775807, 2)))

def apply_qualified[S: IntTuple](x: Tensor[S]) -> Tensor[[qualified(S)]]: ...
def apply_module[S: IntTuple](x: Tensor[S]) -> Tensor[[module_qualified(S)]]: ...
def apply_imported[S: IntTuple](x: Tensor[S]) -> Tensor[[imported(S)]]: ...
def apply_aliased[S: IntTuple](x: Tensor[S]) -> Tensor[[aliased(S)]]: ...
def apply_local[S: IntTuple](x: Tensor[S]) -> Tensor[[local(S)]]: ...
def apply_prefix[S: IntTuple](x: Tensor[S]) -> Tensor[[prefix(S)]]: ...
def apply_empty[S: IntTuple](x: Tensor[S]) -> Tensor[[empty(S)]]: ...
def apply_wrapped[S: IntTuple](x: Tensor[S]) -> Tensor[wrapped(S)]: ...
def apply_zero_prefix[S: IntTuple](x: Tensor[S]) -> Tensor[[zero_prefix(S)]]: ...
def apply_zero_suffix[S: IntTuple](x: Tensor[S]) -> Tensor[[zero_suffix(S)]]: ...
def apply_identity_padded[S: IntTuple](x: Tensor[S]) -> Tensor[[identity_padded(S)]]: ...
def gradual_dimension_zero() -> Tensor[[zero_gradual_dimension(Int[int])]]: ...
def all_ones_result() -> Tensor[[all_ones(IntTuple[7])]]: ...
def zero_overflow_result() -> Tensor[[zero_overflow(IntTuple[7])]]: ...
def literal_overflow() -> Tensor[[qualified(IntTuple[9223372036854775807, 2])]]: ...
def symbolic_overflow[N: IntVar](n: Int[N]) -> Tensor[[qualified(IntTuple[9223372036854775807, N, 2])]]: ...

def test[S: IntTuple, N: IntVar, M: IntVar](
    concrete: Tensor[[2, 3]],
    symbolic: Tensor[[2, N, 3]],
    triple: Tensor[[2, 3, 5]],
    gradual: Tensor[IntTuple],
    unpacked: Tensor[[2, *Elements[S], 3]],
    add: Tensor[[(N + 1)]],
    subtract: Tensor[[(N - 1)]],
    floor_divide: Tensor[[(N // 2)]],
    power: Tensor[[(N ** 2)]],
    multi_factor_computed_awaits_checked_canonicalization: Tensor[[(N + 1), M]],
    computed_and_literal_await_checked_canonicalization: Tensor[[(N + 1), 2]],
    subtract_fallback: Tensor[[(N - 1), M]],
    floor_divide_fallback: Tensor[[(N // 2), M]],
    power_fallback: Tensor[[(N ** 2), M]],
    n: Int[N],
) -> None:
    assert_type(apply_qualified(concrete), Tensor[[6]])
    assert_type(apply_module(concrete), Tensor[[6]])
    reveal_type(apply_imported(symbolic))  # E: revealed type: Tensor[[(6 * N)]]
    assert_type(apply_aliased(concrete), Tensor[[6]])
    assert_type(apply_local(concrete), Tensor[[6]])
    assert_type(apply_prefix(triple), Tensor[[6]])
    assert_type(apply_empty(concrete), Tensor[[1]])
    assert_type(apply_wrapped(concrete), Tensor[[6]])
    reveal_type(apply_qualified(gradual))  # E: revealed type: Tensor[[int]]
    reveal_type(apply_zero_prefix(unpacked))  # E: revealed type: Tensor[[0]]
    reveal_type(apply_zero_suffix(unpacked))  # E: revealed type: Tensor[[0]]
    reveal_type(gradual_dimension_zero())  # E: revealed type: Tensor[[0]]
    reveal_type(zero_overflow_result())  # E: revealed type: Tensor[[0]]
    reveal_type(apply_qualified(add))  # E: revealed type: Tensor[[(1 + N)]]
    reveal_type(apply_qualified(subtract))  # E: revealed type: Tensor[[(-1 + N)]]
    reveal_type(apply_qualified(floor_divide))  # E: revealed type: Tensor[[(N // 2)]]
    reveal_type(apply_qualified(power))  # E: revealed type: Tensor[[(N ** 2)]]
    reveal_type(apply_identity_padded(add))  # E: revealed type: Tensor[[(1 + N)]]
    reveal_type(apply_identity_padded(floor_divide))  # E: revealed type: Tensor[[(N // 2)]]
    reveal_type(apply_identity_padded(power))  # E: revealed type: Tensor[[(N ** 2)]]
    reveal_type(all_ones_result())  # E: revealed type: Tensor[[1]]
    reveal_type(apply_qualified(unpacked))  # E: revealed type: Tensor[[int]]
    reveal_type(apply_qualified(multi_factor_computed_awaits_checked_canonicalization))  # E: revealed type: Tensor[[int]]
    reveal_type(apply_qualified(computed_and_literal_await_checked_canonicalization))  # E: revealed type: Tensor[[int]]
    reveal_type(apply_qualified(subtract_fallback))  # E: revealed type: Tensor[[int]]
    reveal_type(apply_qualified(floor_divide_fallback))  # E: revealed type: Tensor[[int]]
    reveal_type(apply_qualified(power_fallback))  # E: revealed type: Tensor[[int]]
    reveal_type(literal_overflow())  # E: revealed type: Tensor[[int]]
    reveal_type(symbolic_overflow(n))  # E: revealed type: Tensor[[int]]
"#,
);

testcase!(
    test_type_shape_dsl_invalid_prod,
    shape_dsl_tensor_env(),
    r#"
from shape_extensions import Int, IntTuple, type_shape_dsl_function
from shape_extensions.dsl import prod as official_prod

@type_shape_dsl_function
def missing(shape: IntTuple) -> Int:
    return official_prod()  # E: `dsl.prod` requires exactly one positional IntTuple argument  # E: No matching overload found

@type_shape_dsl_function
def extra(shape: IntTuple) -> Int:
    return official_prod(shape, shape)  # E: `dsl.prod` requires exactly one positional IntTuple argument  # E: No matching overload found

@type_shape_dsl_function
def keyword(shape: IntTuple) -> Int:
    return official_prod(x=shape)  # E: `dsl.prod` requires exactly one positional IntTuple argument  # E: Missing argument `xs`  # E: Unexpected keyword argument `x`

@type_shape_dsl_function
def starred(shape: IntTuple) -> Int:
    return official_prod(*(shape,))  # E: `dsl.prod` requires exactly one positional IntTuple argument

@type_shape_dsl_function
def wrong_domain(dimension: Int) -> Int:
    return official_prod(dimension)  # E: shape expression operands must be annotated as `IntTuple`  # E: No matching overload found

@type_shape_dsl_function
def wrong_result(shape: IntTuple) -> IntTuple:
    return official_prod(shape)  # E: returned `IntTuple` product requires an `Int` result  # E: Returned type

def ordinary_prod(shape: IntTuple) -> Int: ...

@type_shape_dsl_function
def ordinary_shadow(shape: IntTuple) -> Int:
    return ordinary_prod(shape)  # E: DSL helper callee must be a validated

@type_shape_dsl_function
def parameter_shadow(official_prod: IntTuple, shape: IntTuple) -> Int:
    return official_prod(shape)  # E: DSL helper callee must be a validated  # E: Expected a callable
"#,
);
