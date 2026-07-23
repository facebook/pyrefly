/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::type_var::Restriction;
use ruff_python_ast::name::Name;

use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTParams;
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
    match &**solutions.get(&KeyExport(Name::new("x"))) {
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
    match &**solutions.get(&KeyExport(Name::new("N"))) {
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
from typing import reveal_type
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
    assert_type(first, str | Int[int])
    reveal_type(rest)  # E: revealed type: list[str | Int[int]]
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
    reveal_type(shapeless[0])  # E: revealed type: Array
    reveal_type(shapeless[None])  # E: revealed type: Array
    reveal_type(shapeless[None, ...])  # E: revealed type: Array

def accepts_precise(x: Array[[2, 3], int]) -> None:
    pass

def shapeless_is_gradual(shapeless: Array) -> None:
    accepts_precise(shapeless)

def values() -> None:
    value = Array()
    reveal_type(value)  # E: revealed type: Array
    reveal_type(value[0])  # E: revealed type: Array

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
    reveal_type(x[unbounded])  # E: revealed type: Array
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
    reveal_type(x[index_with_device])  # E: revealed type: Array
    reveal_type(x[bool_index])  # E: revealed type: Array
    reveal_type(x[float_index])  # E: revealed type: Array
    reveal_type(x[str_index])  # E: revealed type: Array
    reveal_type(x[any_dtype_index])  # E: revealed type: Array
    reveal_type(x[unsupported_index])  # E: revealed type: Array
    reveal_type(x[any_index])  # E: revealed type: Array
    reveal_type(x[mixed])  # E: revealed type: Array
    reveal_type(x[strings])  # E: revealed type: Array
    reveal_type(x[[*strings]])  # E: revealed type: Array
    reveal_type(x[anys])  # E: revealed type: Array
    reveal_type(x[bools])  # E: revealed type: Array
    reveal_type(x[raw])  # E: revealed type: Array
    reveal_type(x[nested])  # E: revealed type: Array
    reveal_type(x[True])  # E: revealed type: Array
    reveal_type(x[bool_literal])  # E: revealed type: Array
    reveal_type(x[unpacked])  # E: revealed type: Array
    reveal_type(x[(unpacked,)])  # E: revealed type: Array
    reveal_type(x[unconstrained])  # E: revealed type: Array
    reveal_type(x[stored_slice])  # E: revealed type: Array
    reveal_type(x[stored_ellipsis])  # E: revealed type: Array
    reveal_type(x[0, stored_slice])  # E: revealed type: Array
    reveal_type(x[0, stored_ellipsis])  # E: revealed type: Array
    reveal_type(x[slice_key])  # E: revealed type: Array
    reveal_type(x[ellipsis_key])  # E: revealed type: Array
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
    reveal_type(x[0])  # E: revealed type: Array
    reveal_type(x[0].shape)  # E: revealed type: IntTuple
    reveal_type(x[:])  # E: revealed type: Array
    reveal_type(x[:].shape)  # E: revealed type: IntTuple
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_broadcast_keeps_shape_coherent,
    shaped_array_env(),
    r#"
from typing import Any, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]:
    shape: Shape
    def dtype(self) -> DType: ...

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
from shape_extensions import Int, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]:
    shape: Shape

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
from typing import reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def make_array[S](shape: S) -> Array[S, float]: ...

def f() -> None:
    reveal_type(make_array((2, 3)))  # E: revealed type: Array
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
    reveal_type(x)  # E: revealed type: Tensor
    reveal_type(x[0])  # E: revealed type: Tensor
    reveal_type(y)  # E: revealed type: Tensor
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
    assert_type(i, Int[int])
    assert_type(s, int)
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
    reveal_type(inferred_union)  # E: revealed type: int
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
    reveal_type(y[0])  # E: revealed type: Tensor
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
    reveal_type(x)  # E: revealed type: Tensor
    reveal_type(y)  # E: revealed type: Tensor
    reveal_type(z)  # E: revealed type: Tensor
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
    bug =
        "dotted-name arguments to @uses_shape_dsl currently silent-noop; should emit a diagnostic",
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
