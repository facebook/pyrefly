/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    let mut env = TestEnv::new();
    env.add_with_path(
        "shape_extensions",
        "shape_extensions.pyi",
        r#"
from typing import Any, Callable, _SpecialForm

shaped_array: Any
SymVar: _SpecialForm
class SizeTuple:
    def __class_getitem__(cls, params: Any) -> Any: ...
class Elements[T]: ...
class Dim[T]: ...
class D: ...
def assert_shape[T](x: T, shape: tuple[Any, ...]) -> T: ...
def defines_assert_shape[F: Callable[..., Any]](fn: F) -> F: ...
"#,
    );
    env
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
from shape_extensions import Elements, SizeTuple, shaped_array

@shaped_array(shape="Shape")
class Tensor[Shape: SizeTuple]: ...
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
    let mut env = TestEnv::new();
    env.add_with_path(
        "shape_extensions",
        "shape_extensions/__init__.pyi",
        r#"
from typing import Any, Callable

shaped_array: Any
class SizeTuple:
    def __class_getitem__(cls, params: Any) -> Any: ...
class Elements[T]: ...
def uses_shape_dsl(ir_fn: Callable[..., Any], *, capture_init: list[str] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
"#,
    );
    env.add_with_path(
        "shape_extensions.dsl",
        "shape_extensions/dsl.pyi",
        r#"
from typing import Any, Callable

def shape_dsl_function(fn: Callable[..., Any]) -> Callable[..., Any]: ...

class ShapedArray:
    shape: list[int]
    def __init__(self, *, shape: list[int]) -> None: ...
"#,
    );
    env.add_with_path(
        "numpy",
        "numpy/__init__.pyi",
        r#"
from shape_extensions import uses_shape_dsl
from shape_extensions import shaped_array
from shape_extensions import SizeTuple
from shape_extensions.dsl import ShapedArray, shape_dsl_function
from typing import Any

type AnyShape = tuple[Any, ...]

@shape_dsl_function
def add_leading_axis_ir(x: ShapedArray) -> ShapedArray:
    return ShapedArray(shape=[1] + x.shape)

@shaped_array(shape="Shape")
class ndarray[Shape: SizeTuple, DType]:
    shape: Shape
    def copy(self) -> ndarray[Shape, DType]: ...
    def item(self) -> DType: ...

@uses_shape_dsl(add_leading_axis_ir)
def add_leading_axis[Shape: SizeTuple, DType](x: ndarray[Shape, DType]) -> ndarray[Shape, DType]: ...

@shaped_array(shape="Shape")
class tcarray[Shape: SizeTuple = AnyShape, DType = int]:
    shape: Shape
    def dtype(self) -> DType: ...
    @uses_shape_dsl(add_leading_axis_ir)
    def add_leading_axis(self) -> tcarray[Shape, DType]: ...

@uses_shape_dsl(add_leading_axis_ir)
def tc_add_leading_axis[Shape: SizeTuple, DType](x: tcarray[Shape, DType]) -> tcarray[Shape, DType]: ...

def tc_identity[Shape: SizeTuple, DType](x: tcarray[Shape, DType]) -> tcarray[Shape, DType]: ...
"#,
    );
    env
}

fn shape_dsl_base_env() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "shape_extensions",
        "shape_extensions/__init__.pyi",
        r#"
from typing import Any, Callable

shaped_array: Any
class SizeTuple:
    def __class_getitem__(cls, params: Any) -> Any: ...
class Elements[T]: ...
def uses_shape_dsl(ir_fn: Callable[..., Any], *, capture_init: list[str] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
"#,
    );
    env.add_with_path(
        "shape_extensions.dsl",
        "shape_extensions/dsl.pyi",
        r#"
from typing import Any, Callable

def shape_dsl_function(fn: Callable[..., Any]) -> Callable[..., Any]: ...
def prod(x: Any) -> Any: ...
def sum(x: Any) -> Any: ...
def parse_einsum_equation(x: Any) -> Any: ...

class ShapedArray:
    shape: list[int]
    def __init__(self, *, shape: list[int]) -> None: ...
"#,
    );
    env
}

fn shape_dsl_tensor_env() -> TestEnv {
    let mut env = shape_dsl_base_env();
    env.add_with_path(
        "torch",
        "torch.pyi",
        r#"
from shape_extensions import Elements, SizeTuple, shaped_array

@shaped_array(shape="Shape")
class Tensor[Shape: SizeTuple]:
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
from shape_extensions import SizeTuple, shaped_array
from shape_extensions import shaped_array as shaped_array_alias

@shaped_array(shape="Shape")
class ImportedArray[Shape: SizeTuple]: ...

@shaped_array_alias(shape="Shape")
class ImportAliasArray[Shape: SizeTuple]: ...

@se.shaped_array(shape="Shape")
class ModuleAliasArray[DType, Shape: SizeTuple]: ...

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
fn test_legacy_symvar_binding_has_symvar_kind() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import SymVar

N = SymVar("N")
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let solutions = state.transaction().get_solutions(&main).unwrap();
    match &**solutions.get(&KeyExport(Name::new("N"))) {
        Type::TypeVar(tv) => assert_eq!(tv.kind(), QuantifiedKind::SymVar),
        ty => panic!("expected `N` to solve to a raw SymVar, got `{ty}`"),
    }
}

#[test]
fn test_legacy_symvar_generic_class_tparam_has_symvar_kind() {
    let mut env = shaped_array_env();
    env.add(
        "main",
        r#"
from shape_extensions import SymVar
from typing import Generic

N = SymVar("N")

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
    assert_eq!(param.kind(), QuantifiedKind::SymVar);
}

#[test]
fn test_non_shape_symvar_is_not_a_kind_marker() {
    let mut env = shaped_array_env();
    env.add(
        "other",
        r#"
class SymVar: ...
"#,
    );
    env.add(
        "main",
        r#"
from other import SymVar
from typing import Generic

class Box[N: SymVar](Generic[N]): ...
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
        Restriction::Bound(Type::ClassType(cls)) if cls.has_qname("other", "SymVar")
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

@shaped_array()  # E: `@shaped_array` requires a `shape` keyword argument
class MissingShape[Shape]: ...

@shaped_array("Shape")  # E: `@shaped_array` expects `shape` as a keyword argument
class PositionalShape[Shape]: ...

@shaped_array(dtype="Shape")  # E: Unexpected keyword argument `dtype` for `@shaped_array`; expected `shape`
class WrongShapeKeyword[Shape]: ...

@shaped_array(shape="Shape", **kwargs)  # E: Unpacking is not supported in `@shaped_array`
class KwargsShape[Shape]: ...

@shaped_array(shape="Shape", shape="Shape")  # E: Parse error: Duplicate keyword argument "shape"
class DuplicateShapeKeyword[Shape]: ...

@shaped_array(shape=123)  # E: `@shaped_array` `shape` argument must be a string literal
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

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a `TypeVar` or `SymVar`, got `TypeVarTuple`
class TypeVarTupleShape[*Shape]: ...

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a `TypeVar` or `SymVar`, got `ParamSpec`
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
from shape_extensions import SymVar, shaped_array

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
    test_shaped_array_sizetuple_bound,
    shaped_array_env(),
    r#"
from typing import Any, Literal, reveal_type
from shape_extensions import Dim, Elements, SizeTuple, SymVar, shaped_array

type _Shape = SizeTuple
type _AnyShape = tuple[Any, ...]

@shaped_array(shape="Shape")
class Array[Shape: _Shape = _AnyShape, DType = Any]:
    shape: Shape

def f[N: SymVar](
    compact: Array[[2, 3], int],
    pep484: Array[tuple[Literal[2], Literal[3]], int],
    size_tuple: Array[SizeTuple[2, 3], int],
    mixed_size_tuple: Array[SizeTuple[2, 3, Dim[N]], int],
    bare_dim: Dim[N],
    bare_list: Array[[N], int],
    bare_size_tuple: Array[SizeTuple[N], int],
    any_dim: Array[[Any], int],
    carrier: SizeTuple[2, 3],
    mixed_carrier: SizeTuple[2, 3, Dim[N]],
    unbounded: SizeTuple,
) -> None:
    reveal_type(compact)  # E: revealed type: Array[[2, 3], int]
    reveal_type(pep484)  # E: revealed type: Array[[2, 3], int]
    reveal_type(size_tuple)  # E: revealed type: Array[[2, 3], int]
    reveal_type(mixed_size_tuple)  # E: revealed type: Array[[2, 3, N], int]
    reveal_type(bare_dim)  # E: revealed type: Dim[N]
    reveal_type(bare_list)  # E: revealed type: Array[[N], int]
    reveal_type(bare_size_tuple)  # E: revealed type: Array[[N], int]
    reveal_type(any_dim)  # E: revealed type: Array[[Any], int]
    p: Array[tuple[Literal[2], Literal[3]], int] = compact
    c: Array[[2, 3], int] = pep484
    st: Array[SizeTuple[2, 3], int] = compact
    mst: Array[tuple[Literal[2], Literal[3], Dim[N]], int] = mixed_size_tuple
    t: tuple[Literal[2], Literal[3]] = carrier
    mt: tuple[Literal[2], Literal[3], Dim[N]] = mixed_carrier
    u: tuple[int, ...] = unbounded

def append_dim[S: SizeTuple, OUT: SymVar](
    explicit: Array[SizeTuple[*Elements[S], OUT], int],
    compact: Array[[*Elements[S], OUT], int],
) -> Array[[*Elements[S], OUT], int]:
    reveal_type(explicit)  # E: revealed type: Array[[*S, OUT], int]
    reveal_type(compact)  # E: revealed type: Array[[*S, OUT], int]
    return explicit

def prepend_and_append[S: SizeTuple, OUT: SymVar](
    source: Array[S, int],
    result: Array[[1, *Elements[S], OUT], int],
) -> Array[[1, *Elements[S], OUT], int]:
    return result

def concrete_unpack[M: SymVar, N: SymVar](
    source: Array[[4, M], int],
    result: Array[[1, 4, M, N], int],
) -> None:
    reveal_type(prepend_and_append(source, result))  # E: revealed type: Array[[1, 4, M, N], int]

def nested_unpack[S0: SizeTuple, M: SymVar, N: SymVar](
    source: Array[[4, *Elements[S0], M], int],
    result: Array[[1, 4, *Elements[S0], M, N], int],
) -> None:
    reveal_type(prepend_and_append(source, result))  # E: revealed type: Array[[1, 4, *S0, M, N], int]

def gradual_middle(
    result: Array[[1, *Elements[SizeTuple], 3], int],
) -> None:
    reveal_type(result)  # E: revealed type: Array[[1, *tuple[int, ...], 3], int]
"#,
);

testcase!(
    test_ordinary_typevar_shape_dimension_is_rejected,
    shaped_array_env(),
    r#"
from typing import Any, Generic, TypeVar
from shape_extensions import Dim, Elements, SizeTuple, SymVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape: SizeTuple = tuple[Any, ...], DType = Any]: ...

class SymBox[N: SymVar]: ...

def invalid[N, Shape: SizeTuple](
    dim: Dim[N],  # E: `N` must be a `SymVar` to be used as a shape dimension
    list_shape: Array[[N], int],  # E: `N` must be a `SymVar` to be used as a shape dimension
    size_tuple: Array[SizeTuple[N], int],  # E: `N` must be a `SymVar` to be used as a shape dimension
    unpack_prefix: Array[SizeTuple[N, *Elements[Shape]], int],  # E: `N` must be a `SymVar` to be used as a shape dimension
    class_arg: SymBox[N],  # E: `N` must be a `SymVar` to be used as a shape dimension
) -> None:
    pass

type Alias[N] = Dim[N]  # E: `N` must be a `SymVar` to be used as a shape dimension

LegacyN = TypeVar("LegacyN")

class LegacyBox(Generic[LegacyN]):
    dim: Dim[LegacyN]  # E: `LegacyN` must be a `SymVar` to be used as a shape dimension
    shape: Array[[LegacyN], int]  # E: `LegacyN` must be a `SymVar` to be used as a shape dimension
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
    test_shaped_array_compact_list_rejects_unbounded_tuple_unpack,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(bad: Array[[2, *tuple[int, ...]], int]) -> None: ...  # E: Unpacked type in `SizeTuple` must use `Elements[...]`, got `tuple[int, ...]`
"#,
);

testcase!(
    test_shaped_array_compact_list_elements_rejects_non_sizetuple_carrier,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(bad: Array[[2, *Elements[int]], int]) -> None: ...  # E: `Elements[...]` requires a `SizeTuple` carrier, got `int`
"#,
);

testcase!(
    test_shaped_array_compact_list_requires_elements_for_sizetuple_unpack,
    shaped_array_env(),
    r#"
from shape_extensions import SizeTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f[S: SizeTuple](bad: Array[[2, *S], int]) -> None: ...  # E: Unpacked type in `SizeTuple` must use `Elements[...]`, got `S`
"#,
);

testcase!(
    test_shaped_array_compact_list_rejects_multiple_unpacked_carriers,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, SizeTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f[S: SizeTuple, T: SizeTuple](bad: Array[[*Elements[S], *Elements[T]], int]) -> None: ...  # E: `SizeTuple` can have at most one unpacked shape carrier
"#,
);

testcase!(
    test_shaped_array_elements_rejects_multiple_args,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, SizeTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f[S: SizeTuple, T: SizeTuple](bad: Array[[*Elements[S, T]], int]) -> None: ...  # E: Expected 1 type argument for `Elements`, got 2
"#,
);

testcase!(
    test_shaped_array_elements_accepts_legacy_typevar_carrier,
    shaped_array_env(),
    r#"
from typing import TypeVar
from shape_extensions import Elements, SizeTuple, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

S = TypeVar("S", bound=SizeTuple)

def f(x: Array[[*Elements[S], 3], int]) -> None: ...
"#,
);

testcase!(
    test_shaped_array_annotation_parsing,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, SizeTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Array[Shape: SizeTuple, DType]:
    def __init__(self) -> None: ...
    def dtype(self) -> DType: ...

class Cpu: ...
class Gpu: ...

@shaped_array(shape="Shape")
class ArrayWithDevice[Shape: SizeTuple, DType, Device: (Gpu, Cpu)]:
    def dtype(self) -> DType: ...
    def device(self) -> Device: ...

@shaped_array(shape="Shape")
class DTypeFirstArray[DType, Shape: SizeTuple]:
    def dtype(self) -> DType: ...

def f(
    x: Array[[2, 3], int],
    y: Array[[], int],
    z: Array[[2, *Elements[SizeTuple]], int],
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
from shape_extensions import SizeTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Array[Shape: SizeTuple, DType]:
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

def values() -> None:
    value = Array()
    reveal_type(value)  # E: revealed type: Array
    reveal_type(value[0])  # E: revealed type: Array

def index_preserves_dtype(concrete: Array[[2, 3], int]) -> Array[[3], int]:
    return concrete[0]
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
    # Integer index drops the leading dim, and `.shape` (read from the raw
    # carrier) stays coherent with the projected shape.
    reveal_type(x[0])  # E: revealed type: Array[[3, 4], int]
    reveal_type(x[0].shape)  # E: revealed type: tuple[Literal[3], Literal[4]]
    reveal_type(x[0].dtype())  # E: revealed type: int

    # Mixed tuple index (slice + int) and `None`/newaxis stay coherent too.
    reveal_type(x[:, 0])  # E: revealed type: Array[[2, 4], int]
    reveal_type(x[:, 0].shape)  # E: revealed type: tuple[Literal[2], Literal[4]]
    reveal_type(x[None])  # E: revealed type: Array[[1, 2, 3, 4], int]
    reveal_type(x[None].shape)  # E: revealed type: tuple[Literal[1], Literal[2], Literal[3], Literal[4]]

    # The carrier rewrite follows the registered shape parameter, even when it
    # is not the first type argument.
    reveal_type(dtype_first[0])  # E: revealed type: DTypeFirstArray[int, [3, 4]]
    reveal_type(dtype_first[0].shape)  # E: revealed type: tuple[Literal[3], Literal[4]]
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
    reveal_type(x[0].shape)  # E: revealed type: tuple[Unknown, ...]
    reveal_type(x[:])  # E: revealed type: Array
    reveal_type(x[:].shape)  # E: revealed type: tuple[Unknown, ...]
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_broadcast_keeps_shape_coherent,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]:
    shape: Shape
    def dtype(self) -> DType: ...

def f(x: Array[[2, 3], int], y: Array[[1, 3], int]) -> None:
    z = x + y
    # Broadcasting `(2, 3)` with `(1, 3)` yields `(2, 3)`, and the raw carrier is
    # rewritten so `.shape` stays coherent. DType is preserved from the carrier.
    reveal_type(z)  # E: revealed type: Array[[2, 3], int]
    reveal_type(z.shape)  # E: revealed type: tuple[Literal[2], Literal[3]]
    reveal_type(z.dtype())  # E: revealed type: int
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
    use_shape(compact_2_3, shape_2_4)  # E: Argument `tuple[Literal[2], Literal[4]]` is not assignable to parameter `shape` with type `tuple[Literal[2], Literal[3]]`
    out: tuple[Literal[2], Literal[3]] = get_shape(compact_2_3)
    bad: tuple[Literal[2], Literal[4]] = get_shape(compact_2_3)  # E: `tuple[Literal[2], Literal[3]]` is not assignable to `tuple[Literal[2], Literal[4]]`
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
    bad: tuple[Literal[3], *Ts] = get_shape(x)  # E: `tuple[Literal[2], *Ts]` is not assignable to `tuple[Literal[3], *Ts]`
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
    use_shape(x, shape_3)  # E: Argument `tuple[Literal[3], *Ts]` is not assignable to parameter `shape` with type `tuple[Literal[2], *Ts]`
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_shape_attr_preserves_generic_carrier,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import SymVar, shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def carrier[S](x: Array[S, float]) -> None:
    reveal_type(x.shape)  # E: revealed type: S

def concrete[M: SymVar](x: Array[[2, 4, M], float]) -> None:
    reveal_type(x.shape)  # E: revealed type: tuple[Literal[2], Literal[4], Dim[M]]

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
    bug = "invalid closed carrier Array[tuple[str, str], int] is not yet rejected",
    test_shaped_array_invalid_closed_carrier,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def want_2_3(x: Array[[2, 3], int]) -> None: ...
def want_bad(x: Array[tuple[str, str], int]) -> None: ...

# `tuple[str, str]` is not a valid shape carrier. It projects to a shapeless
# array internally; a source-aware diagnostic rejecting this form is deferred.
def f(x_bad: Array[tuple[str, str], int]) -> None:
    want_2_3(x_bad)  # E: Argument `Array[tuple[Unknown, ...], int]` is not assignable to parameter `x` with type `Array[[2, 3], int]`
    want_bad(x_bad)

def g(x_2_3: Array[[2, 3], int]) -> None:
    want_bad(x_2_3)  # E: Argument `Array[[2, 3], int]` is not assignable to parameter `x` with type `Array[tuple[Unknown, ...], int]`
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
from shape_extensions import Dim, SizeTuple, SymVar, shaped_array
from typing import TypeVar, reveal_type

T = TypeVar("T")
DefaultT = TypeVar("DefaultT", default=3)  # E: Expected a type form, got instance of `Literal[3]`

class Box[T]: ...
class DefaultBox[T = 3]: ...  # E: Expected a type form, got instance of `Literal[3]`

@shaped_array(shape="Shape")
class Array[Shape: SizeTuple, DType, Device]: ...

@shaped_array(shape="Shape")
class DTypeFirstArray[DType, Shape: SizeTuple]: ...

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

def dims[N: SymVar](concrete: Dim[3], symbolic: Dim[N + 1]) -> None:
    pass
"#,
);

testcase!(
    test_tensor_shapes_keeps_ordinary_literal_arithmetic_int,
    shaped_array_env(),
    r#"
from shape_extensions import Dim, SymVar
from typing import reveal_type

def ordinary_literals() -> None:
    reveal_type(1 + 2)  # E: revealed type: int
    reveal_type(1 - 2)  # E: revealed type: int
    reveal_type(2 * 3)  # E: revealed type: int
    reveal_type(5 // 2)  # E: revealed type: int
    total = 1
    total += 2
    reveal_type(total)  # E: revealed type: int

def dim_literals[N: SymVar](x: Dim[N]) -> None:
    reveal_type(x + 1)  # E: revealed type: Dim
    reveal_type(1 + x)  # E: revealed type: Dim

def ordinary_typevar_value[T: int](x: T) -> None:
    reveal_type(x + 1)  # E: revealed type: int

def ordinary_unrestricted_typevar_value[T](x: T) -> None:
    x + 1  # E: `+` is not supported between `T` and `Literal[1]`
"#,
);

testcase!(
    test_legacy_symvar_treated_as_symvar,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Dim, SymVar
from torch import Tensor
from typing import Generic, assert_type, reveal_type

N = SymVar("N")
M = SymVar("M")

class Box(Generic[N]): ...

def f(n: Dim[N], shifted: Dim[N + 1], x: Tensor[[N, M]], shifted_x: Tensor[[N + 1, M]], y: Box[N]) -> None:
    reveal_type(n)  # E: revealed type: Dim[N]
    assert_type(shifted, Dim[N + 1])
    reveal_type(x)  # E: revealed type: Tensor[[N, M]]
    assert_type(shifted_x, Tensor[[N + 1, M]])
    reveal_type(y)  # E: revealed type: Box[N]
"#,
);

testcase!(
    test_symvar_type_parameter_bound,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Dim, Elements, SizeTuple, SymVar
from shape_extensions import SymVar as SV
import shape_extensions
import shape_extensions as se
from torch import Tensor
from typing import reveal_type

class SymBox[N: SymVar]: ...

def identity_alias[N: SV](x: Dim[N]) -> Dim[N]:
    return x

def identity_module[N: shape_extensions.SymVar](x: Dim[N]) -> Dim[N]:
    return x

def identity_module_alias[N: se.SymVar](x: Dim[N]) -> Dim[N]:
    return x

def shape[N: SymVar, M: SymVar, Shape: SizeTuple](
    n: Dim[N],
    x: Tensor[[N]],
    size: SizeTuple[N, M],
    packed: Tensor[[*Elements[Shape], N]],
    boxed: SymBox[N],
) -> None:
    reveal_type(n)  # E: revealed type: Dim[N]
    reveal_type(x)  # E: revealed type: Tensor[[N]]
    reveal_type(packed)  # E: revealed type: Tensor[[*Shape, N]]
    reveal_type(boxed)  # E: revealed type: SymBox[N]

def default_ok[N: SymVar, M: SymVar = N](x: Dim[M]) -> None:
    pass

def default_expr_ok[N: SymVar, M: SymVar = N + 1](x: Dim[M]) -> None:
    pass

type Shape[N: SymVar] = Tensor[[N]]
type Packed[Shape: SizeTuple, N: SymVar] = Tensor[[*Elements[Shape], N]]
type OrdinaryAlias[T, N: SymVar] = tuple[T, Dim[N]]

def alias_specialization[N: SymVar, ShapeT: SizeTuple](
    x: Shape[N],
    packed: Packed[ShapeT, N],
    ordinary: OrdinaryAlias[int, N],
) -> None:
    reveal_type(x)  # E: revealed type: Tensor[[N]]
    reveal_type(packed)  # E: revealed type: Tensor[[*ShapeT, N]]
    reveal_type(ordinary)  # E: revealed type: tuple[int, Dim[N]]
"#,
);

testcase!(
    test_symvar_rejected_in_ordinary_type_positions,
    shaped_array_env_with_shaped_torch(),
    r#"
from collections.abc import Callable
from shape_extensions import Dim, SymVar
from torch import Tensor
from typing import Generic, Optional, TypeAlias, TypeAliasType, TypeVar

LegacyN = SymVar("LegacyN")
OrdinaryT = TypeVar("OrdinaryT")
OrdinaryDefault = TypeVar("OrdinaryDefault", default=LegacyN)  # E: `LegacyN` is a `SymVar` and cannot be used as an ordinary type
BadSymDefault = SymVar("BadSymDefault", default=OrdinaryT)  # E: `OrdinaryT` must be a `SymVar` to be used as a shape dimension
BadIntDefault = SymVar("BadIntDefault", default=int)  # E: Tensor shape dimensions must be integer literals or type variables, got `type[int]`

class LegacyBox(Generic[LegacyN]): ...
class Box[T]: ...

def legacy_shape(n: Dim[LegacyN], x: Tensor[[LegacyN]]) -> None:
    pass

def legacy_invalid(
    x: LegacyN,  # E: `LegacyN` is a `SymVar` and cannot be used as an ordinary type
    y: list[LegacyN],  # E: `LegacyN` is a `SymVar` and cannot be used as an ordinary type
    z: Box[LegacyN],  # E: `LegacyN` is a `SymVar` and cannot be used as an ordinary type
) -> None:
    pass

def invalid[N: SymVar](
    x: N,  # E: `N` is a `SymVar` and cannot be used as an ordinary type
    y: list[N],  # E: `N` is a `SymVar` and cannot be used as an ordinary type
    z: Box[N],  # E: `N` is a `SymVar` and cannot be used as an ordinary type
    t: type[N],  # E: `N` is a `SymVar` and cannot be used as an ordinary type
    u: N | int,  # E: `N` is a `SymVar` and cannot be used as an ordinary type
    nested: int | (str | N),  # E: `N` is a `SymVar` and cannot be used as an ordinary type
    optional: Optional[N],  # E: `N` is a `SymVar` and cannot be used as an ordinary type
    c: Callable[[], N],  # E: `N` is a `SymVar` and cannot be used as an ordinary type
) -> None:
    pass

type Alias[N: SymVar] = N  # E: `N` is a `SymVar` and cannot be used as an ordinary type
type AliasUnion[N: SymVar] = N | int  # E: `N` is a `SymVar` and cannot be used as an ordinary type
LegacyAlias: TypeAlias = LegacyN | int  # E: `LegacyN` is a `SymVar` and cannot be used as an ordinary type
CallAlias = TypeAliasType("CallAlias", LegacyN | int, type_params=(LegacyN,))  # E: `LegacyN` is a `SymVar` and cannot be used as an ordinary type

def default_bad[T, N: SymVar = T](x: Dim[N]) -> None:  # E: `T` must be a `SymVar` to be used as a shape dimension
    pass

def default_bad_int[N: SymVar = int](x: Dim[N]) -> None:  # E: Tensor shape dimensions must be integer literals or type variables, got `type[int]`
    pass
"#,
);

testcase!(
    test_ordinary_typevar_shape_arithmetic_is_rejected,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D, Dim, SizeTuple
from torch import Tensor
from typing import Generic, TypeVar

LegacyN = TypeVar("LegacyN")

class LegacyBox(Generic[LegacyN]):
    legacy_tensor: Tensor[[LegacyN + 1]]  # E: `LegacyN` must be a `SymVar` to be used in shape arithmetic

def invalid[N](
    dim: Dim[N + 1],  # E: `N` must be a `SymVar` to be used in shape arithmetic
    tensor: Tensor[[N + 1]],  # E: `N` must be a `SymVar` to be used in shape arithmetic
    reversed_tensor: Tensor[[1 + N]],  # E: `N` must be a `SymVar` to be used in shape arithmetic
    tuple_shape: Tensor[SizeTuple[N + 1]],  # E: `N` must be a `SymVar` to be used in shape arithmetic
    negated: Tensor[[-N]],  # E: `N` must be a `SymVar` to be used in shape arithmetic
    bracket_launder: Tensor[[D[N] + 1]],  # E: `N` must be a `SymVar` to be used in shape arithmetic
    call_launder: Tensor[[D(N) // 2]],  # E: `N` must be a `SymVar` to be used in shape arithmetic
    inner_launder: Tensor[[D[N + 1]]],  # E: `N` must be a `SymVar` to be used in shape arithmetic
) -> None:
    pass
"#,
);

testcase!(
    test_kind_errors_recover_with_gradual_components,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Dim, SymVar
from torch import Tensor
from typing import Any, assert_type, reveal_type

def ordinary_type_recovery[N: SymVar](
    x: list[N],  # E: `N` is a `SymVar` and cannot be used as an ordinary type
    y: N | int,  # E: `N` is a `SymVar` and cannot be used as an ordinary type
) -> None:
    reveal_type(x)  # E: revealed type: list[Unknown]
    reveal_type(y)  # E: revealed type: int | Unknown

def symbolic_int_recovery[T](
    dim: Dim[T],  # E: `T` must be a `SymVar` to be used as a shape dimension
    tensor: Tensor[[T, 3]],  # E: `T` must be a `SymVar` to be used as a shape dimension
) -> None:
    assert_type(dim, Dim[Any])
    assert_type(tensor, Tensor[[Any, 3]])
"#,
);

testcase!(
    test_symvar_shape_arithmetic_is_accepted,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Dim, SymVar
from torch import Tensor
from typing import assert_type

LegacyN = SymVar("LegacyN")

def pep695[N: SymVar](dim: Dim[N + 1], tensor: Tensor[[N + 1]], negated: Tensor[[-N]]) -> None:
    pass

def legacy(dim: Dim[LegacyN + 1], tensor: Tensor[[LegacyN + 1]], negated: Tensor[[-LegacyN]]) -> None:
    assert_type(dim, Dim[LegacyN + 1])
    assert_type(tensor, Tensor[[LegacyN + 1]])
    assert_type(negated, Tensor[[-LegacyN]])
"#,
);

testcase!(
    test_symvar_special_form_is_only_a_kind_marker,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import SymVar
from typing import TypeVar

def ok[N: SymVar](x: object) -> None:
    pass

x: SymVar = 1  # E: Expected a type form, got instance of `_SpecialForm`
y: SymVar[int] = 1  # E: Expected a type form, got instance of `object`
T = TypeVar("T", bound=SymVar)  # E: `SymVar` cannot be used as a TypeVar bound # E: Expected a type form, got instance of `_SpecialForm`
U = TypeVar("U", SymVar, int)  # E: `SymVar` cannot be used as a TypeVar constraint # E: Expected a type form, got instance of `_SpecialForm`
V = TypeVar("V", default=SymVar)  # E: `SymVar` cannot be used as a TypeVar default # E: Expected a type form, got instance of `_SpecialForm`

def bad_constraint[T: (SymVar, int)](x: T) -> None:  # E: `SymVar` cannot be used as a TypeVar constraint
    pass

def invalid_param(x: SymVar) -> None:  # E: Expected a type form, got instance of `_SpecialForm`
    pass

def invalid_return() -> SymVar:  # E: Expected a type form, got instance of `_SpecialForm`
    pass
"#,
);

testcase!(
    test_symvar_class_type_parameter_accepts_dimension_expressions,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Dim, SymVar
from typing import Generic, assert_type, reveal_type

class ExplicitBox[N: SymVar]: ...

N = SymVar("N")
M = SymVar("M")

class LegacyBox(Generic[N]): ...

def explicit[N: SymVar](x: ExplicitBox[N + 1]) -> None:
    assert_type(x, ExplicitBox[N + 1])

def legacy(x: LegacyBox[N + M]) -> None:
    reveal_type(x)  # E: revealed type: LegacyBox[(N + M)]

def explicit_literals[S: SymVar](literal: ExplicitBox[3], symbolic: ExplicitBox[S]) -> None:
    assert_type(literal, ExplicitBox[3])
    assert_type(symbolic, ExplicitBox[S])
"#,
);

testcase!(
    test_dim_field_requires_symvar_class_type_parameter,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Dim

class FieldBox[N]:
    dim: Dim[N]  # E: `N` must be a `SymVar` to be used as a shape dimension
"#,
);

testcase!(
    test_sizetuple_elements_carrier_class_args_are_not_scalar_symvars,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Elements, SizeTuple, SymVar
from typing import assert_type

class TupleBox[Shape: SizeTuple]: ...
class PlainBox[N]: ...

def carrier[Bs: SizeTuple, N: SymVar](
    x: TupleBox[[*Elements[Bs], N + 1]],
    y: TupleBox[SizeTuple[*Elements[Bs], N + 1]],
) -> None:
    assert_type(x, TupleBox[[*Elements[Bs], N + 1]])
    assert_type(y, TupleBox[SizeTuple[*Elements[Bs], N + 1]])

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
    test_typevartuple_and_sizetuple_class_args_parse_separately,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import Elements, SizeTuple, SymVar
from typing import assert_type

class Mixed[*Ts, Shape: SizeTuple, N: SymVar]: ...

def f[*Ts, Shape: SizeTuple, N: SymVar](
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
from shape_extensions import D, SymVar
from typing import reveal_type
from torch import Tensor

def f[N: SymVar, M: SymVar](x: Tensor[[D[N] + D[M], D[N] * 2]]) -> None:
    reveal_type(x)  # E: revealed type: Tensor[[(N + M), (2 * N)]]
"#,
);

testcase!(
    test_shape_arithmetic_wrapper_call_form,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D, SymVar
from typing import reveal_type
from torch import Tensor

def f[N: SymVar, M: SymVar](x: Tensor[[D(N) // 2, D(N) ** D(M), -D(M)]]) -> None:
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
from shape_extensions import D, SymVar, assert_shape
from typing import assert_type
from torch import Tensor

def f[N: SymVar, M: SymVar](x: Tensor[[N, M]]) -> None:
    assert_type(assert_shape(x, (D[N], D(M))), Tensor[[N, M]])
    assert_shape(x, (D[M], D[N]))  # E: assert_shape((N, M), (M, N)) failed
    assert_shape(x, [D[N], D(M)])  # E: Second argument to `assert_shape` must be a tuple of tensor dimensions
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
    test_tuple_carrier_shape_context_preserves_starred_sizetuple,
    shaped_array_env(),
    r#"
from shape_extensions import Elements, SizeTuple, shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Tensor[Shape: SizeTuple]: ...

class Foo[Shape: SizeTuple]:
    x: Tensor[SizeTuple[*Elements[Shape]]]

def f[Shape: SizeTuple](x: Foo[Shape]) -> None:
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
    reveal_type(x.shape)  # E: revealed type: tuple[Literal[2], Literal[3]]
    reveal_type(x[0])  # E: revealed type: ndarray[[3], float]
    reveal_type(np.add_leading_axis(x))  # E: revealed type: ndarray[[1, 2, 3], float]
"#,
);

testcase!(
    test_jaxtyping_sizetuple_carrier_shapes,
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

# Jaxtyping shape annotations work on a TypeVar (SizeTuple) shape carrier, not just
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
    # The meta-shape DSL adds a leading axis. The result's raw tuple carrier is
    # re-synced to the computed shape, so both the displayed shape and `.shape`
    # stay coherent; DType is preserved.
    reveal_type(y)  # E: revealed type: tcarray[[1, 2, 3]]
    reveal_type(y.shape)  # E: revealed type: tuple[Literal[1], Literal[2], Literal[3]]
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

class symint: ...
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
    return shape_extensions.dsl.prod(x) + shape_extensions.dsl.sum(x)

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
from shape_extensions import shaped_array, uses_shape_dsl
from my_shapes import identity_ir, double_ir, scalar_kernel_ir, string_guard_ir, list_kernel_ir, iterator_kernel_ir, reductions_ir, svd_reduced_2d_ir, diag_1d_ir, einsum_kernel_ir, not_a_dsl_fn, bad_syntax_ir, kwargs_ir, calls_undefined, bad_no_ret, two_errors_ir, returns_wrong_type_ir, dims_as_scalar_union_ir, unknown_fallback_ir, helper_exact_one_ir, too_few_args_ir, too_many_args_ir
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
from shape_extensions import SizeTuple, uses_shape_dsl
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
def replace_leading_dim[Shape: SizeTuple](x: Tensor[Shape], dim: int) -> Tensor[Shape]: ...
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
