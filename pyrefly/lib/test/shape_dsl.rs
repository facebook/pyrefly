/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;

use crate::test::class_keywords::get_class_metadata;
use crate::test::util::TestEnv;
use crate::test::util::testcase_for_macro;
use crate::testcase;

fn shaped_array_env() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "shape_extensions",
        "shape_extensions.pyi",
        r#"
from typing import Any, Callable

shaped_array: Any
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
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Tensor[*Shape]: ...
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
from shape_extensions.dsl import ShapedArray, shape_dsl_function

@shape_dsl_function
def add_leading_axis_ir(x: ShapedArray) -> ShapedArray:
    return ShapedArray(shape=[1] + x.shape)

@shaped_array(shape="Shape")
class ndarray[DType, *Shape]:
    shape: tuple[*Shape]
    def copy(self) -> ndarray[DType, *Shape]: ...
    def item(self) -> DType: ...

@uses_shape_dsl(add_leading_axis_ir)
def add_leading_axis[DType, *Shape](x: ndarray[DType, *Shape]) -> ndarray[DType, *Shape]: ...

@shaped_array(shape="Shape")
class tcarray[Shape, DType]:
    shape: Shape
    def dtype(self) -> DType: ...

@uses_shape_dsl(add_leading_axis_ir)
def tc_add_leading_axis[Shape, DType](x: tcarray[Shape, DType]) -> tcarray[Shape, DType]: ...
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
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Tensor[*Shape]:
    shape: tuple[*Shape]
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
from shape_extensions import shaped_array
from shape_extensions import shaped_array as shaped_array_alias

@shaped_array(shape="Shape")
class ImportedArray[*Shape]: ...

@shaped_array_alias(shape="Shape")
class ImportAliasArray[*Shape]: ...

@se.shaped_array(shape="Shape")
class ModuleAliasArray[DType, *Shape]: ...

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
        assert_shaped_array_shape(shape, "Shape", QuantifiedKind::TypeVarTuple);
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

testcase!(
    test_shaped_array_invalid_metadata,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array
from typing import Any, Generic, TypeVarTuple

kwargs: Any = {}

@shaped_array  # E: `@shaped_array` requires a `shape` keyword argument
class BareDecorator[*Shape]: ...

@shaped_array()  # E: `@shaped_array` requires a `shape` keyword argument
class MissingShape[*Shape]: ...

@shaped_array("Shape")  # E: `@shaped_array` expects `shape` as a keyword argument
class PositionalShape[*Shape]: ...

@shaped_array(dtype="Shape")  # E: Unexpected keyword argument `dtype` for `@shaped_array`; expected `shape`
class WrongShapeKeyword[*Shape]: ...

@shaped_array(shape="Shape", **kwargs)  # E: Unpacking is not supported in `@shaped_array`
class KwargsShape[*Shape]: ...

@shaped_array(shape="Shape", shape="Shape")  # E: Parse error: Duplicate keyword argument "shape"
class DuplicateShapeKeyword[*Shape]: ...

@shaped_array(shape=123)  # E: `@shaped_array` `shape` argument must be a string literal
class NonStringShape[*Shape]: ...

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a scoped (PEP-695-style) type parameter of class `NoTypeParams`
class NoTypeParams: ...

Shape = TypeVarTuple("Shape")

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a scoped (PEP-695-style) type parameter of class `LegacyGeneric`
class LegacyGeneric(Generic[*Shape]): ...

@shaped_array(shape="Shape")
@shaped_array(shape="Shape")  # E: Duplicate `@shaped_array` decorator
class DuplicateDecorator[*Shape]: ...

@shaped_array  # E: `@shaped_array` requires a `shape` keyword argument
@shaped_array(shape="Shape")  # E: Duplicate `@shaped_array` decorator
class DuplicateDecoratorAfterInvalid[*Shape]: ...

@shaped_array(shape="Missing")  # E: Shape parameter `Missing` is not a type parameter of class `ShapeNotFound`
class ShapeNotFound[*Shape]: ...

@shaped_array(shape="Shape")  # E: Shape parameter `Shape` must be a `TypeVar` or `TypeVarTuple`, got `ParamSpec`
class ShapeIsParamSpec[**Shape, DType]: ...
"#,
);

testcase!(
    test_shaped_array_compact_tuple_carrier,
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
    compact: Array[(2, 3), int],
    pep484: Array[tuple[Literal[2], Literal[3]], int],
    scalar: Array[(), int],
    dtype_first: DTypeFirstArray[int, (2, 3)],
) -> None:
    # Compact and PEP-484 forms reveal identically.
    reveal_type(compact)  # E: revealed type: Array[(2, 3), int]
    reveal_type(pep484)  # E: revealed type: Array[(2, 3), int]
    reveal_type(scalar)  # E: revealed type: Array[(), int]
    reveal_type(dtype_first)  # E: revealed type: DTypeFirstArray[int, (2, 3)]
    reveal_type(compact.dtype())  # E: revealed type: int
"#,
);

testcase!(
    test_shaped_array_pep484_tuple_carrier_canonicalization,
    shaped_array_env(),
    r#"
from typing import Literal, reveal_type
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(
    compact: Array[(2, 3), int],
    pep484: Array[tuple[Literal[2], Literal[3]], int],
    compact_scalar: Array[(), int],
    pep484_scalar: Array[tuple[()], int],
) -> None:
    # The compact and PEP-484 carriers canonicalize to the same shape.
    reveal_type(compact)  # E: revealed type: Array[(2, 3), int]
    reveal_type(pep484)  # E: revealed type: Array[(2, 3), int]
    reveal_type(compact_scalar)  # E: revealed type: Array[(), int]
    reveal_type(pep484_scalar)  # E: revealed type: Array[(), int]

    # Closed concrete shapes are mutually assignable in both directions.
    p: Array[tuple[Literal[2], Literal[3]], int] = compact
    c: Array[(2, 3), int] = pep484
    ps: Array[tuple[()], int] = compact_scalar
    cs: Array[(), int] = pep484_scalar

    wrong_rank2: Array[(2, 4), int] = pep484  # E: `Array[(2, 3), int]` is not assignable to `Array[(2, 4), int]`
    wrong_rank0: Array[(1,), int] = pep484_scalar  # E: `Array[(), int]` is not assignable to `Array[(1,), int]`
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
def non_shape_arg(x: DTypeFirstArray[tuple[int, ...], (2, 3)]) -> None:
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
    reveal_type(x)  # E: revealed type: Array[(2, 3), int]

# Tuple-carrier shapes with a bounded variadic middle remain valid: only
# rank-indefinite unbounded tuple middles are rejected.
def with_typevartuple_middle[*Ts](x: Array[tuple[Literal[2], *Ts], int]) -> None: ...

# Raw generic carriers (a bare type variable in the shape slot) remain valid.
def g[S](x: Array[S, int]) -> None: ...
"#,
);

testcase!(
    test_shaped_array_compact_tuple_arity_error,
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
    test_shaped_array_compact_tuple_invalid_dim,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

# Invalid compact dims get a dimension parser error at the bad element. The
# string is parsed as a forward reference before reaching the dimension parser,
# so the diagnostics are about the unresolved name / non-integer dimension --
# both pointing at the bad element, not a generic invalid type argument.
def f(bad: Array[("rows", 3), int]) -> None: ...  # E: Could not find name `rows`  # E: Tensor shape dimensions must be integer literals or type variables, got `Unknown`
"#,
);

testcase!(
    test_shaped_array_compact_tuple_rejects_starred_dim,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def f(bad: Array[(2, *tuple[int, ...]), int]) -> None: ...  # E: Tensor shape dimensions must be positive integer literals, string literals, type variables, or expressions, got `*tuple[int, ...]`
"#,
);

testcase!(
    test_shaped_array_annotation_parsing,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Array[DType, *Shape]:
    def __init__(self) -> None: ...
    def dtype(self) -> DType: ...

class Cpu: ...
class Gpu: ...

@shaped_array(shape="Shape")
class ArrayWithDevice[DType, *Shape, Device: (Gpu, Cpu)]:
    def dtype(self) -> DType: ...
    def device(self) -> Device: ...

@shaped_array(shape="Shape")
class SuffixArray[*Shape, DType]:
    def dtype(self) -> DType: ...

def f(
    x: Array[int, 2, 3],
    y: Array[int],
    z: Array[int, 2, *tuple[int, ...]],
    w: ArrayWithDevice[str, 2, 3, Cpu],
    w_scalar: ArrayWithDevice[str, Gpu],
    suffix: SuffixArray[2, 3, str],
    suffix_scalar: SuffixArray[str],
) -> None:
    reveal_type(x)  # E: revealed type: Array[int, 2, 3]
    reveal_type(x.dtype())  # E: revealed type: int
    reveal_type(y)  # E: revealed type: Array[int, ()]
    reveal_type(y.dtype())  # E: revealed type: int
    reveal_type(z)  # E: revealed type: Array[int, 2, *tuple[int, ...]]
    reveal_type(z.dtype())  # E: revealed type: int
    reveal_type(w)  # E: revealed type: ArrayWithDevice[str, 2, 3, Cpu]
    reveal_type(w.dtype())  # E: revealed type: str
    reveal_type(w.device())  # E: revealed type: Cpu
    reveal_type(w_scalar)  # E: revealed type: ArrayWithDevice[str, (), Gpu]
    reveal_type(w_scalar.dtype())  # E: revealed type: str
    reveal_type(w_scalar.device())  # E: revealed type: Gpu
    reveal_type(suffix)  # E: revealed type: SuffixArray[2, 3, str]
    reveal_type(suffix.dtype())  # E: revealed type: str
    reveal_type(suffix_scalar)  # E: revealed type: SuffixArray[(), str]
    reveal_type(suffix_scalar.dtype())  # E: revealed type: str

def g(x: Array) -> None:
    reveal_type(x)  # E: revealed type: Array

def bad_arg_count(x: ArrayWithDevice[int]) -> None:  # E: Expected 3 type arguments for `ArrayWithDevice`, got 1
    pass
"#,
);

testcase!(
    test_shaped_array_indexing_and_bare_values,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Array[DType, *Shape]:
    def __init__(self) -> None: ...
    def dtype(self) -> DType: ...

def annotations(concrete: Array[int, 2, 3], scalar: Array[int], shapeless: Array) -> None:
    reveal_type(concrete[0])  # E: revealed type: Array[int, 3]
    reveal_type(concrete[:])  # E: revealed type: Array[int, 2, 3]
    reveal_type(concrete[0].dtype())  # E: revealed type: int
    scalar[0]  # E: Cannot index scalar tensor (rank 0)
    reveal_type(shapeless)  # E: revealed type: Array
    reveal_type(shapeless[0])  # E: revealed type: Array
    reveal_type(shapeless[None])  # E: revealed type: Array[1, *tuple[Unknown, ...]]
    reveal_type(shapeless[None, ...])  # E: revealed type: Array[1, *tuple[Unknown, ...]]

def unbounded_shape(x: Array[int, *tuple[int, ...]], y: Array[int, 2, *tuple[int, ...]]) -> None:
    reveal_type(x)  # E: revealed type: Array[int, *tuple[int, ...]]
    reveal_type(y[0])  # E: revealed type: Array[int, *tuple[int, ...]]

def values() -> None:
    value = Array()
    reveal_type(value)  # E: revealed type: Array[Unknown, *tuple[Unknown, ...]]
    reveal_type(value[0])  # E: revealed type: Array

def index_preserves_dtype(concrete: Array[int, 2, 3]) -> Array[int, 3]:
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

def f(x: Array[(2, 3, 4), int], dtype_first: DTypeFirstArray[int, (2, 3, 4)]) -> None:
    # Integer index drops the leading dim, and `.shape` (read from the raw
    # carrier) stays coherent with the projected shape.
    reveal_type(x[0])  # E: revealed type: Array[(3, 4), int]
    reveal_type(x[0].shape)  # E: revealed type: tuple[Literal[3], Literal[4]]
    reveal_type(x[0].dtype())  # E: revealed type: int

    # Mixed tuple index (slice + int) and `None`/newaxis stay coherent too.
    reveal_type(x[:, 0])  # E: revealed type: Array[(2, 4), int]
    reveal_type(x[:, 0].shape)  # E: revealed type: tuple[Literal[2], Literal[4]]
    reveal_type(x[None])  # E: revealed type: Array[(1, 2, 3, 4), int]
    reveal_type(x[None].shape)  # E: revealed type: tuple[Literal[1], Literal[2], Literal[3], Literal[4]]

    # The carrier rewrite follows the registered shape parameter, even when it
    # is not the first type argument.
    reveal_type(dtype_first[0])  # E: revealed type: DTypeFirstArray[int, (3, 4)]
    reveal_type(dtype_first[0].shape)  # E: revealed type: tuple[Literal[3], Literal[4]]
    reveal_type(dtype_first[0].dtype())  # E: revealed type: int

def scalar(s: Array[(), int]) -> None:
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

def f(x: Array[(2, 3), int], y: Array[(1, 3), int]) -> None:
    z = x + y
    # Broadcasting `(2, 3)` with `(1, 3)` yields `(2, 3)`, and the raw carrier is
    # rewritten so `.shape` stays coherent. DType is preserved from the carrier.
    reveal_type(z)  # E: revealed type: Array[(2, 3), int]
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
    compact_2_3: Array[(2, 3), int],
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
    reveal_type(make_array(shape_2_3))  # E: revealed type: Array[(2, 3), float]
    reveal_type(make_array(scalar_shape))  # E: revealed type: Array[(), float]
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

def f(x_2_3_int: Array[(2, 3), int]) -> None:
    reveal_type(identity(x_2_3_int))  # E: revealed type: Array[(2, 3), int]
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
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

@shaped_array(shape="Shape")
class Tensor[DType, *Shape]: ...

def carrier[S](x: Array[S, float]) -> None:
    reveal_type(x.shape)  # E: revealed type: S

def concrete[M](x: Array[(2, 4, M), float]) -> None:
    reveal_type(x.shape)  # E: revealed type: tuple[Literal[2], Literal[4], Dim[M]]

def unpacked_prefix[*Ts](x: Array[tuple[Literal[2], *Ts], float]) -> None:
    reveal_type(x.shape)  # E: revealed type: tuple[Literal[2], *Ts]

def typevartuple[*Shape](x: Tensor[float, *Shape]) -> None:
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

def want_int(x: Array[(2, 3), int]) -> None: ...

def f(x_str: Array[(2, 3), str]) -> None:
    want_int(x_str)  # E: Argument `Array[(2, 3), str]` is not assignable to parameter `x` with type `Array[(2, 3), int]`
"#,
);

testcase!(
    test_shaped_array_tuple_carrier_closed_shapes_still_check_dimensions,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array

@shaped_array(shape="Shape")
class Array[Shape, DType]: ...

def want_2_4(x: Array[(2, 4), int]) -> None: ...

def f(x_2_3: Array[(2, 3), int]) -> None:
    want_2_4(x_2_3)  # E: Argument `Array[(2, 3), int]` is not assignable to parameter `x` with type `Array[(2, 4), int]`
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

def want_2_3(x: Array[(2, 3), int]) -> None: ...
def want_bad(x: Array[tuple[str, str], int]) -> None: ...

# `tuple[str, str]` is not a valid shape carrier. It projects to a shapeless
# array internally; a source-aware diagnostic rejecting this form is deferred.
def f(x_bad: Array[tuple[str, str], int]) -> None:
    want_2_3(x_bad)  # E: Argument `Array[tuple[Unknown, ...], int]` is not assignable to parameter `x` with type `Array[(2, 3), int]`
    want_bad(x_bad)

def g(x_2_3: Array[(2, 3), int]) -> None:
    want_bad(x_2_3)  # E: Argument `Array[(2, 3), int]` is not assignable to parameter `x` with type `Array[tuple[Unknown, ...], int]`
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
from shape_extensions import Dim, shaped_array
from typing import TypeVar, reveal_type

T = TypeVar("T")
DefaultT = TypeVar("DefaultT", default=3)  # E: Expected a type form, got instance of `Literal[3]`

class Box[T]: ...
class DefaultBox[T = 3]: ...  # E: Expected a type form, got instance of `Literal[3]`

@shaped_array(shape="Shape")
class Array[DType, *Shape, Device]: ...

@shaped_array(shape="Shape")
class SuffixArray[*Shape, DType]: ...

class Cpu: ...
class Gpu: ...

type Image = Array[int, 2, 3, Cpu]

def ordinary_type_arguments(x: Box[3]) -> None:  # E: Expected a type form, got instance of `Literal[3]`
    pass

def shaped_array_segments(
    good: Array[int, 2, 3, Cpu],
    bad_prefix: Array[3, 2, 3, Cpu],  # E: Expected a type form, got instance of `Literal[3]`
    bad_suffix: SuffixArray[2, 3, 3],  # E: Expected a type form, got instance of `Literal[3]`
    alias: Image,
) -> None:
    reveal_type(good)  # E: revealed type: Array[int, 2, 3, Cpu]
    reveal_type(alias)  # E: revealed type: Array[int, 2, 3, Cpu]

def dims[N](concrete: Dim[3], symbolic: Dim[N + 1]) -> None:
    pass
"#,
);

testcase!(
    test_tensor_shapes_keeps_ordinary_literal_arithmetic_int,
    shaped_array_env(),
    r#"
from shape_extensions import Dim
from typing import reveal_type

def ordinary_literals() -> None:
    reveal_type(1 + 2)  # E: revealed type: int
    reveal_type(1 - 2)  # E: revealed type: int
    reveal_type(2 * 3)  # E: revealed type: int
    reveal_type(5 // 2)  # E: revealed type: int
    total = 1
    total += 2
    reveal_type(total)  # E: revealed type: int

def dim_literals[N](x: Dim[N]) -> None:
    reveal_type(x + 1)  # E: revealed type: Dim
    reveal_type(1 + x)  # E: revealed type: Dim

def ordinary_typevar_value[T: int](x: T) -> None:
    reveal_type(x + 1)  # E: revealed type: int

def ordinary_unrestricted_typevar_value[T](x: T) -> None:
    x + 1  # E: `+` is not supported between `T` and `Literal[1]`
"#,
);

testcase!(
    test_decorated_torch_tensor_parses_shapes,
    shaped_array_env_with_shaped_torch(),
    r#"
from typing import reveal_type
from torch import Tensor

def f(x: Tensor[2, 3], y: Tensor) -> None:
    reveal_type(x)  # E: revealed type: Tensor[2, 3]
    reveal_type(y)  # E: revealed type: Tensor
    reveal_type(x[0])  # E: revealed type: Tensor[3]
    reveal_type(y[0])  # E: revealed type: Tensor
"#,
);

testcase!(
    test_shape_arithmetic_wrapper_bracket_form,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D
from typing import reveal_type
from torch import Tensor

def f[N, M](x: Tensor[D[N] + D[M], D[N] * 2]) -> None:
    reveal_type(x)  # E: revealed type: Tensor[(N + M), (2 * N)]
"#,
);

testcase!(
    test_shape_arithmetic_wrapper_call_form,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D
from typing import reveal_type
from torch import Tensor

def f[N, M](x: Tensor[D(N) // 2, D(N) ** D(M), -D(M)]) -> None:
    reveal_type(x)  # E: revealed type: Tensor[(N // 2), (N ** M), (-1 * M)]
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
    no_arg: Tensor[D()],  # E: Expected 1 positional argument for `D`, got 0
    too_many: Tensor[D(N, M)],  # E: Expected 1 positional argument for `D`, got 2
    keyword: Tensor[D(N, dim=M)],  # E: `D` accepts exactly 1 positional argument and no keyword arguments, got 1 positional and 1 keyword
    non_d_subscript: Tensor[Box[N]],  # E: Tensor shape dimensions must be positive integer literals, string literals, type variables, or expressions
    non_d_call: Tensor[Factory(N)],  # E: Tensor shape dimensions must be positive integer literals, string literals, type variables, or expressions
) -> None:
    pass
"#,
);

testcase!(
    test_assert_shape_builtin,
    shaped_array_env_with_shaped_torch(),
    r#"
from shape_extensions import D, assert_shape
from typing import assert_type
from torch import Tensor

def f[N, M](x: Tensor[N, M]) -> None:
    assert_type(assert_shape(x, (D[N], D(M))), Tensor[N, M])
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

def f(x: Tensor[2, 3]) -> None:
    assert_type(check_shape(x, (2, 3)), Tensor[2, 3])
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
    test_variadic_shape_context_preserves_starred_typevartuple,
    shaped_array_env(),
    r#"
from shape_extensions import shaped_array
from typing import reveal_type

@shaped_array(shape="Shape")
class Tensor[*Shape]: ...

class Foo[*Shape]:
    x: Tensor[*Shape]

def f[*Shape](x: Foo[*Shape]) -> None:
    reveal_type(x)  # E: revealed type: Foo[*Shape]
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

def f(x: np.ndarray[float, 2, 3]) -> None:
    reveal_type(x)  # E: revealed type: ndarray[float, 2, 3]
    reveal_type(x.copy())  # E: revealed type: ndarray[float, 2, 3]
    reveal_type(x.item())  # E: revealed type: float
    reveal_type(x.shape)  # E: revealed type: tuple[Literal[2], Literal[3]]
    reveal_type(x[0])  # E: revealed type: ndarray[float, 3]
    reveal_type(np.add_leading_axis(x))  # E: revealed type: ndarray[float, 1, 2, 3]
"#,
);

testcase!(
    test_numpy_tuple_carrier_meta_shape_keeps_shape_coherent,
    shaped_array_env_with_numpy(),
    r#"
import numpy as np
from typing import Literal, reveal_type

def f(x: np.tcarray[(2, 3), int]) -> None:
    y = np.tc_add_leading_axis(x)
    # The meta-shape DSL adds a leading axis. The result's raw tuple carrier is
    # re-synced to the computed shape, so both the displayed shape and `.shape`
    # stay coherent; DType is preserved.
    reveal_type(y)  # E: revealed type: tcarray[(1, 2, 3), int]
    reveal_type(y.shape)  # E: revealed type: tuple[Literal[1], Literal[2], Literal[3]]
    reveal_type(y.dtype())  # E: revealed type: int
"#,
);

fn shape_dsl_env() -> TestEnv {
    let mut env = shape_dsl_base_env();
    env.add_with_path(
        "my_shapes",
        "my_shapes.pyi",
        r#"
from typing import Any
from shape_extensions.dsl import shape_dsl_function
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
from typing import Any, overload
from shape_extensions import uses_shape_dsl
from my_shapes import identity_ir, double_ir, scalar_kernel_ir, string_guard_ir, list_kernel_ir, iterator_kernel_ir, reductions_ir, einsum_kernel_ir, not_a_dsl_fn, bad_syntax_ir, kwargs_ir, calls_undefined, bad_no_ret, two_errors_ir, returns_wrong_type_ir, dims_as_scalar_union_ir, unknown_fallback_ir, helper_exact_one_ir, too_few_args_ir, too_many_args_ir
import my_shapes

non_literal: Any

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
from shape_extensions import uses_shape_dsl
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
def replace_leading_dim[*Shape](x: Tensor[*Shape], dim: int) -> Tensor[*Shape]: ...
"#,
        );
        env
    },
    r#"
from shape_ops import replace_leading_dim
from torch import Tensor
from typing import Literal, assert_type

def f(x: Tensor[2, 3]) -> None:
    assert_type(x.shape, tuple[Literal[2], Literal[3]])
    assert_type(replace_leading_dim(x, 4), Tensor[4, 3])
"#,
);
