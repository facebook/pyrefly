from functools import singledispatch
from trace import Trace
from typing import Any

from sympy import Basic, MatAdd, MatrixExpr, Mul, Transpose, ZeroMatrix
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import GenericIdentity, GenericZeroMatrix
from sympy.series.order import Order
from sympy.tensor.array.expressions.array_expressions import (
    ArrayAdd,
    ArrayContraction,
    ArrayDiagonal,
    ArrayElement,
    ArrayElementwiseApplyFunc,
    ArrayTensorProduct,
    PermuteDims,
    ZeroArray,
)

@singledispatch
def _array2matrix(expr) -> Any: ...
@_array2matrix.register(ZeroArray)
def _(expr: ZeroArray) -> ZeroMatrix | ZeroArray: ...
@_array2matrix.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct) -> Mul | ArrayTensorProduct: ...
@_array2matrix.register(ArrayContraction)
def _(
    expr: ArrayContraction,
) -> (
    GenericIdentity
    | Order
    | object
    | Mul
    | ArrayTensorProduct
    | ZeroArray
    | ArrayContraction
    | Basic
    | PermuteDims
    | Trace
    | None
): ...
@_array2matrix.register(ArrayDiagonal)
def _(expr: ArrayDiagonal) -> ArrayDiagonal: ...
@_array2matrix.register(PermuteDims)
def _(expr: PermuteDims) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims | Any | Transpose | Mul: ...
@_array2matrix.register(ArrayAdd)
def _(expr: ArrayAdd) -> MatAdd | GenericZeroMatrix | ZeroArray | ArrayAdd: ...
@_array2matrix.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc) -> MatrixExpr | ElementwiseApplyFunction | ArrayElementwiseApplyFunc: ...
@_array2matrix.register(ArrayElement)
def _(expr: ArrayElement) -> MatrixElement | ArrayElement: ...
@singledispatch
def _remove_trivial_dims(expr) -> Any: ...
@_remove_trivial_dims.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct) -> tuple[ArrayTensorProduct | Mul, list[int] | list[Any]]: ...
@_remove_trivial_dims.register(ArrayAdd)
def _(
    expr: ArrayAdd,
) -> tuple[ArrayAdd, list[Any]] | tuple[ArrayAdd, Any] | tuple[MatAdd | Any | GenericZeroMatrix | ZeroArray | ArrayAdd, Any]: ...
@_remove_trivial_dims.register(PermuteDims)
def _(
    expr: PermuteDims,
) -> tuple[Any | ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims, list[int] | list[Any]]: ...
@_remove_trivial_dims.register(ArrayContraction)
def _(
    expr: ArrayContraction,
) -> tuple[Any, list[int]] | tuple[Basic | Any | ZeroArray | ArrayTensorProduct | ArrayContraction | PermuteDims, list[int]]: ...
@_remove_trivial_dims.register(ArrayDiagonal)
def _(
    expr: ArrayDiagonal,
) -> (
    tuple[Any | ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims, list[int]]
    | tuple[Any | ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims, list[Any]]
): ...
@_remove_trivial_dims.register(ElementwiseApplyFunction)
def _(expr: ElementwiseApplyFunction) -> tuple[Any, list[Any]] | tuple[MatrixExpr | ElementwiseApplyFunction, list[Any]]: ...
@_remove_trivial_dims.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc) -> tuple[ArrayElementwiseApplyFunc, list[Any]]: ...
def convert_array_to_matrix(expr): ...
def identify_hadamard_products(
    expr: ArrayContraction | ArrayDiagonal,
) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims: ...
def identify_removable_identity_matrices(expr) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims: ...
def remove_identity_matrices(
    expr: ArrayContraction,
) -> tuple[Any | ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims, list[int]]: ...
