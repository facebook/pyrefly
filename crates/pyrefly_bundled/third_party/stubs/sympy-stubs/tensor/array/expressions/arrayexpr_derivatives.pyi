from functools import singledispatch

from sympy import Basic
from sympy.core.expr import Expr
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.hadamard import HadamardProduct
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixSymbol
from sympy.matrices.expressions.special import Identity, OneMatrix
from sympy.matrices.expressions.transpose import Transpose
from sympy.tensor.array.expressions.array_expressions import (
    ArrayAdd,
    ArrayContraction,
    ArrayDiagonal,
    ArrayElementwiseApplyFunc,
    ArraySymbol,
    ArrayTensorProduct,
    PermuteDims,
    Reshape,
    ZeroArray,
    _ArrayExpr,
)

@singledispatch
def array_derive(expr, x): ...
@array_derive.register(Expr)
def _(expr: Expr, x: _ArrayExpr) -> ZeroArray: ...
@array_derive.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct, x: Expr) -> ZeroArray | ArrayAdd: ...
@array_derive.register(ArraySymbol)
def _(expr: ArraySymbol, x: _ArrayExpr) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims: ...
@array_derive.register(MatrixSymbol)
def _(expr: MatrixSymbol, x: _ArrayExpr) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims: ...
@array_derive.register(Identity)
def _(expr: Identity, x: _ArrayExpr) -> ZeroArray: ...
@array_derive.register(OneMatrix)
def _(expr: OneMatrix, x: _ArrayExpr) -> ZeroArray: ...
@array_derive.register(Transpose)
def _(expr: Transpose, x: Expr) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims: ...
@array_derive.register(Inverse)
def _(expr: Inverse, x: Expr) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims: ...
@array_derive.register(ElementwiseApplyFunction)
def _(
    expr: ElementwiseApplyFunction, x: Expr
) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims | ArrayDiagonal: ...
@array_derive.register(ArrayElementwiseApplyFunc)
def _(
    expr: ArrayElementwiseApplyFunc, x: Expr
) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims | ArrayDiagonal: ...
@array_derive.register(MatrixExpr)
def _(expr: MatrixExpr, x: Expr): ...
@array_derive.register(HadamardProduct)
def _(expr: HadamardProduct, x: Expr): ...
@array_derive.register(ArrayContraction)
def _(expr: ArrayContraction, x: Expr) -> Basic | ZeroArray | ArrayTensorProduct | ArrayContraction | PermuteDims: ...
@array_derive.register(ArrayDiagonal)
def _(
    expr: ArrayDiagonal, x: Expr
) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims | ArrayDiagonal: ...
@array_derive.register(ArrayAdd)
def _(expr: ArrayAdd, x: Expr) -> ZeroArray | ArrayAdd: ...
@array_derive.register(PermuteDims)
def _(expr: PermuteDims, x: Expr) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims: ...
@array_derive.register(Reshape)
def _(expr: Reshape, x: Expr) -> Reshape: ...
def matrix_derive(expr, x): ...
