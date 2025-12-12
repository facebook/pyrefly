from sympy import Basic, Pow
from sympy.functions.special.tensor_functions import KroneckerDelta
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

def convert_indexed_to_array(
    expr, first_indices=...
) -> (
    ArrayElement
    | Basic
    | ZeroArray
    | ArrayTensorProduct
    | ArrayContraction
    | PermuteDims
    | ArrayDiagonal
    | KroneckerDelta
    | ArrayAdd
    | ArrayElementwiseApplyFunc
    | Pow
): ...
