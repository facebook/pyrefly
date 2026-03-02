from typing import Any, Literal
from typing_extensions import Self

from sympy import Basic, Derivative, ImmutableDenseNDimArray, ImmutableSparseNDimArray, NDimArray
from sympy.core._print_helpers import Printable
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.tensor.array.expressions.array_expressions import (
    ArrayContraction,
    ArrayDiagonal,
    ArrayTensorProduct,
    PermuteDims,
    ZeroArray,
)

def tensorproduct(
    *args,
) -> (
    NDimArray
    | ImmutableDenseNDimArray
    | ZeroArray
    | ArrayTensorProduct
    | ArrayContraction
    | Basic
    | PermuteDims
    | ImmutableSparseNDimArray
): ...
def tensorcontraction(
    array, *contraction_axes
) -> Basic | ZeroArray | ArrayTensorProduct | ArrayContraction | PermuteDims | ImmutableDenseNDimArray | Any: ...
def tensordiagonal(
    array, *diagonal_axes
) -> ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims | ArrayDiagonal | ImmutableDenseNDimArray | Any: ...
def derive_by_array(expr, dx) -> ImmutableDenseNDimArray | ImmutableSparseNDimArray | Any | ArrayDerivative | Derivative: ...
def permutedims(
    expr, perm=..., index_order_old=..., index_order_new=...
) -> (
    ZeroArray | ArrayTensorProduct | ArrayContraction | Basic | PermuteDims | ImmutableSparseNDimArray | ImmutableDenseNDimArray
): ...

class Flatten(Printable):
    def __init__(self, iterable) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Basic | Literal[0]: ...
    def next(self) -> Basic | Literal[0]: ...
