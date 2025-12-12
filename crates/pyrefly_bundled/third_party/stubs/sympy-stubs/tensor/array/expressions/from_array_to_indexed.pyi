from sympy import Equality, Mul, Ne, Sum
from sympy.core.add import Add
from sympy.core.relational import Relational
from sympy.tensor.array.expressions.array_expressions import ArrayElement

def convert_array_to_indexed(expr, indices) -> Mul | Equality | Relational | Ne | Sum | Add | ArrayElement: ...

class _ConvertArrayToIndexed:
    def __init__(self) -> None: ...
    def do_convert(self, expr, indices) -> Mul | Equality | Relational | Ne | Sum | Add | ArrayElement: ...
