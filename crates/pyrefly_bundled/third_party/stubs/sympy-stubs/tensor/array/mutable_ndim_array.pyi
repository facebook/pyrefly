from typing_extensions import Self

from sympy.tensor.array.ndim_array import NDimArray

class MutableNDimArray(NDimArray):
    def as_immutable(self): ...
    def as_mutable(self) -> Self: ...
