"""Shape-generic named tuples returned by torch operations.

Mirrors ``torch.return_types``, which torch generates from the output names in
``native_functions.yaml`` as tuple subclasses with a property per field.
"""

from typing import Any

from shape_extensions import IntTuple
from torch import Tensor

class aminmax[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def min(self) -> Tensor[Shape]: ...
    @property
    def max(self) -> Tensor[Shape]: ...

class cummax[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class cummin[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class kthvalue[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class max[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class median[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class min[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class mode[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class sort[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class topk[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class linalg_slogdet[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def sign(self) -> Tensor[Shape]: ...
    @property
    def logabsdet(self) -> Tensor[Shape]: ...

class slogdet[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def sign(self) -> Tensor[Shape]: ...
    @property
    def logabsdet(self) -> Tensor[Shape]: ...

def __getattr__(name: str) -> Any: ...
