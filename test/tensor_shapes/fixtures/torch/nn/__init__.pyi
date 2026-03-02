# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Type stubs for torch.nn module.
"""

from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    overload,
    Self,
    TYPE_CHECKING,
    TypedDict,
    TypeVar,
)

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim as _Dim

# Re-export submodules
from . import functional as functional, init as init

# Base class for all neural network modules
class Module:
    """
    Base class for all neural network modules.

    Your models should subclass this class.
    """

    training: bool

    def __init__(self) -> None: ...
    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None: ...
    def register_parameter(self, name: str, param: Parameter | None) -> None: ...
    def apply(self, fn: Callable[[Module], None]) -> Self: ...
    def parameters(self, recurse: bool = True) -> Iterator[Tensor]: ...
    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Tensor]]: ...
    def state_dict(
        self,
        destination: dict[str, Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Tensor]: ...
    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any: ...
    def _register_load_state_dict_pre_hook(
        self,
        hook: Callable[[dict[str, Tensor], str], None],
        with_module: bool = False,
    ) -> Any:
        """Register a hook to be called before loading state_dict."""
        ...

# Parameter wrapper
# In PyTorch, nn.Parameter is a class, but for type checking we model it as a function
# that returns Tensor (not Parameter) to match runtime behavior where operations on
# Parameters return Tensors. This makes the type system simpler and more accurate.
def Parameter[*Shape](
    data: Tensor[*Shape], requires_grad: bool = True
) -> Tensor[*Shape]:
    """
    Wraps a tensor as a module parameter.
    Returns the tensor (for type purposes) since operations on Parameters return Tensors.
    """
    ...

# Buffer wrapper
# Similar to Parameter, Buffer wraps a tensor that is not a parameter but should be
# part of the module's state_dict. For type checking we model it as returning Tensor.
def Buffer[*Shape](data: Tensor[*Shape], persistent: bool = True) -> Tensor[*Shape]:
    """
    Wraps a tensor as a module buffer.
    Returns the tensor (for type purposes) since operations on Buffers return Tensors.
    """
    ...

# Linear layer
class Linear[IN, OUT](Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b"""

    weight: Tensor[OUT, IN]
    bias: Tensor[OUT] | None

    def __init__(
        self,
        in_features: _Dim[IN],
        out_features: _Dim[OUT],
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*Bs](self, input: Tensor[*Bs, IN]) -> Tensor[*Bs, OUT]: ...

# Dropout
class Dropout(Module):
    """During training, randomly zeroes some of the elements of the input tensor with probability p"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[*Shape](self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...

# GELU activation
class GELU(Module):
    """Applies the Gaussian Error Linear Units function"""
    def __init__(self, approximate: str = "none") -> None: ...
    def forward[*Shape](self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...

# Embedding
class Embedding[NUM_EMB, EMB_DIM](Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size"""

    weight: Tensor[NUM_EMB, EMB_DIM]

    def __init__(
        self,
        num_embeddings: _Dim[NUM_EMB],
        embedding_dim: _Dim[EMB_DIM],
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        _freeze: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

    # 1D input: [T] -> [T, EMB_DIM]
    @overload
    def forward[T](self, input: Tensor[T]) -> Tensor[T, EMB_DIM]: ...

    # 2D input: [B, T] -> [B, T, EMB_DIM]
    @overload
    def forward[B, T](self, input: Tensor[B, T]) -> Tensor[B, T, EMB_DIM]: ...

# ModuleDict
class ModuleDict[T](Module):
    """Holds submodules in a dictionary"""
    def __init__(self, modules: T) -> None: ...
    def __getitem__(self, key: str) -> Module: ...
    def __setitem__(self, key: str, module: Module) -> None: ...
    def __getattr__(self, name: str) -> Module: ...  # Support attribute access
    def __iter__(self) -> Iterator[str]: ...
    def keys(self) -> Iterator[str]: ...
    def items(self) -> Iterator[tuple[str, Module]]: ...
    def values(self) -> Iterator[Module]: ...

# Sequential container
class Sequential(Module):
    """
    A sequential container. Modules will be added to it in the order they are passed.
    """
    def __init__(self, *args: Module) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, input: Tensor) -> Tensor: ...

# ModuleList container
class ModuleList[T](Module):
    """
    Holds modules in a list.
    """
    def __init__(self, modules: Iterable[T] | None = None) -> None: ...
    def __getitem__(self, idx: int) -> T: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...
    def append(self, module: T) -> None: ...

__all__ = [
    "functional",
    "init",
    "Module",
    "Parameter",
    "Buffer",
    "Linear",
    "Dropout",
    "GELU",
    "Embedding",
    "ModuleDict",
    "Sequential",
    "ModuleList",
]
