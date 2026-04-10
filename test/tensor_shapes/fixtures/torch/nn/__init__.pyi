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
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None: ...
    def register_parameter(self, name: str, param: Parameter | None) -> None: ...
    def apply(self, fn: Callable[[Module], None]) -> Self: ...
    def to(self, *args: Any, **kwargs: Any) -> Self: ...
    def modules(self) -> Iterator[Module]: ...
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
class Sequential[*Ms](Module):
    """
    A sequential container. Modules will be added to it in the order they are passed.
    When type arguments are known, calling the Sequential chains input through each
    module's forward method, preserving shape information.
    """
    def __init__(self, *args: *Ms) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

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

# ==============================================================================
# Activation Modules (shape-preserving)
# ==============================================================================

class ReLU(Module):
    """Applies ReLU activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class ReLU6(Module):
    """Applies ReLU6 activation (clamps to [0, 6])"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class SiLU(Module):
    """Applies SiLU (Swish) activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Sigmoid(Module):
    """Applies element-wise Sigmoid"""
    def __init__(self) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Tanh(Module):
    """Applies element-wise Tanh"""
    def __init__(self) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Mish(Module):
    """Applies Mish activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Hardswish(Module):
    """Applies Hardswish activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Hardsigmoid(Module):
    """Applies Hardsigmoid activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class LeakyReLU(Module):
    """Applies LeakyReLU activation"""
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class ELU(Module):
    """Applies ELU activation"""
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class SELU(Module):
    """Applies SELU activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class CELU(Module):
    """Applies CELU activation"""
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Softplus(Module):
    """Applies Softplus activation"""
    def __init__(self, beta: float = 1, threshold: float = 20) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class PReLU(Module):
    """Applies PReLU activation"""
    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Threshold(Module):
    """Applies Threshold activation"""
    def __init__(
        self, threshold: float, value: float, inplace: bool = False
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Softmax(Module):
    """Applies Softmax along a dimension"""
    def __init__(self, dim: int | None = None) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class LogSoftmax(Module):
    """Applies LogSoftmax along a dimension"""
    def __init__(self, dim: int | None = None) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

# ==============================================================================
# Normalization Modules (shape-preserving)
# ==============================================================================

class LayerNorm(Module):
    """Applies Layer Normalization"""
    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class RMSNorm(Module):
    """Applies Root Mean Square Layer Normalization"""
    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-8,
        elementwise_affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class GroupNorm(Module):
    """Applies Group Normalization"""
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class BatchNorm1d(Module):
    """Applies Batch Normalization over a 2D or 3D input"""

    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class BatchNorm2d(Module):
    """Applies Batch Normalization over a 4D input"""

    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class BatchNorm3d(Module):
    """Applies Batch Normalization over a 5D input"""

    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class InstanceNorm1d(Module):
    """Applies Instance Normalization over a 3D input"""
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class InstanceNorm2d(Module):
    """Applies Instance Normalization over a 4D input"""
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class InstanceNorm3d(Module):
    """Applies Instance Normalization over a 5D input"""
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

# ==============================================================================
# Dropout Modules (shape-preserving)
# ==============================================================================

class Dropout1d(Module):
    """Randomly zero out entire channels (1D)"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Dropout2d(Module):
    """Randomly zero out entire channels (2D)"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class Dropout3d(Module):
    """Randomly zero out entire channels (3D)"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class AlphaDropout(Module):
    """Applies Alpha Dropout for SELU networks"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

class FeatureAlphaDropout(Module):
    """Randomly masks entire channels with Alpha Dropout"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

# ==============================================================================
# Other Shape-Preserving Modules
# ==============================================================================

class Identity(Module):
    """Identity module that returns the input unchanged"""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward[*S](self, input: Tensor[*S]) -> Tensor[*S]: ...

# ==============================================================================
# Convolution Modules
# ==============================================================================

class Conv1d[InC, OutC, K, S = 1, P = 0, D = 1](Module):
    """1D convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, D are bound from constructor arguments via _Dim[T].
    PEP 696 defaults (S=1, P=0, D=1) apply when arguments are omitted.
    """

    weight: Tensor[OutC, InC, K]

    def __init__(
        self,
        in_channels: _Dim[InC],
        out_channels: _Dim[OutC],
        kernel_size: _Dim[K],
        stride: _Dim[S] = 1,
        padding: _Dim[P] = 0,
        dilation: _Dim[D] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B, L](
        self, input: Tensor[B, InC, L]
    ) -> Tensor[B, OutC, (L + 2 * P - D * (K - 1) - 1) // S + 1]: ...

class Conv2d[InC, OutC, K, S = 1, P = 0, D = 1](Module):
    """2D convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, D are bound from constructor arguments via _Dim[T].
    PEP 696 defaults (S=1, P=0, D=1) apply when arguments are omitted.
    """

    weight: Tensor[OutC, InC, K, K]

    def __init__(
        self,
        in_channels: _Dim[InC],
        out_channels: _Dim[OutC],
        kernel_size: _Dim[K],
        stride: _Dim[S] = 1,
        padding: _Dim[P] = 0,
        dilation: _Dim[D] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B, H, W](
        self, input: Tensor[B, InC, H, W]
    ) -> Tensor[
        B,
        OutC,
        (H + 2 * P - D * (K - 1) - 1) // S + 1,
        (W + 2 * P - D * (K - 1) - 1) // S + 1,
    ]: ...

class Conv3d[InC, OutC, K, S = 1, P = 0, D = 1](Module):
    """3D convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, D are bound from constructor arguments via _Dim[T].
    PEP 696 defaults (S=1, P=0, D=1) apply when arguments are omitted.
    """

    weight: Tensor[OutC, InC, K, K, K]

    def __init__(
        self,
        in_channels: _Dim[InC],
        out_channels: _Dim[OutC],
        kernel_size: _Dim[K],
        stride: _Dim[S] = 1,
        padding: _Dim[P] = 0,
        dilation: _Dim[D] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B, D_, H, W](
        self, input: Tensor[B, InC, D_, H, W]
    ) -> Tensor[
        B,
        OutC,
        (D_ + 2 * P - D * (K - 1) - 1) // S + 1,
        (H + 2 * P - D * (K - 1) - 1) // S + 1,
        (W + 2 * P - D * (K - 1) - 1) // S + 1,
    ]: ...

class ConvTranspose1d[InC, OutC, K, S = 1, P = 0, OP = 0, D = 1](Module):
    """1D transposed convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, OP, D are bound from constructor arguments via _Dim[T].
    PEP 696 defaults apply when arguments are omitted.
    """

    weight: Tensor[InC, OutC, K]

    def __init__(
        self,
        in_channels: _Dim[InC],
        out_channels: _Dim[OutC],
        kernel_size: _Dim[K],
        stride: _Dim[S] = 1,
        padding: _Dim[P] = 0,
        output_padding: _Dim[OP] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Dim[D] = 1,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B, L](
        self, input: Tensor[B, InC, L]
    ) -> Tensor[B, OutC, (L - 1) * S - 2 * P + D * (K - 1) + OP + 1]: ...

class ConvTranspose2d[InC, OutC, K, S = 1, P = 0, OP = 0, D = 1](Module):
    """2D transposed convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, OP, D are bound from constructor arguments via _Dim[T].
    PEP 696 defaults apply when arguments are omitted.
    """

    weight: Tensor[InC, OutC, K, K]

    def __init__(
        self,
        in_channels: _Dim[InC],
        out_channels: _Dim[OutC],
        kernel_size: _Dim[K],
        stride: _Dim[S] = 1,
        padding: _Dim[P] = 0,
        output_padding: _Dim[OP] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Dim[D] = 1,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B, H, W](
        self, input: Tensor[B, InC, H, W]
    ) -> Tensor[
        B,
        OutC,
        (H - 1) * S - 2 * P + D * (K - 1) + OP + 1,
        (W - 1) * S - 2 * P + D * (K - 1) + OP + 1,
    ]: ...

class ConvTranspose3d[InC, OutC, K, S = 1, P = 0, OP = 0, D = 1](Module):
    """3D transposed convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, OP, D are bound from constructor arguments via _Dim[T].
    PEP 696 defaults apply when arguments are omitted.
    """

    weight: Tensor[InC, OutC, K, K, K]

    def __init__(
        self,
        in_channels: _Dim[InC],
        out_channels: _Dim[OutC],
        kernel_size: _Dim[K],
        stride: _Dim[S] = 1,
        padding: _Dim[P] = 0,
        output_padding: _Dim[OP] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Dim[D] = 1,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B, D_, H, W](
        self, input: Tensor[B, InC, D_, H, W]
    ) -> Tensor[
        B,
        OutC,
        (D_ - 1) * S - 2 * P + D * (K - 1) + OP + 1,
        (H - 1) * S - 2 * P + D * (K - 1) + OP + 1,
        (W - 1) * S - 2 * P + D * (K - 1) + OP + 1,
    ]: ...

# ==============================================================================
# Pooling Modules
# ==============================================================================

class MaxPool1d(Module):
    """1D max pooling. Shape inference via DSL + NNModule init capture."""
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class MaxPool2d(Module):
    """2D max pooling. Shape inference via DSL + NNModule init capture."""
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class MaxPool3d(Module):
    """3D max pooling. Shape inference via DSL + NNModule init capture."""
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AvgPool1d(Module):
    """1D average pooling. Shape inference via DSL + NNModule init capture."""
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AvgPool2d(Module):
    """2D average pooling. Shape inference via DSL + NNModule init capture."""
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AvgPool3d(Module):
    """3D average pooling. Shape inference via DSL + NNModule init capture."""
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveAvgPool1d[OL](Module):
    """1D adaptive average pooling"""
    def __init__(self, output_size: _Dim[OL]) -> None: ...
    def forward[B, C](self, input: Tensor[B, C, Any]) -> Tensor[B, C, OL]: ...

class AdaptiveAvgPool2d[OH, OW](Module):
    """2D adaptive average pooling"""
    def __init__(self, output_size: tuple[_Dim[OH], _Dim[OW]]) -> None: ...
    def forward[B, C](self, input: Tensor[B, C, Any, Any]) -> Tensor[B, C, OH, OW]: ...

class AdaptiveAvgPool3d[OD, OH, OW](Module):
    """3D adaptive average pooling"""
    def __init__(self, output_size: tuple[_Dim[OD], _Dim[OH], _Dim[OW]]) -> None: ...
    def forward[B, C](
        self, input: Tensor[B, C, Any, Any, Any]
    ) -> Tensor[B, C, OD, OH, OW]: ...

class AdaptiveMaxPool1d[OL](Module):
    """1D adaptive max pooling"""
    def __init__(self, output_size: _Dim[OL], return_indices: bool = False) -> None: ...
    def forward[B, C](self, input: Tensor[B, C, Any]) -> Tensor[B, C, OL]: ...

class AdaptiveMaxPool2d[OH, OW](Module):
    """2D adaptive max pooling"""
    def __init__(
        self, output_size: tuple[_Dim[OH], _Dim[OW]], return_indices: bool = False
    ) -> None: ...
    def forward[B, C](self, input: Tensor[B, C, Any, Any]) -> Tensor[B, C, OH, OW]: ...

class AdaptiveMaxPool3d[OD, OH, OW](Module):
    """3D adaptive max pooling"""
    def __init__(
        self,
        output_size: tuple[_Dim[OD], _Dim[OH], _Dim[OW]],
        return_indices: bool = False,
    ) -> None: ...
    def forward[B, C](
        self, input: Tensor[B, C, Any, Any, Any]
    ) -> Tensor[B, C, OD, OH, OW]: ...

# ==============================================================================
# Upsampling / Rearrangement Modules
# ==============================================================================

class PixelShuffle(Module):
    """Rearranges channels into spatial dimensions.

    [B, C * r * r, H, W] → [B, C, H * r, W * r]

    Shape inference via DSL + NNModule init capture.
    """

    def __init__(self, upscale_factor: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class GLU(Module):
    """Gated Linear Unit: splits input along dim, applies sigmoid gating.

    GLU(x) = x1 * sigmoid(x2) where x1, x2 = x.split(x.size(dim) // 2, dim)
    Output is same as input except dimension `dim` is halved.

    Shape inference via DSL + NNModule init capture.
    """

    def __init__(self, dim: int = 1) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class LSTM(Module):
    """Long Short-Term Memory RNN.

    Input:  Tensor[B, T, InputSize]  (batch_first=True assumed)
    Output: (Tensor[B, T, HiddenSize * ND],
             Tensor[NL * ND, B, HiddenSize],
             Tensor[NL * ND, B, HiddenSize])

    ND (num_directions) = 1 for unidirectional, 2 for bidirectional.

    Shape inference via DSL + NNModule init capture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None: ...
    def flatten_parameters(self) -> None:
        """Reset parameter data pointer for CUDA contiguous memory. No-op on CPU."""
        ...
    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...

class LSTMCell(Module):
    """Long Short-Term Memory cell.

    Input:  Tensor[B, InputSize]
    Output: (Tensor[B, HiddenSize], Tensor[B, HiddenSize])

    Shape inference via DSL + NNModule init capture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]: ...

class GRUCell(Module):
    """Gated Recurrent Unit cell.

    Input:  Tensor[B, InputSize]
    Output: Tensor[B, HiddenSize]

    Shape-preserving when InputSize == HiddenSize; otherwise returns
    unrefined Tensor (no DSL registration).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor: ...

class Upsample(Module):
    """Upsamples input. Shape inference via DSL + NNModule init capture.

    Supports size (target spatial dims) or scale_factor (int multiplier).
    Float scale_factor not yet supported in DSL.
    """

    def __init__(
        self,
        size: int | None = None,
        scale_factor: int | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# ==============================================================================
# Loss Modules
# ==============================================================================

class CrossEntropyLoss(Module):
    """Cross entropy loss"""
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class MSELoss(Module):
    """Mean squared error loss"""
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class L1Loss(Module):
    """L1 (mean absolute error) loss"""
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class NLLLoss(Module):
    """Negative log likelihood loss"""
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class BCELoss(Module):
    """Binary cross entropy loss"""
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class BCEWithLogitsLoss(Module):
    """Binary cross entropy with logits loss"""
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
    ) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class SmoothL1Loss(Module):
    """Smooth L1 loss"""
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        beta: float = 1.0,
    ) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class HuberLoss(Module):
    """Huber loss"""
    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class KLDivLoss(Module):
    """KL divergence loss"""
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

class CTCLoss(Module):
    """Connectionist Temporal Classification loss"""
    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
    ) -> None: ...
    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor: ...

# ==============================================================================
# Misc Modules
# ==============================================================================

class Flatten(Module):
    """Flattens a contiguous range of dims.

    Shape inference via DSL + NNModule init capture.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Unflatten(Module):
    """Unflattens a dimension"""
    def __init__(self, dim: int | str, unflattened_size: tuple[int, ...]) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ReflectionPad2d(Module):
    """Pads input using reflection of the input boundary.

    Shape inference via DSL + NNModule init capture.
    """

    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ReplicationPad2d(Module):
    """Pads input using replication of the input boundary.

    Shape inference via DSL + NNModule init capture.
    """

    def __init__(self, padding: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# Embedding variants
class EmbeddingBag[NUM_EMB, EMB_DIM](Module):
    """Computes sums or means of 'bags' of embeddings.

    Unlike Embedding, EmbeddingBag aggregates over variable-length groups
    of indices using offsets. Output batch dimension comes from offsets.
    """

    weight: Tensor[NUM_EMB, EMB_DIM]

    def __init__(
        self,
        num_embeddings: _Dim[NUM_EMB],
        embedding_dim: _Dim[EMB_DIM],
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

    # EmbeddingBag forward: batch dim B comes from offsets (default, include_last_offset=False).
    # Embedding dim EMB_DIM is always preserved from init.
    def forward[B](
        self,
        input: Tensor,
        offsets: Tensor[B] | None = None,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor[B, EMB_DIM]: ...

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
    # Activation modules
    "ReLU",
    "ReLU6",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "Mish",
    "Hardswish",
    "Hardsigmoid",
    "LeakyReLU",
    "ELU",
    "SELU",
    "CELU",
    "Softplus",
    "PReLU",
    "Threshold",
    "Softmax",
    "LogSoftmax",
    # Normalization modules
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    # Dropout modules
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
    # Other
    "Identity",
    # Convolution modules
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    # Pooling modules
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    # Loss modules
    "CrossEntropyLoss",
    "MSELoss",
    "L1Loss",
    "NLLLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "KLDivLoss",
    "CTCLoss",
    # RNN cells
    "LSTM",
    "LSTMCell",
    "GRUCell",
    # Misc modules
    "Flatten",
    "Unflatten",
    "ReflectionPad2d",
    "ReplicationPad2d",
    "EmbeddingBag",
    "Upsample",
]
