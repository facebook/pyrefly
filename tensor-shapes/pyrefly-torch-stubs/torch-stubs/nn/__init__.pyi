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

from shape_extensions import Elements, Flag, IntTuple, IntVar

if TYPE_CHECKING:
    from shape_extensions import Int as _Int, ProxyMethod
    from torch import Tensor
    from torch._shapes import (
        flatten_shape,
        glu_shape,
        interpolate_scalar_shape,
        lstm_cell_state_shape,
        pixel_shuffle_shape,
        pool_shape,
        recurrent_output_shape,
        recurrent_state_shape,
        symmetric_pad2d_shape,
    )

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
    def __getattr__(self, name: str) -> Any: ...
    __call__: ProxyMethod["forward"]
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None: ...
    def register_parameter(self, name: str, param: Parameter | None) -> None: ...
    def apply(self, fn: Callable[[Module], None]) -> Self: ...
    def to(self, *args: Any, **kwargs: Any) -> Self: ...
    def train(self, mode: bool = True) -> Self: ...
    def eval(self) -> Self: ...
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
def Parameter[Shape: IntTuple](
    data: Tensor[Shape], requires_grad: bool = True
) -> Tensor[Shape]:
    """
    Wraps a tensor as a module parameter.
    Returns the tensor (for type purposes) since operations on Parameters return Tensors.
    """
    ...

# Buffer wrapper
# Similar to Parameter, Buffer wraps a tensor that is not a parameter but should be
# part of the module's state_dict. For type checking we model it as returning Tensor.
def Buffer[Shape: IntTuple](
    data: Tensor[Shape], persistent: bool = True
) -> Tensor[Shape]:
    """
    Wraps a tensor as a module buffer.
    Returns the tensor (for type purposes) since operations on Buffers return Tensors.
    """
    ...

# Linear layer
class Linear[IN: IntVar, OUT: IntVar](Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b"""

    weight: Tensor[[OUT, IN]]
    bias: Tensor[[OUT]] | None

    def __init__(
        self,
        in_features: _Int[IN],
        out_features: _Int[OUT],
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[Bs: IntTuple](
        self, input: Tensor[[*Elements[Bs], IN]]
    ) -> Tensor[[*Elements[Bs], OUT]]: ...

# Dropout
class Dropout(Module):
    """During training, randomly zeroes some of the elements of the input tensor with probability p"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[Shape: IntTuple](self, input: Tensor[Shape]) -> Tensor[Shape]: ...

# GELU activation
class GELU(Module):
    """Applies the Gaussian Error Linear Units function"""
    def __init__(self, approximate: str = "none") -> None: ...
    def forward[Shape: IntTuple](self, input: Tensor[Shape]) -> Tensor[Shape]: ...

# Embedding
class Embedding[NUM_EMB: IntVar, EMB_DIM: IntVar](Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size"""

    weight: Tensor[[NUM_EMB, EMB_DIM]]

    def __init__(
        self,
        num_embeddings: _Int[NUM_EMB],
        embedding_dim: _Int[EMB_DIM],
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
    def forward[T: IntVar](self, input: Tensor[[T]]) -> Tensor[[T, EMB_DIM]]: ...

    # 2D input: [B, T] -> [B, T, EMB_DIM]
    @overload
    def forward[B: IntVar, T: IntVar](
        self, input: Tensor[[B, T]]
    ) -> Tensor[[B, T, EMB_DIM]]: ...

# ModuleDict
class ModuleDict[T](Module):
    """Holds submodules in a dictionary"""
    def __init__(self, modules: T) -> None: ...
    def __getitem__(self, key: str) -> T: ...
    def __setitem__(self, key: str, module: Module) -> None: ...
    def __getattr__(self, name: str) -> T: ...  # Support attribute access
    def __iter__(self) -> Iterator[str]: ...
    def keys(self) -> Iterator[str]: ...
    def items(self) -> Iterator[tuple[str, T]]: ...
    def values(self) -> Iterator[T]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class ReLU6(Module):
    """Applies ReLU6 activation (clamps to [0, 6])"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class SiLU(Module):
    """Applies SiLU (Swish) activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Sigmoid(Module):
    """Applies element-wise Sigmoid"""
    def __init__(self) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Tanh(Module):
    """Applies element-wise Tanh"""
    def __init__(self) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Mish(Module):
    """Applies Mish activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Hardswish(Module):
    """Applies Hardswish activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Hardsigmoid(Module):
    """Applies Hardsigmoid activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class LeakyReLU(Module):
    """Applies LeakyReLU activation"""
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class ELU(Module):
    """Applies ELU activation"""
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class SELU(Module):
    """Applies SELU activation"""
    def __init__(self, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class CELU(Module):
    """Applies CELU activation"""
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Softplus(Module):
    """Applies Softplus activation"""
    def __init__(self, beta: float = 1, threshold: float = 20) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class PReLU(Module):
    """Applies PReLU activation"""
    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Threshold(Module):
    """Applies Threshold activation"""
    def __init__(
        self, threshold: float, value: float, inplace: bool = False
    ) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Softmax(Module):
    """Applies Softmax along a dimension"""
    def __init__(self, dim: int | None = None) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class LogSoftmax(Module):
    """Applies LogSoftmax along a dimension"""
    def __init__(self, dim: int | None = None) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class GroupNorm(Module):
    """Applies Group Normalization"""

    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

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
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

# ==============================================================================
# Dropout Modules (shape-preserving)
# ==============================================================================

class Dropout1d(Module):
    """Randomly zero out entire channels (1D)"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Dropout2d(Module):
    """Randomly zero out entire channels (2D)"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class Dropout3d(Module):
    """Randomly zero out entire channels (3D)"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class AlphaDropout(Module):
    """Applies Alpha Dropout for SELU networks"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

class FeatureAlphaDropout(Module):
    """Randomly masks entire channels with Alpha Dropout"""
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

# ==============================================================================
# Other Shape-Preserving Modules
# ==============================================================================

class Identity(Module):
    """Identity module that returns the input unchanged"""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]: ...

# ==============================================================================
# Convolution Modules
# ==============================================================================

class Conv1d[
    InC: IntVar,
    OutC: IntVar,
    K: IntVar,
    S: IntVar = 1,
    P: IntVar = 0,
    D: IntVar = 1,
](Module):
    """1D convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, D are bound from constructor arguments via _Int[T].
    PEP 696 defaults (S=1, P=0, D=1) apply when arguments are omitted.
    """

    weight: Tensor[[OutC, InC, K]]

    def __init__(
        self,
        in_channels: _Int[InC],
        out_channels: _Int[OutC],
        kernel_size: _Int[K],
        stride: _Int[S] = 1,
        padding: _Int[P] = 0,
        dilation: _Int[D] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B: IntVar, L: IntVar](
        self, input: Tensor[[B, InC, L]]
    ) -> Tensor[[B, OutC, (L + 2 * P - D * (K - 1) - 1) // S + 1]]: ...

class Conv2d[
    InC: IntVar,
    OutC: IntVar,
    K: IntVar,
    S: IntVar = 1,
    P: IntVar = 0,
    D: IntVar = 1,
](Module):
    """2D convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, D are bound from constructor arguments via _Int[T].
    PEP 696 defaults (S=1, P=0, D=1) apply when arguments are omitted.

    kernel_size, stride, padding, and dilation also accept tuple[int, int]
    for per-axis values.  When a tuple is passed the corresponding type
    parameter is unbound and the spatial formula preserves arithmetic around
    that unknown dimension.  Proper per-axis tracking would require DSL-based
    inference, but nn.Sequential currently dispatches via stub signatures, not
    DSL.
    """

    weight: Tensor[[OutC, InC, K, K]]
    bias: Tensor[[OutC]] | None

    def __init__(
        self,
        in_channels: _Int[InC],
        out_channels: _Int[OutC],
        kernel_size: _Int[K] | tuple[int, int],
        stride: _Int[S] | tuple[int, int] = 1,
        padding: _Int[P] | tuple[int, int] | str = 0,
        dilation: _Int[D] | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, input: Tensor[[B, InC, H, W]]
    ) -> Tensor[
        [
            B,
            OutC,
            (H + 2 * P - D * (K - 1) - 1) // S + 1,
            (W + 2 * P - D * (K - 1) - 1) // S + 1,
        ]
    ]: ...

class Conv3d[
    InC: IntVar,
    OutC: IntVar,
    K: IntVar,
    S: IntVar = 1,
    P: IntVar = 0,
    D: IntVar = 1,
](Module):
    """3D convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, D are bound from constructor arguments via _Int[T].
    PEP 696 defaults (S=1, P=0, D=1) apply when arguments are omitted.
    """

    weight: Tensor[[OutC, InC, K, K, K]]

    def __init__(
        self,
        in_channels: _Int[InC],
        out_channels: _Int[OutC],
        kernel_size: _Int[K],
        stride: _Int[S] = 1,
        padding: _Int[P] = 0,
        dilation: _Int[D] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B: IntVar, D_: IntVar, H: IntVar, W: IntVar](
        self, input: Tensor[[B, InC, D_, H, W]]
    ) -> Tensor[
        [
            B,
            OutC,
            (D_ + 2 * P - D * (K - 1) - 1) // S + 1,
            (H + 2 * P - D * (K - 1) - 1) // S + 1,
            (W + 2 * P - D * (K - 1) - 1) // S + 1,
        ]
    ]: ...

class ConvTranspose1d[
    InC: IntVar,
    OutC: IntVar,
    K: IntVar,
    S: IntVar = 1,
    P: IntVar = 0,
    OP: IntVar = 0,
    D: IntVar = 1,
](Module):
    """1D transposed convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, OP, D are bound from constructor arguments via _Int[T].
    PEP 696 defaults apply when arguments are omitted.
    """

    weight: Tensor[[InC, OutC, K]]

    def __init__(
        self,
        in_channels: _Int[InC],
        out_channels: _Int[OutC],
        kernel_size: _Int[K],
        stride: _Int[S] = 1,
        padding: _Int[P] = 0,
        output_padding: _Int[OP] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Int[D] = 1,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B: IntVar, L: IntVar](
        self, input: Tensor[[B, InC, L]]
    ) -> Tensor[[B, OutC, (L - 1) * S - 2 * P + D * (K - 1) + OP + 1]]: ...

class ConvTranspose2d[
    InC: IntVar,
    OutC: IntVar,
    K: IntVar,
    S: IntVar = 1,
    P: IntVar = 0,
    OP: IntVar = 0,
    D: IntVar = 1,
](Module):
    """2D transposed convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, OP, D are bound from constructor arguments via _Int[T].
    PEP 696 defaults apply when arguments are omitted.
    """

    weight: Tensor[[InC, OutC, K, K]]

    def __init__(
        self,
        in_channels: _Int[InC],
        out_channels: _Int[OutC],
        kernel_size: _Int[K],
        stride: _Int[S] = 1,
        padding: _Int[P] = 0,
        output_padding: _Int[OP] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Int[D] = 1,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, input: Tensor[[B, InC, H, W]]
    ) -> Tensor[
        [
            B,
            OutC,
            (H - 1) * S - 2 * P + D * (K - 1) + OP + 1,
            (W - 1) * S - 2 * P + D * (K - 1) + OP + 1,
        ]
    ]: ...

class ConvTranspose3d[
    InC: IntVar,
    OutC: IntVar,
    K: IntVar,
    S: IntVar = 1,
    P: IntVar = 0,
    OP: IntVar = 0,
    D: IntVar = 1,
](Module):
    """3D transposed convolution. Tracks channel and spatial dimensions.

    Type parameters S, P, OP, D are bound from constructor arguments via _Int[T].
    PEP 696 defaults apply when arguments are omitted.
    """

    weight: Tensor[[InC, OutC, K, K, K]]

    def __init__(
        self,
        in_channels: _Int[InC],
        out_channels: _Int[OutC],
        kernel_size: _Int[K],
        stride: _Int[S] = 1,
        padding: _Int[P] = 0,
        output_padding: _Int[OP] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Int[D] = 1,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[B: IntVar, D_: IntVar, H: IntVar, W: IntVar](
        self, input: Tensor[[B, InC, D_, H, W]]
    ) -> Tensor[
        [
            B,
            OutC,
            (D_ - 1) * S - 2 * P + D * (K - 1) + OP + 1,
            (H - 1) * S - 2 * P + D * (K - 1) + OP + 1,
            (W - 1) * S - 2 * P + D * (K - 1) + OP + 1,
        ]
    ]: ...

# ==============================================================================
# Pooling Modules
# ==============================================================================

class MaxPool1d[
    KernelSize: Flag[int],
    Stride: Flag[int | None] = None,
    Padding: Flag[int] = 0,
    Dilation: Flag[int] = 1,
    CeilMode: Flag[bool] = False,
](Module):
    """1D max pooling with scalar controls tracked by the type-level DSL."""
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Stride = None,
        padding: Padding = 0,
        dilation: Dilation = 1,
        return_indices: bool = False,
        ceil_mode: CeilMode = False,
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[
        pool_shape(Shape, 1, KernelSize, Stride, Padding, Dilation, CeilMode)
    ]: ...

class MaxPool2d[
    KernelSize: Flag[int],
    Stride: Flag[int | None] = None,
    Padding: Flag[int] = 0,
    Dilation: Flag[int] = 1,
    CeilMode: Flag[bool] = False,
](Module):
    """2D max pooling with scalar controls tracked by the type-level DSL."""
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Stride = None,
        padding: Padding = 0,
        dilation: Dilation = 1,
        return_indices: bool = False,
        ceil_mode: CeilMode = False,
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[
        pool_shape(Shape, 2, KernelSize, Stride, Padding, Dilation, CeilMode)
    ]: ...

class MaxPool3d[
    KernelSize: Flag[int],
    Stride: Flag[int | None] = None,
    Padding: Flag[int] = 0,
    Dilation: Flag[int] = 1,
    CeilMode: Flag[bool] = False,
](Module):
    """3D max pooling with scalar controls tracked by the type-level DSL."""
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Stride = None,
        padding: Padding = 0,
        dilation: Dilation = 1,
        return_indices: bool = False,
        ceil_mode: CeilMode = False,
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[
        pool_shape(Shape, 3, KernelSize, Stride, Padding, Dilation, CeilMode)
    ]: ...

class AvgPool1d[
    KernelSize: Flag[int],
    Stride: Flag[int | None] = None,
    Padding: Flag[int] = 0,
    CeilMode: Flag[bool] = False,
](Module):
    """1D average pooling with scalar controls tracked by the type-level DSL."""
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Stride = None,
        padding: Padding = 0,
        ceil_mode: CeilMode = False,
        count_include_pad: bool = True,
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[pool_shape(Shape, 1, KernelSize, Stride, Padding, 1, CeilMode)]: ...

class AvgPool2d[
    KernelSize: Flag[int],
    Stride: Flag[int | None] = None,
    Padding: Flag[int] = 0,
    CeilMode: Flag[bool] = False,
](Module):
    """2D average pooling with scalar controls tracked by the type-level DSL."""
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Stride = None,
        padding: Padding = 0,
        ceil_mode: CeilMode = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[pool_shape(Shape, 2, KernelSize, Stride, Padding, 1, CeilMode)]: ...

class AvgPool3d[
    KernelSize: Flag[int],
    Stride: Flag[int | None] = None,
    Padding: Flag[int] = 0,
    CeilMode: Flag[bool] = False,
](Module):
    """3D average pooling with scalar controls tracked by the type-level DSL."""
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Stride = None,
        padding: Padding = 0,
        ceil_mode: CeilMode = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[pool_shape(Shape, 3, KernelSize, Stride, Padding, 1, CeilMode)]: ...

class AdaptiveAvgPool1d[OL: IntVar](Module):
    """1D adaptive average pooling"""
    def __init__(self, output_size: _Int[OL]) -> None: ...
    def forward[B: IntVar, C: IntVar](
        self, input: Tensor[[B, C, Any]]
    ) -> Tensor[[B, C, OL]]: ...

class AdaptiveAvgPool2d[OH: IntVar, OW: IntVar](Module):
    """2D adaptive average pooling"""
    def __init__(self, output_size: tuple[_Int[OH], _Int[OW]]) -> None: ...
    def forward[B: IntVar, C: IntVar](
        self, input: Tensor[[B, C, Any, Any]]
    ) -> Tensor[[B, C, OH, OW]]: ...

class AdaptiveAvgPool3d[OD: IntVar, OH: IntVar, OW: IntVar](Module):
    """3D adaptive average pooling"""
    def __init__(self, output_size: tuple[_Int[OD], _Int[OH], _Int[OW]]) -> None: ...
    def forward[B: IntVar, C: IntVar](
        self, input: Tensor[[B, C, Any, Any, Any]]
    ) -> Tensor[[B, C, OD, OH, OW]]: ...

class AdaptiveMaxPool1d[OL: IntVar](Module):
    """1D adaptive max pooling"""
    def __init__(self, output_size: _Int[OL], return_indices: bool = False) -> None: ...
    def forward[B: IntVar, C: IntVar](
        self, input: Tensor[[B, C, Any]]
    ) -> Tensor[[B, C, OL]]: ...

class AdaptiveMaxPool2d[OH: IntVar, OW: IntVar](Module):
    """2D adaptive max pooling"""
    def __init__(
        self, output_size: tuple[_Int[OH], _Int[OW]], return_indices: bool = False
    ) -> None: ...
    def forward[B: IntVar, C: IntVar](
        self, input: Tensor[[B, C, Any, Any]]
    ) -> Tensor[[B, C, OH, OW]]: ...

class AdaptiveMaxPool3d[OD: IntVar, OH: IntVar, OW: IntVar](Module):
    """3D adaptive max pooling"""
    def __init__(
        self,
        output_size: tuple[_Int[OD], _Int[OH], _Int[OW]],
        return_indices: bool = False,
    ) -> None: ...
    def forward[B: IntVar, C: IntVar](
        self, input: Tensor[[B, C, Any, Any, Any]]
    ) -> Tensor[[B, C, OD, OH, OW]]: ...

# ==============================================================================
# Upsampling / Rearrangement Modules
# ==============================================================================

class PixelShuffle[UpscaleFactor: _Int](Module):
    """Rearranges channels into spatial dimensions.

    [B, C * r * r, H, W] → [B, C, H * r, W * r]

    Shape inference via type-level DSL.
    """

    def __init__(self, upscale_factor: UpscaleFactor) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[pixel_shuffle_shape(Shape, UpscaleFactor)]: ...

class GLU[Dim: Flag[int]](Module):
    """Gated Linear Unit: splits input along dim, applies sigmoid gating.

    GLU(x) = x1 * sigmoid(x2) where x1, x2 = x.split(x.size(dim) // 2, dim)
    Output is same as input except dimension `dim` is halved.

    Shape inference via type-level DSL.
    """

    def __init__(self, dim: Dim = -1) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[glu_shape(Shape, Dim)]: ...

class LSTM[
    InputSize: _Int,
    HiddenSize: _Int,
    NumLayers: _Int = 1,
    Bidirectional: Flag[bool] = False,
](Module):
    """Long Short-Term Memory RNN.

    Input:  Tensor[[B, T, InputSize]]  (batch_first=True assumed)
    Output: (Tensor[[B, T, HiddenSize * ND]],
             Tensor[[NL * ND, B, HiddenSize]],
             Tensor[[NL * ND, B, HiddenSize]])

    ND (num_directions) = 1 for unidirectional, 2 for bidirectional.

    Shape inference via the type-level DSL and class type parameters.
    """

    def __init__(
        self,
        input_size: InputSize,
        hidden_size: HiddenSize,
        num_layers: NumLayers = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: Bidirectional = False,
    ) -> None: ...
    def flatten_parameters(self) -> None:
        """Reset parameter data pointer for CUDA contiguous memory. No-op on CPU."""
        ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> tuple[
        Tensor[recurrent_output_shape(Shape, HiddenSize, Bidirectional)],
        Tensor[recurrent_state_shape(Shape, HiddenSize, NumLayers, Bidirectional)],
        Tensor[recurrent_state_shape(Shape, HiddenSize, NumLayers, Bidirectional)],
    ]: ...

class LSTMCell[InputSize: _Int, HiddenSize: _Int](Module):
    """Long Short-Term Memory cell.

    Input:  Tensor[[B, InputSize]]
    Output: (Tensor[[B, HiddenSize]], Tensor[[B, HiddenSize]])

    Shape inference via the type-level DSL and class type parameters.
    """

    def __init__(
        self,
        input_size: InputSize,
        hidden_size: HiddenSize,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape], hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[
        Tensor[lstm_cell_state_shape(Shape, HiddenSize)],
        Tensor[lstm_cell_state_shape(Shape, HiddenSize)],
    ]: ...

class GRU[
    InputSize: _Int,
    HiddenSize: _Int,
    NumLayers: _Int = 1,
    Bidirectional: Flag[bool] = False,
](Module):
    """Gated Recurrent Unit RNN.

    Input:  Tensor[[B, T, InputSize]]  (batch_first=True assumed)
    Output: (Tensor[[B, T, HiddenSize * ND]],
             Tensor[[NL * ND, B, HiddenSize]])

    ND (num_directions) = 1 for unidirectional, 2 for bidirectional.

    Shape inference via the type-level DSL and class type parameters.
    """

    def __init__(
        self,
        input_size: InputSize,
        hidden_size: HiddenSize,
        num_layers: NumLayers = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: Bidirectional = False,
    ) -> None: ...
    def flatten_parameters(self) -> None:
        """Reset parameter data pointer for CUDA contiguous memory. No-op on CPU."""
        ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape], hx: Tensor | None = None
    ) -> tuple[
        Tensor[recurrent_output_shape(Shape, HiddenSize, Bidirectional)],
        Tensor[recurrent_state_shape(Shape, HiddenSize, NumLayers, Bidirectional)],
    ]: ...

class GRUCell(Module):
    """Gated Recurrent Unit cell.

    Input:  Tensor[[B, InputSize]]
    Output: Tensor[[B, HiddenSize]]

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

class Upsample[
    Size: _Int | None,
    Scale: _Int | None,
    TupleSize: tuple[int, ...] | None = None,
    FloatScale: float | tuple[float, ...] | None = None,
](Module):
    """Upsamples input with scalar integer arguments tracked by the V2 DSL.

    Literal preservation only applies to parameters bound by the `Int` special
    form, so the valid-but-gradual tuple and float arguments get their own
    parameters. A non-`None` `TupleSize`/`FloatScale` is what steers `forward`
    away from the precise arm.

    The scalar arguments deliberately have no type-parameter default: a bare
    `nn.Upsample` annotation names an instance whose arguments are unknown, and
    defaulting them to `None` would make it indistinguishable from `Upsample()`,
    whose omitted arguments really do bind `None` and are an error. The
    tuple/float parameters keep their defaults because they only ever steer
    `forward` away from the precise arm.
    """

    @overload
    def __init__(
        self,
        size: Size = None,
        scale_factor: Scale = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        size: TupleSize,
        scale_factor: None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        size: None = None,
        scale_factor: FloatScale = ...,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> None: ...
    @overload
    def forward[Shape: IntTuple, S: _Int | None, F: _Int | None](
        self: Upsample[S, F, None, None], input: Tensor[Shape]
    ) -> Tensor[interpolate_scalar_shape(Shape, S, F)]: ...
    @overload
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

class ParameterList[T](Module):
    """Holds parameters in a list."""
    def __init__(self, parameters: Iterable[T] | None = None) -> None: ...
    def __getitem__(self, idx: int) -> T: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...

class LazyLinear[OUT: IntVar](Module):
    """Linear layer with lazy in_features initialization.

    out_features is known at construction; in_features is inferred at first forward.
    """

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        out_features: _Int[OUT],
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...
    def forward[Bs: IntTuple](
        self, input: Tensor[[*Elements[Bs], Any]]
    ) -> Tensor[[*Elements[Bs], OUT]]: ...

class Flatten[StartDim: Flag[int], EndDim: Flag[int]](Module):
    """Flattens a contiguous range of dims.

    Shape inference via type-level DSL.
    """

    def __init__(self, start_dim: StartDim = 1, end_dim: EndDim = -1) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[flatten_shape(Shape, StartDim, EndDim)]: ...

class Unflatten(Module):
    """Unflattens a dimension"""
    def __init__(self, dim: int | str, unflattened_size: tuple[int, ...]) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class ReflectionPad2d[Padding: Flag[int]](Module):
    """Pads input using reflection of the input boundary.

    Shape inference via type-level DSL.
    """

    def __init__(self, padding: Padding) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[symmetric_pad2d_shape(Shape, Padding)]: ...

class ReplicationPad2d[Padding: Flag[int]](Module):
    """Pads input using replication of the input boundary.

    Shape inference via type-level DSL.
    """

    def __init__(self, padding: Padding) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[symmetric_pad2d_shape(Shape, Padding)]: ...

# Embedding variants
class EmbeddingBag[NUM_EMB: IntVar, EMB_DIM: IntVar](Module):
    """Computes sums or means of 'bags' of embeddings.

    Unlike Embedding, EmbeddingBag aggregates over variable-length groups
    of indices using offsets. Output batch dimension comes from offsets.
    """

    weight: Tensor[[NUM_EMB, EMB_DIM]]

    def __init__(
        self,
        num_embeddings: _Int[NUM_EMB],
        embedding_dim: _Int[EMB_DIM],
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
    def forward[B: IntVar](
        self,
        input: Tensor,
        offsets: Tensor[[B]] | None = None,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor[[B, EMB_DIM]]: ...

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
    "GRU",
    "GRUCell",
    # Misc modules
    "ParameterList",
    "LazyLinear",
    "Flatten",
    "Unflatten",
    "ReflectionPad2d",
    "ReplicationPad2d",
    "EmbeddingBag",
    "Upsample",
]
