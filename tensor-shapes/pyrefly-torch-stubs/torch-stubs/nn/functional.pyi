# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Type stubs for torch.nn.functional module.
Functional neural network operations including convolution, pooling, activation, and normalization.
"""

import builtins
from typing import Literal, overload

import shape_extensions
from shape_extensions import (
    Elements,
    Flag,
    Int as _Int,
    IntTuple,
    IntVar,
    uses_shape_dsl,
)
from torch._shapes import (
    adaptive_pool1d_shape,
    adaptive_pool2d_shape,
    adaptive_pool3d_shape,
    adaptive_pool_gradual_shape,
    classification_loss_shape,
    conv_shape,
    conv_transpose_shape,
    cosine_embedding_score_shape,
    cosine_similarity_shape,
    interpolate_ir,
    kl_div_loss_shape,
    loss_shape,
    pad_shape,
    pairwise_distance_shape,
    pool_ir,
)

from .. import Tensor

__all__ = [
    # Convolution
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    # Pooling
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    # Adaptive pooling
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    # Interpolation
    "interpolate",
    "upsample",
    # Activation functions
    "relu",
    "gelu",
    "silu",
    "selu",
    "elu",
    "leaky_relu",
    "relu6",
    "softplus",
    "softsign",
    "hardtanh",
    "hardsigmoid",
    "hardswish",
    "sigmoid",
    "tanh",
    "mish",
    "glu",
    "prelu",
    "rrelu",
    "celu",
    "threshold",
    "tanhshrink",
    "softshrink",
    "hardshrink",
    "logsigmoid",
    "softmax",
    "log_softmax",
    "softmin",
    # Linear
    "linear",
    # Embedding
    "embedding",
    # Normalization
    "batch_norm",
    "instance_norm",
    "layer_norm",
    "group_norm",
    "rms_norm",
    "normalize",
    "local_response_norm",
    # Dropout
    "dropout",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "alpha_dropout",
    "feature_alpha_dropout",
    # Attention
    "scaled_dot_product_attention",
]

# ====================================================================
# Phase 3: Convolution & Pooling Operations
# ====================================================================

# Convolution operations
def conv1d[
    InputShape: IntTuple,
    WeightShape: IntTuple,
    Stride: Flag[builtins.int | tuple[builtins.int]],
    Padding: Flag[builtins.int | tuple[builtins.int]],
    Dilation: Flag[builtins.int | tuple[builtins.int]],
](
    self: Tensor[InputShape],
    weight: Tensor[WeightShape],
    bias: Tensor | None = None,
    stride: Stride = 1,
    padding: Padding = 0,
    dilation: Dilation = 1,
    groups: int = 1,
) -> Tensor[conv_shape(InputShape, WeightShape, Stride, Padding, Dilation)]:
    """1D convolution. Shape inference via meta-shape: torch.nn.functional.conv1d"""
    ...

def conv2d[
    InputShape: IntTuple,
    WeightShape: IntTuple,
    Stride: Flag[builtins.int | tuple[builtins.int, builtins.int]],
    Padding: Flag[builtins.int | tuple[builtins.int, builtins.int]],
    Dilation: Flag[builtins.int | tuple[builtins.int, builtins.int]],
](
    self: Tensor[InputShape],
    weight: Tensor[WeightShape],
    bias: Tensor | None = None,
    stride: Stride = 1,
    padding: Padding = 0,
    dilation: Dilation = 1,
    groups: int = 1,
) -> Tensor[conv_shape(InputShape, WeightShape, Stride, Padding, Dilation)]:
    """2D convolution. Shape inference via meta-shape: torch.nn.functional.conv2d"""
    ...

def conv3d[
    InputShape: IntTuple,
    WeightShape: IntTuple,
    Stride: Flag[builtins.int | tuple[builtins.int, builtins.int, builtins.int]],
    Padding: Flag[builtins.int | tuple[builtins.int, builtins.int, builtins.int]],
    Dilation: Flag[builtins.int | tuple[builtins.int, builtins.int, builtins.int]],
](
    self: Tensor[InputShape],
    weight: Tensor[WeightShape],
    bias: Tensor | None = None,
    stride: Stride = 1,
    padding: Padding = 0,
    dilation: Dilation = 1,
    groups: int = 1,
) -> Tensor[conv_shape(InputShape, WeightShape, Stride, Padding, Dilation)]:
    """3D convolution. Shape inference via meta-shape: torch.nn.functional.conv3d"""
    ...

# Transposed convolution operations
def conv_transpose1d[
    InputShape: IntTuple,
    WeightShape: IntTuple,
    Stride: Flag[builtins.int | tuple[builtins.int]],
    Padding: Flag[builtins.int | tuple[builtins.int]],
    OutputPadding: Flag[builtins.int | tuple[builtins.int]],
    Dilation: Flag[builtins.int | tuple[builtins.int]],
    Groups: Flag[builtins.int],
](
    self: Tensor[InputShape],
    weight: Tensor[WeightShape],
    bias: Tensor | None = None,
    stride: Stride = 1,
    padding: Padding = 0,
    output_padding: OutputPadding = 0,
    dilation: Dilation = 1,
    groups: Groups = 1,
) -> Tensor[
    conv_transpose_shape(
        InputShape, WeightShape, Stride, Padding, OutputPadding, Dilation, Groups
    )
]:
    """1D transposed convolution. Shape inference via meta-shape: torch.nn.functional.conv_transpose1d"""
    ...

def conv_transpose2d[
    InputShape: IntTuple,
    WeightShape: IntTuple,
    Stride: Flag[builtins.int | tuple[builtins.int, builtins.int]],
    Padding: Flag[builtins.int | tuple[builtins.int, builtins.int]],
    OutputPadding: Flag[builtins.int | tuple[builtins.int, builtins.int]],
    Dilation: Flag[builtins.int | tuple[builtins.int, builtins.int]],
    Groups: Flag[builtins.int],
](
    self: Tensor[InputShape],
    weight: Tensor[WeightShape],
    bias: Tensor | None = None,
    stride: Stride = 1,
    padding: Padding = 0,
    output_padding: OutputPadding = 0,
    dilation: Dilation = 1,
    groups: Groups = 1,
) -> Tensor[
    conv_transpose_shape(
        InputShape, WeightShape, Stride, Padding, OutputPadding, Dilation, Groups
    )
]:
    """2D transposed convolution. Shape inference via meta-shape: torch.nn.functional.conv_transpose2d"""
    ...

def conv_transpose3d[
    InputShape: IntTuple,
    WeightShape: IntTuple,
    Stride: Flag[builtins.int | tuple[builtins.int, builtins.int, builtins.int]],
    Padding: Flag[builtins.int | tuple[builtins.int, builtins.int, builtins.int]],
    OutputPadding: Flag[builtins.int | tuple[builtins.int, builtins.int, builtins.int]],
    Dilation: Flag[builtins.int | tuple[builtins.int, builtins.int, builtins.int]],
    Groups: Flag[builtins.int],
](
    self: Tensor[InputShape],
    weight: Tensor[WeightShape],
    bias: Tensor | None = None,
    stride: Stride = 1,
    padding: Padding = 0,
    output_padding: OutputPadding = 0,
    dilation: Dilation = 1,
    groups: Groups = 1,
) -> Tensor[
    conv_transpose_shape(
        InputShape, WeightShape, Stride, Padding, OutputPadding, Dilation, Groups
    )
]:
    """3D transposed convolution. Shape inference via meta-shape: torch.nn.functional.conv_transpose3d"""
    ...

# Max pooling operations
@uses_shape_dsl(pool_ir)
@overload
def max_pool1d(
    self: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    dilation: int | tuple[int] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[False] = False,
) -> Tensor:
    """1D max pooling. Shape inference via meta-shape: torch.nn.functional.max_pool1d"""
    ...

@overload
def max_pool1d(
    self: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    dilation: int | tuple[int] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[True] = True,
) -> tuple[Tensor, Tensor]:
    """1D max pooling with indices. Shape inference via meta-shape: torch.nn.functional.max_pool1d"""
    ...

@uses_shape_dsl(pool_ir)
@overload
def max_pool2d(
    self: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[False] = False,
) -> Tensor:
    """2D max pooling. Shape inference via meta-shape: torch.nn.functional.max_pool2d"""
    ...

@overload
def max_pool2d(
    self: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[True] = True,
) -> tuple[Tensor, Tensor]:
    """2D max pooling with indices. Shape inference via meta-shape: torch.nn.functional.max_pool2d"""
    ...

@uses_shape_dsl(pool_ir)
@overload
def max_pool3d(
    self: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[False] = False,
) -> Tensor:
    """3D max pooling. Shape inference via meta-shape: torch.nn.functional.max_pool3d"""
    ...

@overload
def max_pool3d(
    self: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    ceil_mode: bool = False,
    return_indices: Literal[True] = True,
) -> tuple[Tensor, Tensor]:
    """3D max pooling with indices. Shape inference via meta-shape: torch.nn.functional.max_pool3d"""
    ...

# Average pooling operations
@uses_shape_dsl(pool_ir)
def avg_pool1d(
    self: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """1D average pooling. Shape inference via meta-shape: torch.nn.functional.avg_pool1d"""
    ...

@uses_shape_dsl(pool_ir)
def avg_pool2d(
    self: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """2D average pooling. Shape inference via meta-shape: torch.nn.functional.avg_pool2d"""
    ...

@uses_shape_dsl(pool_ir)
def avg_pool3d(
    self: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """3D average pooling. Shape inference via meta-shape: torch.nn.functional.avg_pool3d"""
    ...

# Adaptive max pooling operations
@overload
def adaptive_max_pool1d[Shape: IntTuple, O: _Int](
    input: Tensor[Shape],
    output_size: O,
    return_indices: Literal[False] = False,
) -> Tensor[adaptive_pool1d_shape(Shape, O)]:
    """1D adaptive max pooling. Shape inference via type-level DSL."""
    ...

@overload
def adaptive_max_pool1d[Shape: IntTuple, O: _Int](
    input: Tensor[Shape],
    output_size: tuple[O],
    return_indices: Literal[False] = False,
) -> Tensor[adaptive_pool1d_shape(Shape, O)]: ...
@overload
def adaptive_max_pool1d[Shape: IntTuple](
    input: Tensor[Shape],
    output_size: int | tuple[int],
    return_indices: Literal[True],
) -> tuple[
    Tensor[adaptive_pool_gradual_shape(Shape, 1)],
    Tensor[adaptive_pool_gradual_shape(Shape, 1)],
]: ...
@overload
def adaptive_max_pool1d[Shape: IntTuple](
    input: Tensor[Shape], output_size: int | tuple[int], return_indices: bool
) -> (
    Tensor[adaptive_pool_gradual_shape(Shape, 1)]
    | tuple[
        Tensor[adaptive_pool_gradual_shape(Shape, 1)],
        Tensor[adaptive_pool_gradual_shape(Shape, 1)],
    ]
): ...
@overload
def adaptive_max_pool2d[Shape: IntTuple, O: _Int](
    input: Tensor[Shape],
    output_size: O,
    return_indices: Literal[False] = False,
) -> Tensor[adaptive_pool2d_shape(Shape, O, O)]:
    """2D adaptive max pooling. Shape inference via type-level DSL."""
    ...

@overload
def adaptive_max_pool2d[Shape: IntTuple, OH: _Int, OW: _Int](
    input: Tensor[Shape],
    output_size: tuple[OH, OW],
    return_indices: Literal[False] = False,
) -> Tensor[adaptive_pool2d_shape(Shape, OH, OW)]: ...
@overload
def adaptive_max_pool2d[Shape: IntTuple](
    input: Tensor[Shape],
    output_size: tuple[int | None, int | None],
    return_indices: Literal[False] = False,
) -> Tensor[adaptive_pool_gradual_shape(Shape, 2)]: ...
@overload
def adaptive_max_pool2d[Shape: IntTuple](
    input: Tensor[Shape],
    output_size: int | tuple[int | None, int | None],
    return_indices: Literal[True],
) -> tuple[
    Tensor[adaptive_pool_gradual_shape(Shape, 2)],
    Tensor[adaptive_pool_gradual_shape(Shape, 2)],
]: ...
@overload
def adaptive_max_pool2d[Shape: IntTuple](
    input: Tensor[Shape],
    output_size: int | tuple[int | None, int | None],
    return_indices: bool,
) -> (
    Tensor[adaptive_pool_gradual_shape(Shape, 2)]
    | tuple[
        Tensor[adaptive_pool_gradual_shape(Shape, 2)],
        Tensor[adaptive_pool_gradual_shape(Shape, 2)],
    ]
): ...
@overload
def adaptive_max_pool3d[Shape: IntTuple, O: _Int](
    input: Tensor[Shape],
    output_size: O,
    return_indices: Literal[False] = False,
) -> Tensor[adaptive_pool3d_shape(Shape, O, O, O)]:
    """3D adaptive max pooling. Shape inference via type-level DSL."""
    ...

@overload
def adaptive_max_pool3d[Shape: IntTuple, OD: _Int, OH: _Int, OW: _Int](
    input: Tensor[Shape],
    output_size: tuple[OD, OH, OW],
    return_indices: Literal[False] = False,
) -> Tensor[adaptive_pool3d_shape(Shape, OD, OH, OW)]: ...
@overload
def adaptive_max_pool3d[Shape: IntTuple](
    input: Tensor[Shape],
    output_size: tuple[int | None, int | None, int | None],
    return_indices: Literal[False] = False,
) -> Tensor[adaptive_pool_gradual_shape(Shape, 3)]: ...
@overload
def adaptive_max_pool3d[Shape: IntTuple](
    input: Tensor[Shape],
    output_size: int | tuple[int | None, int | None, int | None],
    return_indices: Literal[True],
) -> tuple[
    Tensor[adaptive_pool_gradual_shape(Shape, 3)],
    Tensor[adaptive_pool_gradual_shape(Shape, 3)],
]: ...
@overload
def adaptive_max_pool3d[Shape: IntTuple](
    input: Tensor[Shape],
    output_size: int | tuple[int | None, int | None, int | None],
    return_indices: bool,
) -> (
    Tensor[adaptive_pool_gradual_shape(Shape, 3)]
    | tuple[
        Tensor[adaptive_pool_gradual_shape(Shape, 3)],
        Tensor[adaptive_pool_gradual_shape(Shape, 3)],
    ]
): ...

# Adaptive average pooling operations
@overload
def adaptive_avg_pool1d[Shape: IntTuple, O: _Int](
    input: Tensor[Shape], output_size: O
) -> Tensor[adaptive_pool1d_shape(Shape, O)]:
    """1D adaptive average pooling. Shape inference via type-level DSL."""
    ...

@overload
def adaptive_avg_pool1d[Shape: IntTuple, O: _Int](
    input: Tensor[Shape], output_size: tuple[O]
) -> Tensor[adaptive_pool1d_shape(Shape, O)]: ...
@overload
def adaptive_avg_pool2d[Shape: IntTuple, O: _Int](
    input: Tensor[Shape], output_size: O
) -> Tensor[adaptive_pool2d_shape(Shape, O, O)]:
    """2D adaptive average pooling. Shape inference via type-level DSL."""
    ...

@overload
def adaptive_avg_pool2d[Shape: IntTuple, OH: _Int, OW: _Int](
    input: Tensor[Shape], output_size: tuple[OH, OW]
) -> Tensor[adaptive_pool2d_shape(Shape, OH, OW)]: ...
@overload
def adaptive_avg_pool2d[Shape: IntTuple](
    input: Tensor[Shape], output_size: tuple[int | None, int | None]
) -> Tensor[adaptive_pool_gradual_shape(Shape, 2)]: ...
@overload
def adaptive_avg_pool3d[Shape: IntTuple, O: _Int](
    input: Tensor[Shape], output_size: O
) -> Tensor[adaptive_pool3d_shape(Shape, O, O, O)]:
    """3D adaptive average pooling. Shape inference via type-level DSL."""
    ...

@overload
def adaptive_avg_pool3d[Shape: IntTuple, OD: _Int, OH: _Int, OW: _Int](
    input: Tensor[Shape], output_size: tuple[OD, OH, OW]
) -> Tensor[adaptive_pool3d_shape(Shape, OD, OH, OW)]: ...
@overload
def adaptive_avg_pool3d[Shape: IntTuple](
    input: Tensor[Shape], output_size: tuple[int | None, int | None, int | None]
) -> Tensor[adaptive_pool_gradual_shape(Shape, 3)]: ...

# Interpolation/upsampling operations
@uses_shape_dsl(interpolate_ir)
def interpolate(
    self: Tensor,
    size: int | tuple[int, ...] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
    antialias: bool = False,
) -> Tensor:
    """Interpolate/upsample tensor. Shape inference via meta-shape: torch.nn.functional.interpolate"""
    ...

@uses_shape_dsl(interpolate_ir)
def upsample(
    self: Tensor,
    size: int | tuple[int, ...] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> Tensor:
    """Upsample tensor (deprecated, use interpolate). Shape inference via meta-shape: torch.nn.functional.upsample"""
    ...

# Phase 2: Activation functions
def relu[Shape: IntTuple](input: Tensor[Shape], inplace: bool = False) -> Tensor[Shape]:
    """ReLU activation. Shape inference via generic fixture signature."""
    ...

def gelu[Shape: IntTuple](
    input: Tensor[Shape], approximate: str = "none"
) -> Tensor[Shape]:
    """GELU activation. Shape inference via generic fixture signature."""
    ...

def silu[Shape: IntTuple](input: Tensor[Shape], inplace: bool = False) -> Tensor[Shape]:
    """SiLU (Swish) activation. Shape inference via generic fixture signature."""
    ...

def selu[Shape: IntTuple](input: Tensor[Shape], inplace: bool = False) -> Tensor[Shape]:
    """SELU activation. Shape inference via generic fixture signature."""
    ...

def elu[Shape: IntTuple](
    input: Tensor[Shape], alpha: float = 1.0, inplace: bool = False
) -> Tensor[Shape]:
    """ELU activation. Shape inference via generic fixture signature."""
    ...

def leaky_relu[Shape: IntTuple](
    input: Tensor[Shape], negative_slope: float = 0.01, inplace: bool = False
) -> Tensor[Shape]:
    """Leaky ReLU activation. Shape inference via generic fixture signature."""
    ...

def relu6[Shape: IntTuple](
    input: Tensor[Shape], inplace: bool = False
) -> Tensor[Shape]:
    """ReLU6 activation. Shape inference via generic fixture signature."""
    ...

def softplus[Shape: IntTuple](
    input: Tensor[Shape], beta: float = 1, threshold: float = 20
) -> Tensor[Shape]:
    """Softplus activation. Shape inference via generic fixture signature."""
    ...

def softsign[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Softsign activation. Shape inference via generic fixture signature."""
    ...

def hardtanh[Shape: IntTuple](
    input: Tensor[Shape],
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor[Shape]:
    """Hardtanh activation. Shape inference via generic fixture signature."""
    ...

def hardsigmoid[Shape: IntTuple](
    input: Tensor[Shape], inplace: bool = False
) -> Tensor[Shape]:
    """Hardsigmoid activation. Shape inference via generic fixture signature."""
    ...

def hardswish[Shape: IntTuple](
    input: Tensor[Shape], inplace: bool = False
) -> Tensor[Shape]:
    """Hardswish activation. Shape inference via generic fixture signature."""
    ...

def sigmoid[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Sigmoid activation. Shape inference via generic fixture signature."""
    ...

def tanh[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Tanh activation. Shape inference via generic fixture signature."""
    ...

def mish[Shape: IntTuple](input: Tensor[Shape], inplace: bool = False) -> Tensor[Shape]:
    """Mish activation. Shape inference via generic fixture signature."""
    ...

def glu(input: Tensor, dim: int = -1) -> Tensor:
    """GLU activation. Shape inference via meta-shape: torch.nn.functional.glu"""
    ...

def prelu[Shape: IntTuple](input: Tensor[Shape], weight: Tensor) -> Tensor[Shape]:
    """PReLU activation. Shape inference via generic fixture signature."""
    ...

def rrelu[Shape: IntTuple](
    input: Tensor[Shape],
    lower: float = 0.125,
    upper: float = 0.333,
    training: bool = False,
    inplace: bool = False,
) -> Tensor[Shape]:
    """RReLU activation. Shape inference via generic fixture signature."""
    ...

def celu[Shape: IntTuple](
    input: Tensor[Shape], alpha: float = 1.0, inplace: bool = False
) -> Tensor[Shape]:
    """CELU activation. Shape inference via generic fixture signature."""
    ...

# Normalization operations
def batch_norm[Shape: IntTuple](
    input: Tensor[Shape],
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor[Shape]:
    """Batch normalization. Shape inference via generic fixture signature."""
    ...

def instance_norm[Shape: IntTuple](
    input: Tensor[Shape],
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor[Shape]:
    """Instance normalization. Shape inference via generic fixture signature."""
    ...

def layer_norm[Shape: IntTuple](
    input: Tensor[Shape],
    normalized_shape: tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor[Shape]:
    """Layer normalization. Shape inference via generic fixture signature."""
    ...

def group_norm[Shape: IntTuple](
    input: Tensor[Shape],
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor[Shape]:
    """Group normalization. Shape inference via generic fixture signature."""
    ...

def normalize[Shape: IntTuple](
    input: Tensor[Shape], p: float = 2.0, dim: int = 1, eps: float = 1e-12
) -> Tensor[Shape]:
    """Normalize tensor. Shape inference via generic fixture signature."""
    ...

def local_response_norm[Shape: IntTuple](
    input: Tensor[Shape],
    size: int,
    alpha: float = 0.0001,
    beta: float = 0.75,
    k: float = 1.0,
) -> Tensor[Shape]:
    """Local response normalization. Shape inference via generic fixture signature."""
    ...

# Dropout operations
def dropout[Shape: IntTuple](
    input: Tensor[Shape], p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor[Shape]:
    """Dropout. Shape inference via generic fixture signature."""
    ...

def alpha_dropout[Shape: IntTuple](
    input: Tensor[Shape], p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor[Shape]:
    """Alpha dropout. Shape inference via generic fixture signature."""
    ...

def feature_alpha_dropout[Shape: IntTuple](
    input: Tensor[Shape], p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor[Shape]:
    """Feature alpha dropout. Shape inference via generic fixture signature."""
    ...

# Additional activation functions
def threshold[Shape: IntTuple](
    input: Tensor[Shape], threshold: float, value: float, inplace: bool = False
) -> Tensor[Shape]:
    """Threshold activation. Shape inference via generic fixture signature."""
    ...

def tanhshrink[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Tanhshrink activation. Shape inference via generic fixture signature."""
    ...

def softshrink[Shape: IntTuple](
    input: Tensor[Shape], lambd: float = 0.5
) -> Tensor[Shape]:
    """Softshrink activation. Shape inference via generic fixture signature."""
    ...

def hardshrink[Shape: IntTuple](
    input: Tensor[Shape], lambd: float = 0.5
) -> Tensor[Shape]:
    """Hardshrink activation. Shape inference via generic fixture signature."""
    ...

def logsigmoid[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]:
    """Log-sigmoid activation. Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Phase 6: Loss Functions
# ==============================================================================

def mse_loss[
    InputShape: IntTuple,
    TargetShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor[TargetShape],
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(InputShape, TargetShape),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """Mean squared error loss. Shape inference via type-level DSL."""
    ...

def l1_loss[
    InputShape: IntTuple,
    TargetShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor[TargetShape],
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(InputShape, TargetShape),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """L1 loss. Shape inference via type-level DSL."""
    ...

def nll_loss[
    InputShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor,
    weight: Tensor | None = None,
    size_average: SizeAverage = None,
    ignore_index: int = -100,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[classification_loss_shape(InputShape, Reduction, SizeAverage, Reduce)]:
    """Negative log likelihood loss. Shape inference via type-level DSL."""
    ...

def cross_entropy[
    InputShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor,
    weight: Tensor | None = None,
    size_average: SizeAverage = None,
    ignore_index: int = -100,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
    label_smoothing: float = 0.0,
) -> Tensor[classification_loss_shape(InputShape, Reduction, SizeAverage, Reduce)]:
    """Cross entropy loss. Shape inference via type-level DSL."""
    ...

def binary_cross_entropy[
    InputShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor[InputShape],
    weight: Tensor | None = None,
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[loss_shape(InputShape, Reduction, SizeAverage, Reduce)]:
    """Binary cross entropy loss. Shape inference via type-level DSL."""
    ...

def binary_cross_entropy_with_logits[
    InputShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor[InputShape],
    weight: Tensor | None = None,
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
    pos_weight: Tensor | None = None,
) -> Tensor[loss_shape(InputShape, Reduction, SizeAverage, Reduce)]:
    """Binary cross entropy with logits. Shape inference via type-level DSL."""
    ...

def kl_div[
    InputShape: IntTuple,
    TargetShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor[TargetShape],
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
    log_target: bool = False,
) -> Tensor[
    kl_div_loss_shape(
        shape_extensions.broadcast(InputShape, TargetShape),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """KL divergence loss. Shape inference via type-level DSL."""
    ...

def smooth_l1_loss[
    InputShape: IntTuple,
    TargetShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor[TargetShape],
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
    beta: float = 1.0,
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(InputShape, TargetShape),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """Smooth L1 loss. Shape inference via type-level DSL."""
    ...

def huber_loss[InputShape: IntTuple, TargetShape: IntTuple, Reduction: Flag[str]](
    input: Tensor[InputShape],
    target: Tensor[TargetShape],
    reduction: Reduction = "mean",
    delta: float = 1.0,
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(InputShape, TargetShape), Reduction, None, None
    )
]:
    """Huber loss. Shape inference via type-level DSL."""
    ...

def poisson_nll_loss[
    InputShape: IntTuple,
    TargetShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor[TargetShape],
    log_input: bool = True,
    full: bool = False,
    size_average: SizeAverage = None,
    eps: float = 1e-8,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(InputShape, TargetShape),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """Poisson NLL loss. Shape inference via type-level DSL."""
    ...

def cosine_embedding_loss[
    Input1Shape: IntTuple,
    Input2Shape: IntTuple,
    TargetShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input1: Tensor[Input1Shape],
    input2: Tensor[Input2Shape],
    target: Tensor[TargetShape],
    margin: float = 0.0,
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(
            cosine_embedding_score_shape(
                Input1Shape,
                Input2Shape,
                shape_extensions.broadcast(Input1Shape, Input2Shape),
                TargetShape,
            ),
            TargetShape,
        ),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """Cosine embedding loss. Shape inference via type-level DSL."""
    ...

def margin_ranking_loss[
    Input1Shape: IntTuple,
    Input2Shape: IntTuple,
    TargetShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input1: Tensor[Input1Shape],
    input2: Tensor[Input2Shape],
    target: Tensor[TargetShape],
    margin: float = 0.0,
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(
            shape_extensions.broadcast(Input1Shape, Input2Shape), TargetShape
        ),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """Margin ranking loss. Shape inference via type-level DSL."""
    ...

def triplet_margin_loss[
    AnchorShape: IntTuple,
    PositiveShape: IntTuple,
    NegativeShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    anchor: Tensor[AnchorShape],
    positive: Tensor[PositiveShape],
    negative: Tensor[NegativeShape],
    margin: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(
            pairwise_distance_shape(
                AnchorShape,
                PositiveShape,
                shape_extensions.broadcast(AnchorShape, PositiveShape),
            ),
            pairwise_distance_shape(
                AnchorShape,
                NegativeShape,
                shape_extensions.broadcast(AnchorShape, NegativeShape),
            ),
        ),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """Triplet margin loss. Shape inference via type-level DSL."""
    ...

def hinge_embedding_loss[
    InputShape: IntTuple,
    TargetShape: IntTuple,
    SizeAverage: Flag[bool | None],
    Reduce: Flag[bool | None],
    Reduction: Flag[str],
](
    input: Tensor[InputShape],
    target: Tensor[TargetShape],
    margin: float = 1.0,
    size_average: SizeAverage = None,
    reduce: Reduce = None,
    reduction: Reduction = "mean",
) -> Tensor[
    loss_shape(
        shape_extensions.broadcast(InputShape, TargetShape),
        Reduction,
        SizeAverage,
        Reduce,
    )
]:
    """Hinge embedding loss. Shape inference via type-level DSL."""
    ...

# Padding operation
@overload
def pad[Shape: IntTuple, Pad: Flag[tuple[builtins.int, ...]]](
    input: Tensor[Shape],
    pad: Pad,
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor[pad_shape(Shape, Pad)]:
    """Pad tensor. Shape inference via type-level DSL."""
    ...

@overload
def pad(
    input: Tensor,
    pad: list[builtins.int],
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor[IntTuple]:
    """Pad tensor by a list of amounts. A list carries no element literals, so the
    padded shape stays gradual.

    TODO(stroxler): Preserve list element literals when mutable sequence arguments can carry
    shape values into the type-level DSL.
    """
    ...

# Softmax activation
def softmax[Shape: IntTuple](
    input: Tensor[Shape], dim: int | None = None, dtype: int | None = None
) -> Tensor[Shape]:
    """Softmax activation. Shape inference via generic fixture signature."""
    ...

def log_softmax[Shape: IntTuple](
    input: Tensor[Shape], dim: int | None = None, dtype: int | None = None
) -> Tensor[Shape]:
    """Log-softmax activation. Shape inference via generic fixture signature."""
    ...

def softmin[Shape: IntTuple](
    input: Tensor[Shape], dim: int | None = None, dtype: int | None = None
) -> Tensor[Shape]:
    """Softmin activation. Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Linear
# ==============================================================================

def linear[Bs: IntTuple, IN: IntVar, OUT: IntVar](
    input: Tensor[[*Elements[Bs], IN]],
    weight: Tensor[[OUT, IN]],
    bias: Tensor[[OUT]] | None = None,
) -> Tensor[[*Elements[Bs], OUT]]:
    """Linear transformation: y = xA^T + b. Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Embedding
# ==============================================================================

@overload
def embedding[T: IntVar, V: IntVar, D: IntVar](
    input: Tensor[[T]],
    weight: Tensor[[V, D]],
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor[[T, D]]: ...
@overload
def embedding[B: IntVar, T: IntVar, V: IntVar, D: IntVar](
    input: Tensor[[B, T]],
    weight: Tensor[[V, D]],
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor[[B, T, D]]: ...

# ==============================================================================
# Normalization (additional)
# ==============================================================================

def rms_norm[S: IntTuple](
    input: Tensor[S],
    normalized_shape: list[int] | tuple[int, ...],
    weight: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor[S]:
    """RMS normalization. Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Dropout (additional)
# ==============================================================================

def dropout1d[S: IntTuple](
    input: Tensor[S], p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor[S]:
    """1D channel-wise dropout. Shape inference via generic fixture signature."""
    ...

def dropout2d[S: IntTuple](
    input: Tensor[S], p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor[S]:
    """2D channel-wise dropout. Shape inference via generic fixture signature."""
    ...

def dropout3d[S: IntTuple](
    input: Tensor[S], p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor[S]:
    """3D channel-wise dropout. Shape inference via generic fixture signature."""
    ...

# Attention operations
def scaled_dot_product_attention[
    B: IntVar,
    H: IntVar,
    Tq: IntVar,
    Tkv: IntVar,
    D: IntVar,
    Dv: IntVar,
](
    query: Tensor[[B, H, Tq, D]],
    key: Tensor[[B, H, Tkv, D]],
    value: Tensor[[B, H, Tkv, Dv]],
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor[[B, H, Tq, Dv]]:
    """Scaled dot product attention. Shape inference via meta-shape: torch.nn.functional.scaled_dot_product_attention"""
    ...

def cosine_similarity[S1: IntTuple, S2: IntTuple, Dim: Flag[builtins.int]](
    x1: Tensor[S1], x2: Tensor[S2], dim: Dim = 1, eps: float = 1e-8
) -> Tensor[cosine_similarity_shape(shape_extensions.broadcast(S1, S2), Dim)]:
    """Cosine similarity: dot product along dim, normalized."""
    ...

def grid_sample[B: IntVar, C: IntVar, Hout: IntVar, Wout: IntVar](
    input: Tensor[[B, C, *Elements[IntTuple]]],
    grid: Tensor[[B, Hout, Wout, 2]],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool | None = None,
) -> Tensor[[B, C, Hout, Wout]]:
    """Sample input using grid of coordinates. Output spatial dims match grid."""
    ...
