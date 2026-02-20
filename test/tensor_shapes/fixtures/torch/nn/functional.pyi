# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Type stubs for torch.nn.functional module.
Functional neural network operations including convolution, pooling, activation, and normalization.
"""

from typing import Literal, overload

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
    # Normalization
    "batch_norm",
    "instance_norm",
    "layer_norm",
    "group_norm",
    "normalize",
    "local_response_norm",
    # Dropout
    "dropout",
    "alpha_dropout",
    "feature_alpha_dropout",
    # Attention
    "scaled_dot_product_attention",
]

# ====================================================================
# Phase 3: Convolution & Pooling Operations
# ====================================================================

# Convolution operations
def conv1d(
    self: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
    dilation: int | tuple[int] = 1,
    groups: int = 1,
) -> Tensor:
    """1D convolution. Shape inference via meta-shape: torch.nn.functional.conv1d"""
    ...

def conv2d(
    self: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """2D convolution. Shape inference via meta-shape: torch.nn.functional.conv2d"""
    ...

def conv3d(
    self: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """3D convolution. Shape inference via meta-shape: torch.nn.functional.conv3d"""
    ...

# Transposed convolution operations
def conv_transpose1d(
    self: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
    output_padding: int | tuple[int] = 0,
    dilation: int | tuple[int] = 1,
    groups: int = 1,
) -> Tensor:
    """1D transposed convolution. Shape inference via meta-shape: torch.nn.functional.conv_transpose1d"""
    ...

def conv_transpose2d(
    self: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """2D transposed convolution. Shape inference via meta-shape: torch.nn.functional.conv_transpose2d"""
    ...

def conv_transpose3d(
    self: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    output_padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """3D transposed convolution. Shape inference via meta-shape: torch.nn.functional.conv_transpose3d"""
    ...

# Max pooling operations
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
def adaptive_max_pool1d(
    self: Tensor, output_size: int | tuple[int], return_indices: bool = False
) -> Tensor:
    """1D adaptive max pooling. Shape inference via meta-shape: torch.nn.functional.adaptive_max_pool1d"""
    ...

def adaptive_max_pool2d(
    self: Tensor,
    output_size: int | tuple[int, int] | None,
    return_indices: bool = False,
) -> Tensor:
    """2D adaptive max pooling. Shape inference via meta-shape: torch.nn.functional.adaptive_max_pool2d"""
    ...

def adaptive_max_pool3d(
    self: Tensor,
    output_size: int | tuple[int, int, int] | None,
    return_indices: bool = False,
) -> Tensor:
    """3D adaptive max pooling. Shape inference via meta-shape: torch.nn.functional.adaptive_max_pool3d"""
    ...

# Adaptive average pooling operations
def adaptive_avg_pool1d(self: Tensor, output_size: int | tuple[int]) -> Tensor:
    """1D adaptive average pooling. Shape inference via meta-shape: torch.nn.functional.adaptive_avg_pool1d"""
    ...

def adaptive_avg_pool2d(
    self: Tensor, output_size: int | tuple[int, int] | None
) -> Tensor:
    """2D adaptive average pooling. Shape inference via meta-shape: torch.nn.functional.adaptive_avg_pool2d"""
    ...

def adaptive_avg_pool3d(
    self: Tensor, output_size: int | tuple[int, int, int] | None
) -> Tensor:
    """3D adaptive average pooling. Shape inference via meta-shape: torch.nn.functional.adaptive_avg_pool3d"""
    ...

# Interpolation/upsampling operations
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
def relu[*Shape](input: Tensor[*Shape], inplace: bool = False) -> Tensor[*Shape]:
    """ReLU activation. Shape inference via generic fixture signature."""
    ...

def gelu[*Shape](input: Tensor[*Shape], approximate: str = "none") -> Tensor[*Shape]:
    """GELU activation. Shape inference via generic fixture signature."""
    ...

def silu[*Shape](input: Tensor[*Shape], inplace: bool = False) -> Tensor[*Shape]:
    """SiLU (Swish) activation. Shape inference via generic fixture signature."""
    ...

def selu[*Shape](input: Tensor[*Shape], inplace: bool = False) -> Tensor[*Shape]:
    """SELU activation. Shape inference via generic fixture signature."""
    ...

def elu[*Shape](
    input: Tensor[*Shape], alpha: float = 1.0, inplace: bool = False
) -> Tensor[*Shape]:
    """ELU activation. Shape inference via generic fixture signature."""
    ...

def leaky_relu[*Shape](
    input: Tensor[*Shape], negative_slope: float = 0.01, inplace: bool = False
) -> Tensor[*Shape]:
    """Leaky ReLU activation. Shape inference via generic fixture signature."""
    ...

def relu6[*Shape](input: Tensor[*Shape], inplace: bool = False) -> Tensor[*Shape]:
    """ReLU6 activation. Shape inference via generic fixture signature."""
    ...

def softplus[*Shape](
    input: Tensor[*Shape], beta: float = 1, threshold: float = 20
) -> Tensor[*Shape]:
    """Softplus activation. Shape inference via generic fixture signature."""
    ...

def softsign[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Softsign activation. Shape inference via generic fixture signature."""
    ...

def hardtanh[*Shape](
    input: Tensor[*Shape],
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor[*Shape]:
    """Hardtanh activation. Shape inference via generic fixture signature."""
    ...

def hardsigmoid[*Shape](input: Tensor[*Shape], inplace: bool = False) -> Tensor[*Shape]:
    """Hardsigmoid activation. Shape inference via generic fixture signature."""
    ...

def hardswish[*Shape](input: Tensor[*Shape], inplace: bool = False) -> Tensor[*Shape]:
    """Hardswish activation. Shape inference via generic fixture signature."""
    ...

def sigmoid[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Sigmoid activation. Shape inference via generic fixture signature."""
    ...

def tanh[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Tanh activation. Shape inference via generic fixture signature."""
    ...

def mish[*Shape](input: Tensor[*Shape], inplace: bool = False) -> Tensor[*Shape]:
    """Mish activation. Shape inference via generic fixture signature."""
    ...

def glu(input: Tensor, dim: int = -1) -> Tensor:
    """GLU activation. Shape inference via meta-shape: torch.nn.functional.glu"""
    ...

def prelu[*Shape](input: Tensor[*Shape], weight: Tensor) -> Tensor[*Shape]:
    """PReLU activation. Shape inference via generic fixture signature."""
    ...

def rrelu[*Shape](
    input: Tensor[*Shape],
    lower: float = 0.125,
    upper: float = 0.333,
    training: bool = False,
    inplace: bool = False,
) -> Tensor[*Shape]:
    """RReLU activation. Shape inference via generic fixture signature."""
    ...

def celu[*Shape](
    input: Tensor[*Shape], alpha: float = 1.0, inplace: bool = False
) -> Tensor[*Shape]:
    """CELU activation. Shape inference via generic fixture signature."""
    ...

# Normalization operations
def batch_norm[*Shape](
    input: Tensor[*Shape],
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor[*Shape]:
    """Batch normalization. Shape inference via generic fixture signature."""
    ...

def instance_norm[*Shape](
    input: Tensor[*Shape],
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor[*Shape]:
    """Instance normalization. Shape inference via generic fixture signature."""
    ...

def layer_norm[*Shape](
    input: Tensor[*Shape],
    normalized_shape: tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor[*Shape]:
    """Layer normalization. Shape inference via generic fixture signature."""
    ...

def group_norm[*Shape](
    input: Tensor[*Shape],
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor[*Shape]:
    """Group normalization. Shape inference via generic fixture signature."""
    ...

def normalize[*Shape](
    input: Tensor[*Shape], p: float = 2.0, dim: int = 1, eps: float = 1e-12
) -> Tensor[*Shape]:
    """Normalize tensor. Shape inference via generic fixture signature."""
    ...

def local_response_norm[*Shape](
    input: Tensor[*Shape],
    size: int,
    alpha: float = 0.0001,
    beta: float = 0.75,
    k: float = 1.0,
) -> Tensor[*Shape]:
    """Local response normalization. Shape inference via generic fixture signature."""
    ...

# Dropout operations
def dropout[*Shape](
    input: Tensor[*Shape], p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor[*Shape]:
    """Dropout. Shape inference via generic fixture signature."""
    ...

def alpha_dropout[*Shape](
    input: Tensor[*Shape], p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor[*Shape]:
    """Alpha dropout. Shape inference via generic fixture signature."""
    ...

def feature_alpha_dropout[*Shape](
    input: Tensor[*Shape], p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor[*Shape]:
    """Feature alpha dropout. Shape inference via generic fixture signature."""
    ...

# Additional activation functions
def threshold[*Shape](
    input: Tensor[*Shape], threshold: float, value: float, inplace: bool = False
) -> Tensor[*Shape]:
    """Threshold activation. Shape inference via generic fixture signature."""
    ...

def tanhshrink[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Tanhshrink activation. Shape inference via generic fixture signature."""
    ...

def softshrink[*Shape](input: Tensor[*Shape], lambd: float = 0.5) -> Tensor[*Shape]:
    """Softshrink activation. Shape inference via generic fixture signature."""
    ...

def hardshrink[*Shape](input: Tensor[*Shape], lambd: float = 0.5) -> Tensor[*Shape]:
    """Hardshrink activation. Shape inference via generic fixture signature."""
    ...

def logsigmoid[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]:
    """Log-sigmoid activation. Shape inference via generic fixture signature."""
    ...

# ==============================================================================
# Phase 6: Loss Functions
# ==============================================================================

def mse_loss(
    self: Tensor,
    target: Tensor,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """Mean squared error loss. Shape inference via meta-shape: torch.nn.functional.mse_loss"""
    ...

def l1_loss(
    self: Tensor,
    target: Tensor,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """L1 loss. Shape inference via meta-shape: torch.nn.functional.l1_loss"""
    ...

def nll_loss(
    self: Tensor,
    target: Tensor,
    weight: Tensor = None,
    size_average: bool = None,
    ignore_index: int = -100,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """Negative log likelihood loss. Shape inference via meta-shape: torch.nn.functional.nll_loss"""
    ...

def cross_entropy(
    self: Tensor,
    target: Tensor,
    weight: Tensor = None,
    size_average: bool = None,
    ignore_index: int = -100,
    reduce: bool = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross entropy loss. Shape inference via meta-shape: torch.nn.functional.cross_entropy"""
    ...

def binary_cross_entropy(
    self: Tensor,
    target: Tensor,
    weight: Tensor = None,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """Binary cross entropy loss. Shape inference via meta-shape: torch.nn.functional.binary_cross_entropy"""
    ...

def binary_cross_entropy_with_logits(
    self: Tensor,
    target: Tensor,
    weight: Tensor = None,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
    pos_weight: Tensor = None,
) -> Tensor:
    """Binary cross entropy with logits. Shape inference via meta-shape: torch.nn.functional.binary_cross_entropy_with_logits"""
    ...

def kl_div(
    self: Tensor,
    target: Tensor,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    """KL divergence loss. Shape inference via meta-shape: torch.nn.functional.kl_div"""
    ...

def smooth_l1_loss(
    self: Tensor,
    target: Tensor,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    """Smooth L1 loss. Shape inference via meta-shape: torch.nn.functional.smooth_l1_loss"""
    ...

def huber_loss(
    self: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0
) -> Tensor:
    """Huber loss. Shape inference via meta-shape: torch.nn.functional.huber_loss"""
    ...

def poisson_nll_loss(
    self: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    size_average: bool = None,
    eps: float = 1e-8,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """Poisson NLL loss. Shape inference via meta-shape: torch.nn.functional.poisson_nll_loss"""
    ...

def cosine_embedding_loss(
    self: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """Cosine embedding loss. Shape inference via meta-shape: torch.nn.functional.cosine_embedding_loss"""
    ...

def margin_ranking_loss(
    self: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """Margin ranking loss. Shape inference via meta-shape: torch.nn.functional.margin_ranking_loss"""
    ...

def triplet_margin_loss(
    self: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """Triplet margin loss. Shape inference via meta-shape: torch.nn.functional.triplet_margin_loss"""
    ...

def hinge_embedding_loss(
    self: Tensor,
    target: Tensor,
    margin: float = 1.0,
    size_average: bool = None,
    reduce: bool = None,
    reduction: str = "mean",
) -> Tensor:
    """Hinge embedding loss. Shape inference via meta-shape: torch.nn.functional.hinge_embedding_loss"""
    ...

# Padding operation
def pad(
    self: Tensor, pad: tuple[int, ...], mode: str = "constant", value: float = 0.0
) -> Tensor:
    """Pad tensor. Shape inference via meta-shape: torch.nn.functional.pad"""
    ...

# Softmax activation
def softmax[*Shape](
    input: Tensor[*Shape], dim: int | None = None, dtype: int | None = None
) -> Tensor[*Shape]:
    """Softmax activation. Shape inference via generic fixture signature."""
    ...

# Attention operations
def scaled_dot_product_attention[
    B,
    H,
    Tq,
    Tkv,
    D,
    Dv,
](
    query: Tensor[B, H, Tq, D],
    key: Tensor[B, H, Tkv, D],
    value: Tensor[B, H, Tkv, Dv],
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor[B, H, Tq, Dv]:
    """Scaled dot product attention. Shape inference via meta-shape: torch.nn.functional.scaled_dot_product_attention"""
    ...
