# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Type stubs for torch.nn.init module.
Weight initialization functions for neural network parameters.

All initialization functions are in-place operations that preserve the input
tensor's shape and return the same tensor. They use the Tensor[*Shape] pattern
to maintain shape information through initialization calls.
"""

from typing import Literal

from torch import Tensor

__all__ = [
    # Uniform initializations
    "uniform_",
    "normal_",
    # Constant initializations
    "constant_",
    "ones_",
    "zeros_",
    "eye_",
    # Kaiming initializations
    "kaiming_uniform_",
    "kaiming_normal_",
    # Xavier initializations
    "xavier_uniform_",
    "xavier_normal_",
    # Orthogonal initialization
    "orthogonal_",
    # Sparse initialization
    "sparse_",
    # Trunc normal
    "trunc_normal_",
]

# Uniform and normal initializations
def uniform_[*Shape](
    tensor: Tensor[*Shape], a: float = 0.0, b: float = 1.0
) -> Tensor[*Shape]:
    """Fill tensor with values from uniform distribution U(a, b)."""
    ...

def normal_[*Shape](
    tensor: Tensor[*Shape], mean: float = 0.0, std: float = 1.0
) -> Tensor[*Shape]:
    """Fill tensor with values from normal distribution N(mean, std)."""
    ...

# Constant initializations
def constant_[*Shape](tensor: Tensor[*Shape], val: float) -> Tensor[*Shape]:
    """Fill tensor with constant value."""
    ...

def ones_[*Shape](tensor: Tensor[*Shape]) -> Tensor[*Shape]:
    """Fill tensor with ones."""
    ...

def zeros_[*Shape](tensor: Tensor[*Shape]) -> Tensor[*Shape]:
    """Fill tensor with zeros."""
    ...

def eye_[*Shape](tensor: Tensor[*Shape]) -> Tensor[*Shape]:
    """Fill 2D tensor as identity matrix."""
    ...

# Kaiming (He) initialization
def kaiming_uniform_[*Shape](
    tensor: Tensor[*Shape],
    a: float = 0,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor[*Shape]:
    """Kaiming uniform initialization."""
    ...

def kaiming_normal_[*Shape](
    tensor: Tensor[*Shape],
    a: float = 0,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor[*Shape]:
    """Kaiming normal initialization."""
    ...

# Xavier (Glorot) initialization
def xavier_uniform_[*Shape](
    tensor: Tensor[*Shape], gain: float = 1.0
) -> Tensor[*Shape]:
    """Xavier uniform initialization."""
    ...

def xavier_normal_[*Shape](tensor: Tensor[*Shape], gain: float = 1.0) -> Tensor[*Shape]:
    """Xavier normal initialization."""
    ...

# Orthogonal initialization
def orthogonal_[*Shape](tensor: Tensor[*Shape], gain: float = 1.0) -> Tensor[*Shape]:
    """Orthogonal matrix initialization."""
    ...

# Sparse initialization
def sparse_[*Shape](
    tensor: Tensor[*Shape], sparsity: float, std: float = 0.01
) -> Tensor[*Shape]:
    """Sparse initialization."""
    ...

# Truncated normal initialization
def trunc_normal_[*Shape](
    tensor: Tensor[*Shape],
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> Tensor[*Shape]:
    """Fill tensor with truncated normal distribution."""
    ...
