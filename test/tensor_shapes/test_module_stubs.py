# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for nn.Module subclass stubs: activations, normalization, dropout,
convolution, pooling, loss, and misc modules.
"""

from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# Activation Modules
# ============================================================================


def test_relu():
    relu = nn.ReLU()
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    assert_type(relu(x), Tensor[2, 3, 4])


def test_relu6():
    m = nn.ReLU6()
    x: Tensor[4, 8] = torch.randn(4, 8)
    assert_type(m(x), Tensor[4, 8])


def test_silu():
    m = nn.SiLU()
    x: Tensor[2, 16] = torch.randn(2, 16)
    assert_type(m(x), Tensor[2, 16])


def test_sigmoid():
    m = nn.Sigmoid()
    x: Tensor[3, 5] = torch.randn(3, 5)
    assert_type(m(x), Tensor[3, 5])


def test_tanh():
    m = nn.Tanh()
    x: Tensor[3, 5] = torch.randn(3, 5)
    assert_type(m(x), Tensor[3, 5])


def test_mish():
    m = nn.Mish()
    x: Tensor[2, 4] = torch.randn(2, 4)
    assert_type(m(x), Tensor[2, 4])


def test_hardswish():
    m = nn.Hardswish()
    x: Tensor[2, 4] = torch.randn(2, 4)
    assert_type(m(x), Tensor[2, 4])


def test_hardsigmoid():
    m = nn.Hardsigmoid()
    x: Tensor[2, 4] = torch.randn(2, 4)
    assert_type(m(x), Tensor[2, 4])


def test_leaky_relu():
    m = nn.LeakyReLU(0.1)
    x: Tensor[4, 8] = torch.randn(4, 8)
    assert_type(m(x), Tensor[4, 8])


def test_elu():
    m = nn.ELU()
    x: Tensor[4, 8] = torch.randn(4, 8)
    assert_type(m(x), Tensor[4, 8])


def test_selu():
    m = nn.SELU()
    x: Tensor[4, 8] = torch.randn(4, 8)
    assert_type(m(x), Tensor[4, 8])


def test_celu():
    m = nn.CELU()
    x: Tensor[4, 8] = torch.randn(4, 8)
    assert_type(m(x), Tensor[4, 8])


def test_softplus():
    m = nn.Softplus()
    x: Tensor[4, 8] = torch.randn(4, 8)
    assert_type(m(x), Tensor[4, 8])


def test_prelu():
    m = nn.PReLU()
    x: Tensor[4, 8] = torch.randn(4, 8)
    assert_type(m(x), Tensor[4, 8])


def test_threshold():
    m = nn.Threshold(0.1, 20.0)
    x: Tensor[4, 8] = torch.randn(4, 8)
    assert_type(m(x), Tensor[4, 8])


def test_softmax():
    m = nn.Softmax(dim=1)
    x: Tensor[4, 10] = torch.randn(4, 10)
    assert_type(m(x), Tensor[4, 10])


def test_logsoftmax():
    m = nn.LogSoftmax(dim=1)
    x: Tensor[4, 10] = torch.randn(4, 10)
    assert_type(m(x), Tensor[4, 10])


# ============================================================================
# Normalization Modules
# ============================================================================


def test_layer_norm():
    m = nn.LayerNorm(512)
    x: Tensor[4, 128, 512] = torch.randn(4, 128, 512)
    assert_type(m(x), Tensor[4, 128, 512])


def test_rms_norm():
    m = nn.RMSNorm(512)
    x: Tensor[4, 128, 512] = torch.randn(4, 128, 512)
    assert_type(m(x), Tensor[4, 128, 512])


def test_group_norm():
    m = nn.GroupNorm(8, 64)
    x: Tensor[4, 64, 28, 28] = torch.randn(4, 64, 28, 28)
    assert_type(m(x), Tensor[4, 64, 28, 28])


def test_batch_norm_1d():
    m = nn.BatchNorm1d(32)
    x: Tensor[8, 32] = torch.randn(8, 32)
    assert_type(m(x), Tensor[8, 32])


def test_batch_norm_2d():
    m = nn.BatchNorm2d(64)
    x: Tensor[4, 64, 28, 28] = torch.randn(4, 64, 28, 28)
    assert_type(m(x), Tensor[4, 64, 28, 28])


def test_batch_norm_3d():
    m = nn.BatchNorm3d(32)
    x: Tensor[4, 32, 8, 8, 8] = torch.randn(4, 32, 8, 8, 8)
    assert_type(m(x), Tensor[4, 32, 8, 8, 8])


def test_instance_norm_2d():
    m = nn.InstanceNorm2d(64)
    x: Tensor[4, 64, 28, 28] = torch.randn(4, 64, 28, 28)
    assert_type(m(x), Tensor[4, 64, 28, 28])


# ============================================================================
# Dropout Modules
# ============================================================================


def test_dropout1d():
    m = nn.Dropout1d(0.5)
    x: Tensor[4, 32, 16] = torch.randn(4, 32, 16)
    assert_type(m(x), Tensor[4, 32, 16])


def test_dropout2d():
    m = nn.Dropout2d(0.5)
    x: Tensor[4, 32, 16, 16] = torch.randn(4, 32, 16, 16)
    assert_type(m(x), Tensor[4, 32, 16, 16])


def test_dropout3d():
    m = nn.Dropout3d(0.5)
    x: Tensor[4, 32, 8, 8, 8] = torch.randn(4, 32, 8, 8, 8)
    assert_type(m(x), Tensor[4, 32, 8, 8, 8])


def test_alpha_dropout():
    m = nn.AlphaDropout(0.5)
    x: Tensor[4, 32] = torch.randn(4, 32)
    assert_type(m(x), Tensor[4, 32])


# ============================================================================
# Identity Module
# ============================================================================


def test_identity():
    m = nn.Identity()
    x: Tensor[4, 3, 32, 32] = torch.randn(4, 3, 32, 32)
    assert_type(m(x), Tensor[4, 3, 32, 32])


# ============================================================================
# Convolution Modules
# ============================================================================


def test_conv1d():
    conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)
    x: Tensor[4, 16, 100] = torch.randn(4, 16, 100)
    y = conv(x)
    assert_type(y, Tensor)


def test_conv2d_padding():
    conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    x: Tensor[4, 3, 32, 32] = torch.randn(4, 3, 32, 32)
    y = conv(x)
    assert_type(y, Tensor)


def test_conv_transpose2d():
    conv = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
    x: Tensor[4, 128, 16, 16] = torch.randn(4, 128, 16, 16)
    y = conv(x)
    assert_type(y, Tensor)


# ============================================================================
# Pooling Modules
# ============================================================================


def test_maxpool2d():
    pool = nn.MaxPool2d(2, 2)
    x: Tensor[4, 64, 32, 32] = torch.randn(4, 64, 32, 32)
    y = pool(x)
    # Pool module forward returns unrefined Tensor (spatial dims not tracked at module level)
    assert_type(y, Tensor)


def test_avgpool2d():
    pool = nn.AvgPool2d(2, 2)
    x: Tensor[4, 64, 32, 32] = torch.randn(4, 64, 32, 32)
    y = pool(x)
    assert_type(y, Tensor)


def test_adaptive_avg_pool2d():
    pool = nn.AdaptiveAvgPool2d((1, 1))
    x: Tensor[4, 512, 7, 7] = torch.randn(4, 512, 7, 7)
    y = pool(x)
    assert_type(y, Tensor[4, 512, 1, 1])


def test_adaptive_avg_pool1d():
    pool = nn.AdaptiveAvgPool1d(5)
    x: Tensor[4, 64, 100] = torch.randn(4, 64, 100)
    y = pool(x)
    assert_type(y, Tensor[4, 64, 5])


# ============================================================================
# Loss Modules
# ============================================================================


def test_cross_entropy_loss():
    loss_fn = nn.CrossEntropyLoss()
    logits: Tensor[4, 10] = torch.randn(4, 10)
    targets: Tensor[4] = torch.randint(0, 10, (4,))
    loss = loss_fn(logits, targets)
    assert_type(loss, Tensor)


def test_mse_loss():
    loss_fn = nn.MSELoss()
    pred: Tensor[4, 8] = torch.randn(4, 8)
    target: Tensor[4, 8] = torch.randn(4, 8)
    loss = loss_fn(pred, target)
    assert_type(loss, Tensor)


# ============================================================================
# F.* stubs
# ============================================================================


def test_f_linear():
    x: Tensor[4, 128, 256] = torch.randn(4, 128, 256)
    w: Tensor[512, 256] = torch.randn(512, 256)
    y = F.linear(x, w)
    assert_type(y, Tensor[4, 128, 512])


def test_f_log_softmax():
    x: Tensor[4, 10] = torch.randn(4, 10)
    y = F.log_softmax(x, dim=1)
    assert_type(y, Tensor[4, 10])


def test_f_softmin():
    x: Tensor[4, 10] = torch.randn(4, 10)
    y = F.softmin(x, dim=1)
    assert_type(y, Tensor[4, 10])


def test_f_dropout1d():
    x: Tensor[4, 32, 16] = torch.randn(4, 32, 16)
    y = F.dropout1d(x, p=0.5)
    assert_type(y, Tensor[4, 32, 16])


def test_f_dropout2d():
    x: Tensor[4, 32, 16, 16] = torch.randn(4, 32, 16, 16)
    y = F.dropout2d(x, p=0.5)
    assert_type(y, Tensor[4, 32, 16, 16])


def test_f_embedding_1d():
    indices: Tensor[10] = torch.randint(0, 100, (10,))
    weight: Tensor[100, 64] = torch.randn(100, 64)
    y = F.embedding(indices, weight)
    assert_type(y, Tensor[10, 64])


def test_f_embedding_2d():
    indices: Tensor[4, 10] = torch.randint(0, 100, (4, 10))
    weight: Tensor[100, 64] = torch.randn(100, 64)
    y = F.embedding(indices, weight)
    assert_type(y, Tensor[4, 10, 64])


# ============================================================================
# torch.* stubs
# ============================================================================


def test_addmm():
    bias: Tensor[5, 10] = torch.randn(5, 10)
    x: Tensor[5, 8] = torch.randn(5, 8)
    w: Tensor[8, 10] = torch.randn(8, 10)
    y = torch.addmm(bias, x, w)
    assert_type(y, Tensor[5, 10])


def test_cross():
    a: Tensor[4, 3] = torch.randn(4, 3)
    b: Tensor[4, 3] = torch.randn(4, 3)
    y = torch.cross(a, b)
    assert_type(y, Tensor[4, 3])


# ============================================================================
# Flatten / Unflatten
# ============================================================================


def test_flatten_module():
    m = nn.Flatten()
    x: Tensor[4, 3, 32, 32] = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert_type(y, Tensor)


# ============================================================================
# Module as Callable
# ============================================================================


def test_module_as_callable():
    """nn.Module instances are subtypes of Callable matching their forward signature."""
    relu = nn.ReLU()
    # nn.Module subclasses: __call__ delegates to forward
    f: Callable[[Tensor[4, 8]], Tensor[4, 8]] = relu
    y = f(torch.randn(4, 8))
    assert_type(y, Tensor[4, 8])
