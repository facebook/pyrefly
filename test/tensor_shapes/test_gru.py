# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test nn.GRU shape inference via DSL.

GRU output shapes depend on input_size, hidden_size, num_layers, and
bidirectional — all captured from __init__ and used in nn_gru_forward_ir.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_gru_unidirectional():
    """Single-layer unidirectional GRU."""
    gru = nn.GRU(input_size=256, hidden_size=128)
    x: Tensor[4, 10, 256] = torch.randn(4, 10, 256)
    output, h_n = gru(x)
    assert_type(output, Tensor[4, 10, 128])
    assert_type(h_n, Tensor[1, 4, 128])


def test_gru_bidirectional():
    """Bidirectional GRU doubles the output hidden dim."""
    gru = nn.GRU(input_size=64, hidden_size=32, bidirectional=True)
    x: Tensor[8, 5, 64] = torch.randn(8, 5, 64)
    output, h_n = gru(x)
    assert_type(output, Tensor[8, 5, 64])
    assert_type(h_n, Tensor[2, 8, 32])


def test_gru_multi_layer():
    """Multi-layer GRU: h_n has num_layers stacked."""
    gru = nn.GRU(input_size=128, hidden_size=64, num_layers=3)
    x: Tensor[2, 20, 128] = torch.randn(2, 20, 128)
    output, h_n = gru(x)
    assert_type(output, Tensor[2, 20, 64])
    assert_type(h_n, Tensor[3, 2, 64])
