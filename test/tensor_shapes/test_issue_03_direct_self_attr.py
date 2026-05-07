# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch_shapes import Dim


class RMSNorm[D](nn.Module):
    def __init__(self, dim: Dim[D], eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


class DirectAssignmentFails[D](nn.Module):
    """This class demonstrates Issue 3 with direct assignment."""

    ffn_norm: RMSNorm[D]

    def __init__(self, dim: Dim[D]) -> None:
        super().__init__()
        # Direct assignment - type is lost
        self.ffn_norm = RMSNorm(dim, 1e-5)
        # This assert_type will fail if Issue 3 is still present
        # because self.ffn_norm will be RMSNorm[Unknown] instead of RMSNorm[D]
        assert_type(self.ffn_norm, RMSNorm[D])


class WorkaroundWorks[D](nn.Module):
    """This class demonstrates the workaround for Issue 3."""

    ffn_norm: RMSNorm[D]

    def __init__(self, dim: Dim[D]) -> None:
        super().__init__()
        # Workaround: assign to local first
        ffn = RMSNorm(dim, 1e-5)
        self.ffn_norm = ffn
        # This should work
        assert_type(self.ffn_norm, RMSNorm[D])
