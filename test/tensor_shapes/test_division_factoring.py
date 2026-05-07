# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for (a*b)//b → a when b is an expression.

The distributive law expands B*(2*A-1) into -B + 2*A*B (a sum).
Division must factor the sum back to cancel with the denominator.
"""

from __future__ import annotations

from typing import assert_type

import torch
from torch import Tensor
from torch_shapes import Dim


def test_reshape_inferred_dim[A, B](x: Tensor[2 * A - 1, B]) -> None:
    """reshape(-1) computes ((2*A-1)*B) // (2*A-1), should simplify to B."""
    n = x.shape[0]
    step1 = x.reshape(1, n, -1)
    assert_type(step1, Tensor[1, 2 * A - 1, B])


def test_three_factor[A, B, C](x: Tensor[A, B, C]) -> None:
    """(A*B*C) // (A*B) should simplify to C."""
    ab = x.shape[0] * x.shape[1]
    y = x.reshape(ab, -1)
    assert_type(y, Tensor[A * B, C])


def test_expression_divisor[N, C](x: Tensor[2 * N + 1, C]) -> None:
    """((2*N+1)*C) // (2*N+1) should simplify to C."""
    n = x.shape[0]
    y = x.reshape(1, n, -1)
    assert_type(y, Tensor[1, 2 * N + 1, C])
