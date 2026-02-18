# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_single_arg_symint[N](n: Dim[N]) -> None:
    # Two-argument form - this should definitely work
    t1 = torch.arange(0, n)
    assert_type(t1, Tensor[N])

    # Single-argument form - this is the bug
    # If Issue 5 is still present, this will return Tensor instead of Tensor[N]
    t2 = torch.arange(n)
    assert_type(t2, Tensor[N])


def test_workaround[N](n: Dim[N]) -> None:
    # Workaround: always use two-argument form
    t = torch.arange(0, n)
    assert_type(t, Tensor[N])
