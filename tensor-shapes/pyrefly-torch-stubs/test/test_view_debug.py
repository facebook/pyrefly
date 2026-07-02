# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Debug .view() argument passing"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_various_view_syntaxes():
    """Test different ways to call .view()"""
    x: Tensor[10, 20] = torch.randn(10, 20)

    # Single argument - flatten
    y1 = x.view(-1)
    assert_type(y1, Tensor[200])

    # Multiple arguments
    y2 = x.view(2, -1)
    assert_type(y2, Tensor[2, 100])

    # Tuple argument (if supported)
    # y3 = x.view((-1,))
    # reveal_type(y3)

    # Explicit dimensions
    y4 = x.view(200)
    assert_type(y4, Tensor[200])

    y5 = x.view(10, 20)
    assert_type(y5, Tensor[10, 20])
