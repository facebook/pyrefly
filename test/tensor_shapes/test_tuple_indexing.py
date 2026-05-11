# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_tuple_indexing():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    assert_type(x, Tensor[2, 3, 4])

    # Tuple with single element as index
    y = x[:, (-1,), :]
    assert_type(y, Tensor[2, 1, 4])

    # Tuple with multiple elements
    z = x[:, (0, 2), :]
    assert_type(z, Tensor[2, 2, 4])

    # List indexing doesn't preserve length at compile time
    w = x[:, [-1], :]
    assert_type(w, Tensor[2, Any, 4])  # Unknown dimension
