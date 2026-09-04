# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor


@torch.no_grad()
def decorated[Shape: IntTuple](tensor: Tensor[Shape]) -> Tensor[Shape]:
    return tensor.relu()


@torch.no_grad
def bare_decorated[Shape: IntTuple](
    tensor: Tensor[Shape], *, scale: float = 1.0
) -> Tensor[Shape]:
    return tensor * scale


def test_no_grad_decorators_preserve_shapes() -> None:
    tensor = torch.ones((2, 3), requires_grad=True)

    result = decorated(tensor)
    assert_shape(result.shape, (2, 3))
    assert not result.requires_grad

    result = bare_decorated(tensor, scale=2.0)
    assert_shape(result.shape, (2, 3))
    assert not result.requires_grad


def test_no_grad_context_manager_preserves_shapes() -> None:
    tensor = torch.ones((4, 5), requires_grad=True)
    with torch.no_grad():
        result = tensor.relu()
    assert_shape(result.shape, (4, 5))
    assert not result.requires_grad
    assert torch.is_grad_enabled()


if TYPE_CHECKING:

    def check_decorated_signatures[Shape: IntTuple](tensor: Tensor[Shape]) -> None:
        assert_type(decorated(tensor), Tensor[Shape])
        assert_type(bare_decorated(tensor, scale=2.0), Tensor[Shape])
