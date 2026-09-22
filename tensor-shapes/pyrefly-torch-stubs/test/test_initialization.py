# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor
from torch.nn import init


def test_deprecated_initializers_preserve_shape() -> None:
    tensor = torch.zeros((2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        results = (
            init.constant(tensor, 0),
            init.eye(tensor),
            init.kaiming_normal(tensor),
            init.kaiming_uniform(tensor),
            init.normal(tensor),
            init.orthogonal(tensor),
            init.sparse(tensor, 0.5),
            init.uniform(tensor),
            init.xavier_normal(tensor),
            init.xavier_uniform(tensor),
        )
    for result in results:
        assert result is tensor
        assert_shape(result.shape, (2, 3))


if TYPE_CHECKING:

    def check_symbolic_initializers[Shape: IntTuple](tensor: Tensor[Shape]) -> None:
        assert_type(init.constant(tensor, 0), Tensor[Shape])
        assert_type(init.normal(tensor), Tensor[Shape])
        assert_type(init.uniform(tensor), Tensor[Shape])
