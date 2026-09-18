# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple
from torch import Tensor
from torch.nn import init


def test_dirac_preserves_shape_and_tensor() -> None:
    for shape in ((4, 3, 5), (4, 3, 5, 5), (4, 3, 5, 5, 5)):
        tensor = torch.empty(shape)
        assert init.dirac_(tensor) is tensor
        assert tuple(tensor.shape) == shape

    tensor = torch.empty((4, 3, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = init.dirac(tensor, groups=2)
    assert result is tensor
    assert_shape(result.shape, (4, 3, 5))


def test_calculate_gain() -> None:
    assert init.calculate_gain("relu") == 2**0.5
    assert init.calculate_gain("leaky_relu", 0) == 2**0.5
    with assert_raises(ValueError):
        init.calculate_gain("unsupported")  # E: not assignable


if TYPE_CHECKING:

    def check_dirac_shapes[Shape: IntTuple](tensor: Tensor[Shape]) -> None:
        assert_type(init.dirac_(tensor), Tensor[Shape])
        assert_type(init.dirac(tensor, groups=2), Tensor[Shape])
        assert_type(init.calculate_gain("relu"), float)
