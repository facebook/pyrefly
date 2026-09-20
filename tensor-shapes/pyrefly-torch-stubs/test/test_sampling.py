# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor


def test_tensor_sampling_shapes() -> None:
    probabilities = torch.rand((2, 3))
    rates = torch.rand((2, 3))

    assert_shape(torch.bernoulli(probabilities).shape, (2, 3))
    assert_shape(probabilities.bernoulli().shape, (2, 3))
    assert_shape(torch.poisson(rates).shape, (2, 3))

    scalar = torch.rand(())
    assert_shape(torch.bernoulli(scalar).shape, ())
    assert_shape(torch.poisson(scalar).shape, ())


def test_in_place_sampling_shapes() -> None:
    tensor = 0.5 * torch.ones((4, 5))
    generator = torch.Generator()

    assert_shape(tensor.bernoulli_().shape, (4, 5))
    assert_shape(tensor.normal_().shape, (4, 5))
    assert_shape(tensor.random_(generator=generator).shape, (4, 5))
    assert_shape(tensor.uniform_(generator=generator).shape, (4, 5))


if TYPE_CHECKING:

    def check_symbolic_sampling[Shape: IntTuple](tensor: Tensor[Shape]) -> None:
        assert_type(torch.bernoulli(tensor), Tensor[Shape])
        assert_type(tensor.bernoulli(), Tensor[Shape])
        assert_type(tensor.bernoulli_(), Tensor[Shape])
        assert_type(tensor.normal_(), Tensor[Shape])
        assert_type(torch.poisson(tensor), Tensor[Shape])
        assert_type(tensor.random_(), Tensor[Shape])
        assert_type(tensor.uniform_(), Tensor[Shape])
