# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_repeat_shapes() -> None:
    assert_shape(torch.ones((2, 1)).repeat(3, 4).shape, (6, 4))
    assert_shape(torch.ones((2, 3)).repeat(4, 1, 2).shape, (4, 2, 6))
    assert_shape(torch.ones((2, 3)).repeat(1, 0).shape, (2, 0))


def test_repeat_rejects_invalid_repeats() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(tensor.repeat(1, 1).shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Number of dimensions of repeat dims can not be smaller
        tensor.repeat(2)

    with assert_raises(RuntimeError):
        # E: repeat dimensions must be non-negative
        tensor.repeat(1, -1)


def test_expand_shapes() -> None:
    tensor = torch.ones((2, 1, 4))
    assert_shape(tensor.expand(2, 5, 4).shape, (2, 5, 4))
    assert_shape(tensor.expand((2, 5, 4)).shape, (2, 5, 4))
    assert_shape(tensor.expand(-1, -1, -1).shape, (2, 1, 4))
    assert_shape(torch.ones((2, 3)).expand(4, -1, -1).shape, (4, 2, 3))
    assert_shape(torch.tensor(1).expand(()).shape, ())
    assert_shape(torch.tensor(1).expand(2, 3).shape, (2, 3))
    assert_shape(torch.empty((0, 1)).expand(0, 4).shape, (0, 4))


def test_expand_rejects_invalid_targets() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(tensor.expand(2, 3).shape, (2, 3))

    with assert_raises(TypeError):
        # TODO: BUG: Reject an empty variadic argument list statically.
        torch.tensor(1).expand()

    with assert_raises(RuntimeError):
        tensor.expand(2)  # E: expand target rank cannot be smaller than input rank

    with assert_raises(RuntimeError):
        # E: expand cannot use -1 for a new leading dimension
        tensor.expand(-1, 2, 3)

    with assert_raises(RuntimeError):
        # E: expand target dimension cannot be less than -1
        tensor.expand(-2, 3)

    with assert_raises(RuntimeError):
        # E: expand cannot resize a non-singleton dimension
        tensor.expand(4, 3)


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> None:
        n = x.size(0)
        m = x.size(1)
        assert_type(x.repeat(n, 3), Tensor[[N * N, 3 * M]])
        assert_type(x.expand(n, m), Tensor[[N, M]])
        assert_type(x.expand(-1, m), Tensor[[N, M]])

    def check_expand_gradual(
        concrete: Tensor[[2, 1]],
        open_rank: Tensor[IntTuple],
        bare: Tensor,
        broad_size: int,
        broad_tuple: tuple[int, ...],
    ) -> None:
        assert_type(concrete.expand(broad_size, 3), Tensor[[2, 3]])
        assert_type(concrete.expand(broad_tuple), Tensor[IntTuple])
        assert_type(open_rank.expand(2, 3), Tensor[IntTuple])
        assert_type(bare.expand(2, 3), Tensor)

    def check_partially_known_shape[Batch: IntTuple](
        x: Tensor[[*Elements[Batch], 2]],
    ) -> None:
        # TODO: BUG: Preserve partially known shapes through tuple unpacking.
        assert_type(x.expand(x.size()), Tensor[IntTuple])

    def check_invalid_argument_types(x: Tensor[[2, 3]]) -> None:
        x.repeat([2, 3])  # E: No matching overload found
        x.expand([2, 3])  # E: No matching overload found
        x.expand((2, 3.0))  # E: No matching overload found
        x.expand((True, 2))  # E: No matching overload found
