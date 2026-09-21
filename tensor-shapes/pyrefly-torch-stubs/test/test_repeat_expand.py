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


def test_repeat_interleave_shapes() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(torch.repeat_interleave(tensor, 2, dim=1).shape, (2, 6))
    assert_shape(tensor.repeat_interleave(3, dim=0).shape, (6, 3))
    assert_shape(tensor.repeat_interleave(0, dim=1).shape, (2, 0))


def test_repeat_interleave_rejects_invalid_controls() -> None:
    tensor = torch.ones((2, 3))
    tensor_repeats = torch.tensor([2, 3])

    with assert_raises(RuntimeError):
        tensor.repeat_interleave(-1, dim=-1)  # E: repeats must be non-negative
    with assert_raises(RuntimeError):
        # E: output_size must be non-negative
        tensor.repeat_interleave(tensor_repeats, dim=0, output_size=-1)
    with assert_raises(RuntimeError):
        # E: output_size does not match the result
        tensor.repeat_interleave(99, dim=1, output_size=5)
    with assert_raises(IndexError):
        tensor.repeat_interleave(2, dim=2)  # E: dimension out of range
    with assert_raises(IndexError):
        torch.repeat_interleave(tensor, 2, dim=-3)  # E: dimension out of range

    scalar = torch.tensor(1)
    with assert_raises(IndexError):
        scalar.repeat_interleave(2, dim=1)  # E: dimension out of range
    with assert_raises(TypeError):
        tensor.repeat_interleave(1.5)  # E: No matching overload


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

    def check_repeat_interleave[B: IntVar](
        tensor: Tensor[[B, 32]], repeats: int, output_size: int, dim: int
    ) -> None:
        assert_type(torch.repeat_interleave(tensor, 2, dim=1), Tensor[[B, 64]])
        assert_type(torch.repeat_interleave(tensor, repeats, dim=1), Tensor[[B, int]])
        assert_type(tensor.repeat_interleave(2, dim), Tensor[IntTuple])
        assert_type(tensor.repeat_interleave(0, dim=1), Tensor[[B, 0]])
        assert_type(
            torch.repeat_interleave(
                tensor,
                torch.ones(32),
                dim=1,
                output_size=output_size,
            ),
            Tensor[[B, int]],
        )

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

    def check_unbounded_repeat_count[Value, Text: str](
        tensor: Tensor[[4, 32]], value: Value, text: Text
    ) -> None:
        torch.repeat_interleave(tensor, value, dim=1)  # E: No matching overload
        torch.repeat_interleave(tensor, text, dim=1)  # E: No matching overload
