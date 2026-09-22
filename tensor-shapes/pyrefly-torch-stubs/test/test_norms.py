# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_norm_shapes() -> None:
    matrix = torch.randn((3, 4))
    assert_shape(
        torch.norm(matrix).shape,  # noqa: CITRINE(torchfix_deprecated_symbol_call)
        (),
    )

    tensor = torch.randn((2, 3, 4))
    assert_shape(
        torch.norm(  # noqa: CITRINE(torchfix_deprecated_symbol_call)
            tensor, dim=(0, -1)
        ).shape,
        (3,),
    )
    assert_shape(tensor.norm(dim=1, keepdim=True).shape, (2, 1, 4))


def test_dist_shapes() -> None:
    matrix = torch.randn((3, 4))
    assert_shape(torch.dist(matrix, torch.zeros((3, 4))).shape, ())
    assert_shape(matrix.dist(torch.zeros((3, 4))).shape, ())
    assert_shape(torch.dist(torch.ones((2, 1)), torch.ones(3)).shape, ())


def test_layer_norm_shape() -> None:
    tensor = torch.randn((2, 3, 4))
    weight = torch.randn((4,))
    assert_shape(F.layer_norm(tensor, weight.shape, weight).shape, (2, 3, 4))


def test_normalization_shapes() -> None:
    tensor = torch.randn((2, 3, 4, 5))
    for result in (
        F.batch_norm(tensor, torch.zeros(3), torch.ones(3)),
        F.instance_norm(tensor),
        F.layer_norm(tensor, (4, 5)),
        F.group_norm(tensor, 3),
        F.normalize(tensor),
        F.local_response_norm(tensor, 3),
    ):
        assert_shape(result.shape, (2, 3, 4, 5))


def test_norm_rejects_invalid_dimensions() -> None:
    matrix = torch.randn((2, 3))
    assert_shape(matrix.norm(dim=1).shape, (2,))

    with assert_raises(IndexError):
        # E: dimension out of range
        torch.norm(  # noqa: CITRINE(torchfix_deprecated_symbol_call)
            matrix, dim=2
        )

    with assert_raises(RuntimeError):
        # E: duplicate dimension
        torch.norm(  # noqa: CITRINE(torchfix_deprecated_symbol_call)
            matrix, dim=(0, -2)
        )


def test_dist_rejects_incompatible_shapes() -> None:
    matrix = torch.randn((2, 3))
    assert_shape(torch.dist(matrix, matrix).shape, ())

    with assert_raises(RuntimeError):
        # TODO: BUG: Reject incompatible `dist` input shapes statically.
        torch.dist(matrix, torch.ones((4, 5)))


if TYPE_CHECKING:

    def check_norm_shapes[Batch: IntTuple, N: IntVar](
        tensor: Tensor[[*Elements[Batch], N]], dim: int, keepdim: bool
    ) -> None:
        assert_type(tensor.norm(dim=-1), Tensor[Batch])
        assert_type(torch.norm(tensor, dim=-1), Tensor[Batch])  # noqa: CITRINE(torchfix_deprecated_symbol_call)
        assert_type(tensor.norm(dim=dim), Tensor[IntTuple])
        assert_type(tensor.norm(dim=-1, keepdim=keepdim), Tensor[IntTuple])
        assert_type(torch.dist(tensor, tensor), Tensor[[]])

    def check_layer_norm[Batch: IntTuple, N: IntVar](
        tensor: Tensor[[*Elements[Batch], N]], weight: Tensor[[N]]
    ) -> None:
        assert_type(
            F.layer_norm(tensor, weight.shape, weight),
            Tensor[[*Elements[Batch], N]],
        )

    def check_shape_preserving_normalization[Shape: IntTuple](
        tensor: Tensor[Shape],
    ) -> None:
        assert_type(F.batch_norm(tensor, None, None), Tensor[Shape])
        assert_type(F.instance_norm(tensor), Tensor[Shape])
        assert_type(F.group_norm(tensor, 1), Tensor[Shape])
        assert_type(F.normalize(tensor), Tensor[Shape])
        assert_type(F.local_response_norm(tensor, 3), Tensor[Shape])
