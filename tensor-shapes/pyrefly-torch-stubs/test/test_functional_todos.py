# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import assert_shape, IntVar
from torch import Tensor


def test_adaptive_max_pool_with_indices_shapes() -> None:
    sequence = torch.randn((2, 3, 12))
    values, indices = F.adaptive_max_pool1d_with_indices(sequence, (5,))
    assert_shape(values.shape, (2, 3, 5))
    assert_shape(indices.shape, (2, 3, 5))

    image = torch.randn((2, 3, 8, 10))
    values, indices = F.adaptive_max_pool2d_with_indices(image, (4, 6))
    assert_shape(values.shape, (2, 3, 4, 6))
    assert_shape(indices.shape, (2, 3, 4, 6))
    values, indices = F.adaptive_max_pool2d_with_indices(image, (4, None))
    assert tuple(values.shape) == (2, 3, 4, 10)
    assert tuple(indices.shape) == (2, 3, 4, 10)

    volume = torch.randn((2, 3, 8, 10, 12))
    values, indices = F.adaptive_max_pool3d_with_indices(volume, (2, 3, 4))
    assert_shape(values.shape, (2, 3, 2, 3, 4))
    assert_shape(indices.shape, (2, 3, 2, 3, 4))


def test_max_pool_with_indices_shapes() -> None:
    sequence = torch.randn((2, 3, 12))
    values, indices = F.max_pool1d_with_indices(sequence, 2)
    assert_shape(values.shape, (2, 3, 6))
    assert_shape(indices.shape, (2, 3, 6))

    image = torch.randn((2, 3, 8, 12))
    values, indices = F.max_pool2d_with_indices(
        image, (2, 3), stride=(2, 3), return_indices=False
    )
    assert_shape(values.shape, (2, 3, 4, 4))
    assert_shape(indices.shape, (2, 3, 4, 4))

    volume = torch.randn((2, 3, 8, 10, 12))
    values, indices = F.max_pool3d_with_indices(volume, (2, 2, 3), (2, 2, 3))
    assert_shape(values.shape, (2, 3, 4, 5, 4))
    assert_shape(indices.shape, (2, 3, 4, 5, 4))


def test_max_unpool_shapes() -> None:
    sequence = torch.randn((2, 3, 11))
    values, indices = F.max_pool1d_with_indices(sequence, 2)
    assert_shape(
        F.max_unpool1d(values, indices, 2, output_size=(11,)).shape, (2, 3, 11)
    )

    image = torch.randn((2, 3, 9, 11))
    values, indices = F.max_pool2d_with_indices(image, 2)
    assert_shape(
        F.max_unpool2d(values, indices, 2, output_size=(2, 3, 9, 11)).shape,
        (2, 3, 9, 11),
    )
    even_values, even_indices = F.max_pool2d_with_indices(torch.randn((2, 3, 8, 10)), 2)
    assert tuple(F.max_unpool2d(even_values, even_indices, 2).shape) == (2, 3, 8, 10)

    volume = torch.randn((2, 3, 9, 11, 13))
    values, indices = F.max_pool3d_with_indices(volume, 2)
    assert_shape(
        F.max_unpool3d(values, indices, 2, output_size=(9, 11, 13)).shape,
        (2, 3, 9, 11, 13),
    )


def test_lp_pool_shapes() -> None:
    sequence = torch.randn((2, 3, 12))
    assert_shape(F.lp_pool1d(sequence, 2, 3, stride=2, ceil_mode=True).shape, (2, 3, 6))

    image = torch.randn((2, 3, 8, 9))
    assert_shape(F.lp_pool2d(image, 2, (2, 3), stride=(2, 3)).shape, (2, 3, 4, 3))

    volume = torch.randn((2, 3, 8, 10, 12))
    assert_shape(
        F.lp_pool3d(volume, 2, (2, 2, 3), stride=(2, 2, 3)).shape,
        (2, 3, 4, 5, 4),
    )


if TYPE_CHECKING:

    def check_adaptive_max_pool_with_indices_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 12]],
        image: Tensor[[B, 3, 8, 10]],
        volume: Tensor[[B, 3, 8, 10, 12]],
        output_size: int,
    ) -> None:
        assert_type(
            F.adaptive_max_pool1d_with_indices(sequence, 5),
            tuple[Tensor[[B, 3, 5]], Tensor[[B, 3, 5]]],
        )
        assert_type(
            F.adaptive_max_pool2d_with_indices(image, (4, 6)),
            tuple[Tensor[[B, 3, 4, 6]], Tensor[[B, 3, 4, 6]]],
        )
        assert_type(
            F.adaptive_max_pool3d_with_indices(volume, 2),
            tuple[Tensor[[B, 3, 2, 2, 2]], Tensor[[B, 3, 2, 2, 2]]],
        )
        assert_type(
            F.adaptive_max_pool2d_with_indices(image, output_size),
            tuple[Tensor[[B, 3, int, int]], Tensor[[B, 3, int, int]]],
        )
        assert_type(
            F.adaptive_max_pool2d_with_indices(image, (4, None)),
            tuple[Tensor[[B, 3, int, int]], Tensor[[B, 3, int, int]]],
        )

    def check_max_pool_with_indices_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 12]],
        image: Tensor[[B, 3, 8, 12]],
        volume: Tensor[[B, 3, 8, 10, 12]],
    ) -> None:
        assert_type(
            F.max_pool1d_with_indices(sequence, 2),
            tuple[Tensor[[B, 3, 6]], Tensor[[B, 3, 6]]],
        )
        assert_type(
            F.max_pool2d_with_indices(image, (2, 3), stride=(2, 3)),
            tuple[Tensor[[B, 3, 4, 4]], Tensor[[B, 3, 4, 4]]],
        )
        assert_type(
            F.max_pool3d_with_indices(volume, (2, 2, 3), (2, 2, 3)),
            tuple[Tensor[[B, 3, 4, 5, 4]], Tensor[[B, 3, 4, 5, 4]]],
        )

    def check_max_unpool_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 5]],
        sequence_indices: Tensor[[B, 3, 5]],
        image: Tensor[[B, 3, 4, 5]],
        image_indices: Tensor[[B, 3, 4, 5]],
        volume: Tensor[[B, 3, 4, 5, 6]],
        volume_indices: Tensor[[B, 3, 4, 5, 6]],
        output_size: tuple[int, ...],
    ) -> None:
        assert_type(
            F.max_unpool1d(sequence, sequence_indices, 2, output_size=(11,)),
            Tensor[[B, 3, 11]],
        )
        assert_type(
            F.max_unpool2d(image, image_indices, 2, output_size=(B, 3, 9, 11)),
            Tensor[[B, 3, 9, 11]],
        )
        assert_type(
            F.max_unpool3d(volume, volume_indices, 2, output_size=(9, 11, 13)),
            Tensor[[B, 3, 9, 11, 13]],
        )
        assert_type(
            F.max_unpool2d(image, image_indices, 2),
            Tensor[[B, 3, int, int]],
        )
        assert_type(
            F.max_unpool3d(volume, volume_indices, 2, output_size=output_size),
            Tensor[[B, 3, int, int, int]],
        )

    def check_lp_pool_shapes[B: IntVar](
        sequence: Tensor[[B, 3, 12]],
        image: Tensor[[B, 3, 8, 9]],
        volume: Tensor[[B, 3, 8, 10, 12]],
    ) -> None:
        assert_type(
            F.lp_pool1d(sequence, 2, 3, stride=2, ceil_mode=True),
            Tensor[[B, 3, 6]],
        )
        assert_type(
            F.lp_pool2d(image, 2, (2, 3), stride=(2, 3)),
            Tensor[[B, 3, 4, 3]],
        )
        assert_type(
            F.lp_pool3d(volume, 2, (2, 2, 3), stride=(2, 2, 3)),
            Tensor[[B, 3, 4, 5, 4]],
        )
