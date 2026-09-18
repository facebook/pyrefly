# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, IntTuple


def test_interpolate_size_and_scale() -> None:
    signal = torch.randn((2, 3, 8))
    image = torch.randn((2, 3, 8, 12))
    volume = torch.randn((2, 3, 4, 8, 12))

    assert_shape(F.interpolate(signal, 4).shape, (2, 3, 4))
    assert_shape(F.interpolate(image, (4, 6)).shape, (2, 3, 4, 6))
    # TODO: BUG: Preserve shape for tuple-valued scale factors.
    assert_shape(
        F.interpolate(volume, scale_factor=(2.0, 0.5, 0.5)).shape,
        IntTuple,
        runtime=(2, 3, 8, 4, 6),
    )
    # TODO: BUG: Preserve output sizes through `nn.Upsample`.
    assert_shape(
        nn.Upsample(size=(5, 7))(image).shape,
        IntTuple,
        runtime=(2, 3, 5, 7),
    )


def test_interpolate_rejects_invalid_rank_and_controls() -> None:
    image = torch.randn((2, 3, 8, 8))
    assert_shape(image.shape, (2, 3, 8, 8))

    with assert_raises(NotImplementedError):
        # E: interpolate requires rank 3, 4, or 5
        F.interpolate(torch.randn((3, 8)), 2)
    with assert_raises(ValueError):
        # E: size must match the spatial rank
        F.interpolate(image, (2,))
    with assert_raises(ValueError):
        # E: scale_factor must match
        F.interpolate(image, scale_factor=(2,))
    with assert_raises(RuntimeError):
        # E: interpolate size must be positive
        F.interpolate(image, (0, 2))
    with assert_raises(RuntimeError):
        # E: scale_factor must be positive
        F.interpolate(image, scale_factor=(2, -1))
    with assert_raises(ValueError):
        # E: interpolate requires size or scale_factor
        F.interpolate(image)
    with assert_raises(ValueError):
        # E: accepts only one of size or scale_factor
        F.interpolate(image, 2, 2)


def test_upsample_rejects_invalid_arguments() -> None:
    image = torch.randn((2, 3, 8, 8))
    assert_shape(image.shape, (2, 3, 8, 8))

    with assert_raises(ValueError):
        # E: interpolate requires size or scale_factor
        nn.Upsample()(image)
    with assert_raises(ValueError):
        # E: accepts only one
        nn.Upsample(size=2, scale_factor=2)(image)
    nn.Upsample(size=1.5)  # E: No matching overload
