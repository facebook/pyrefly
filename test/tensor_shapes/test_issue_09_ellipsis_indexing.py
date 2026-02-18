# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_ellipsis_indexing[B, T, NHeads, HeadDim](
    x: Tensor[B, T, NHeads, HeadDim, 2],
) -> None:
    # Ellipsis indexing now works correctly
    result = x[..., 0]
    assert_type(result, Tensor[B, T, NHeads, HeadDim])

    # Explicit colon slices also work
    result_explicit = x[:, :, :, :, 0]
    assert_type(result_explicit, Tensor[B, T, NHeads, HeadDim])


def test_ellipsis_multiple_indices[B, T, NHeads, HeadDim](
    x: Tensor[B, T, NHeads, HeadDim, 2],
) -> None:
    # Ellipsis with multiple integer indices
    result = x[..., 0, 0]
    assert_type(result, Tensor[B, T, NHeads])

    # Explicit colon slices also work
    result_explicit = x[:, :, :, 0, 0]
    assert_type(result_explicit, Tensor[B, T, NHeads])
