# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_starred_unpack[B, T, NHeads, HeadDim](
    x: Tensor[B, T, NHeads, HeadDim],
) -> None:
    # First check that tuple slicing works
    sizes = x.size()
    assert_type(sizes, tuple[Dim[B], Dim[T], Dim[NHeads], Dim[HeadDim]])

    sliced = x.size()[:-1]
    assert_type(sliced, tuple[Dim[B], Dim[T], Dim[NHeads]])

    # Starred unpacking now preserves element types
    result = x.float().reshape(*sliced, -1, 2)
    assert_type(result, Tensor[B, T, NHeads, HeadDim // 2, 2])
