# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

from shape_extensions import IntVar


if TYPE_CHECKING:
    from shape_extensions import Int
    from torch import Tensor


def test_starred_unpack[
    B: IntVar,
    T: IntVar,
    NHeads: IntVar,
    HeadDim: IntVar,
](
    x: Tensor[[B, T, NHeads, HeadDim]],
) -> None:
    # First check that tuple slicing works
    sizes = x.size()
    assert_type(sizes, tuple[Int[B], Int[T], Int[NHeads], Int[HeadDim]])

    sliced = x.size()[:-1]
    assert_type(sliced, tuple[Int[B], Int[T], Int[NHeads]])

    # Starred unpacking now preserves element types
    result = x.float().reshape(*sliced, -1, 2)
    assert_type(result, Tensor[[B, T, NHeads, HeadDim // 2, 2]])
