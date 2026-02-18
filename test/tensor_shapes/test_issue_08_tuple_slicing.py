# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_tuple_slicing[B, T, NHeads, HeadDim](
    x: Tensor[B, T, NHeads, HeadDim],
) -> None:
    # Full size() works correctly
    full_size = x.size()
    assert_type(full_size, tuple[Dim[B], Dim[T], Dim[NHeads], Dim[HeadDim]])

    # Sliced size now preserves positional type information
    sliced_size = x.size()[:-1]
    assert_type(sliced_size, tuple[Dim[B], Dim[T], Dim[NHeads]])

    # Explicit size(dim) calls also work
    s0 = x.size(0)
    s1 = x.size(1)
    s2 = x.size(2)
    assert_type(s0, Dim[B])
    assert_type(s1, Dim[T])
    assert_type(s2, Dim[NHeads])


def test_reshape_with_slice[B, T, NHeads, HeadDim](
    x: Tensor[B, T, NHeads, HeadDim],
) -> None:
    # This pattern fails with tuple slicing
    # xshaped = x.float().reshape(*x.size()[:-1], -1, 2)

    # Workaround: use explicit size() calls
    xshaped = x.float().reshape(x.size(0), x.size(1), x.size(2), -1, 2)
    assert_type(xshaped, Tensor[B, T, NHeads, HeadDim // 2, 2])
