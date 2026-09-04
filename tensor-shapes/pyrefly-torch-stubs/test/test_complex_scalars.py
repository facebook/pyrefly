# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from shape_extensions import assert_shape


def test_complex_scalar_operations() -> None:
    value = torch.tensor([1 + 0j, 2 + 0j])

    assert_shape((1j + value).shape, (2,))
    assert_shape((1j - value).shape, (2,))
    assert_shape((1j * value).shape, (2,))
    assert_shape((1j / value).shape, (2,))
    assert_shape((1j**value).shape, (2,))

    value[0] = 1j
    _ = value.new_full((2,), 1j)
