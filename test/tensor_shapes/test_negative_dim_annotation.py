# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# -1 + L in a return type annotation
def test_neg_plus[B, L](x: Tensor[B, 3, L]) -> Tensor[B, 3, (-1 + L)]:
    return x  # type: ignore


# L - 1 (should be equivalent)
def test_sub[B, L](x: Tensor[B, 3, L]) -> Tensor[B, 3, (L - 1)]:
    return x  # type: ignore


# Both should produce the same canonical form
def test_equivalence[B, L](x: Tensor[B, 3, (-1 + L)], y: Tensor[B, 3, (L - 1)]) -> None:
    assert_type(x, Tensor[B, 3, (L - 1)])
    assert_type(y, Tensor[B, 3, (-1 + L)])
