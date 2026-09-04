# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_item_scalar(x: Tensor[[]]) -> None:
    assert_type(x.item(), float | int)


def test_item_gradual(x: Tensor) -> None:
    assert_type(x.item(), float | int)
