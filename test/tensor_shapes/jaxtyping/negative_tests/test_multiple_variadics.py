# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Negative test: multiple variadic specifiers (both * and ...) in one annotation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Shaped
    from torch import Tensor


def test_multiple_variadics(
    x: Shaped[Tensor, "*batch ... 3"],
) -> None:
    pass
