# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that mixing native and jaxtyping tensor syntax in the same function is an error."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def mixed_syntax(
    x: Float[Tensor, "batch 3"],
) -> Tensor[3]:
    return x
