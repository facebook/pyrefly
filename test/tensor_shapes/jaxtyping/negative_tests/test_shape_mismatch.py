# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Negative test: direct shape mismatch on return type."""

from jaxtyping import Shaped
from torch import Tensor


def wrong_return_size(x: Shaped[Tensor, "batch 3"]) -> Shaped[Tensor, "batch 4"]:
    """Return type has size 4 but input has size 3 in last dim."""
    return x
