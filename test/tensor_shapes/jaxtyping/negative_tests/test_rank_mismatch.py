# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Negative test: rank mismatch on return type."""

from jaxtyping import Shaped
from torch import Tensor


def wrong_rank(x: Shaped[Tensor, "batch 3"]) -> Shaped[Tensor, "batch 3 4"]:
    """Return type has rank 3 but input has rank 2."""
    return x
