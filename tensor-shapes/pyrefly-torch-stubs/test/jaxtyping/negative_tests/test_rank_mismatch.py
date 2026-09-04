# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Negative test: rank mismatch on return type."""

from jaxtyping import Shaped
from shape_extensions import static_jaxtyping
from torch import Tensor


@static_jaxtyping("batch")
def wrong_rank(x: Shaped[Tensor, "batch 3"]) -> Shaped[Tensor, "batch 3 4"]:
    """Return type has rank 3 but input has rank 2."""
    # E: Returned type `Tensor[[batch, 3]]` is not assignable
    #    to declared return type `Tensor[[batch, 3, 4]]`
    return x
