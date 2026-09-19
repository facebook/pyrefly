# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import Tensor


def check_invalid_cosine_embedding_shapes(
    vector: Tensor[[3]],
    matrix: Tensor[[2, 3]],
    rank_four: Tensor[[2, 3, 4, 5]],
    scalar_target: Tensor[[]],
    vector_target: Tensor[[2]],
    wrong_length_target: Tensor[[3]],
) -> None:
    F.cosine_embedding_loss(  # E: cosine_embedding_loss requires 1D or 2D inputs
        rank_four, rank_four, vector_target, reduction="none"
    )
    F.cosine_embedding_loss(  # E: cosine_embedding_loss requires a scalar target for 1D inputs
        vector, vector, vector_target, reduction="none"
    )
    F.cosine_embedding_loss(  # E: cosine_embedding_loss requires a 1D target for 2D inputs
        matrix, matrix, scalar_target, reduction="none"
    )
    F.cosine_embedding_loss(  # E: Cannot broadcast dimension Int[2] with dimension Int[3] at position 0
        matrix, matrix, wrong_length_target, reduction="none"
    )
    scalar: Tensor[[]] = torch.randn(())
    F.triplet_margin_loss(  # E: triplet_margin_loss requires at least 1D input
        scalar, scalar, scalar, reduction="none"
    )
