# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from shape_extensions import assert_shape, IntVar
from torch import Tensor


def test_dropout_shapes() -> None:
    tensor = torch.ones((2, 3, 4))
    for result in (
        nn.Dropout(p=0.0)(tensor),
        nn.AlphaDropout(p=0.0)(tensor),
        F.dropout(tensor, p=0.0),
        F.alpha_dropout(tensor, p=0.0),
        F.feature_alpha_dropout(tensor, p=0.0),
    ):
        assert_shape(result.shape, (2, 3, 4))


def test_channel_dropout_shapes() -> None:
    vector_batch = torch.ones((2, 3, 4))
    assert_shape(nn.Dropout1d(p=0.0)(vector_batch).shape, (2, 3, 4))
    assert_shape(F.dropout1d(vector_batch, p=0.0).shape, (2, 3, 4))

    image_batch = torch.ones((2, 3, 4, 5))
    assert_shape(nn.Dropout2d(p=0.0)(image_batch).shape, (2, 3, 4, 5))
    assert_shape(F.dropout2d(image_batch, p=0.0).shape, (2, 3, 4, 5))

    volume_batch = torch.ones((2, 3, 4, 5, 6))
    assert_shape(nn.Dropout3d(p=0.0)(volume_batch).shape, (2, 3, 4, 5, 6))
    assert_shape(F.dropout3d(volume_batch, p=0.0).shape, (2, 3, 4, 5, 6))


if TYPE_CHECKING:

    def check_symbolic_dropout[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> None:
        assert_type(nn.Dropout()(x), Tensor[[N, M]])
        assert_type(nn.AlphaDropout()(x), Tensor[[N, M]])
        assert_type(F.dropout(x), Tensor[[N, M]])
        assert_type(F.alpha_dropout(x), Tensor[[N, M]])
        assert_type(F.feature_alpha_dropout(x), Tensor[[N, M]])
