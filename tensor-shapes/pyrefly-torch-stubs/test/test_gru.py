# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import assert_raises, assert_shape, Int, IntVar
from torch import Tensor


def test_gru_batch_first_shapes() -> None:
    gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True)
    output, state = gru(torch.ones((4, 10, 256)))
    assert_shape(output.shape, (4, 10, 128))
    assert_shape(state.shape, (1, 4, 128))

    bidirectional = nn.GRU(64, 32, bidirectional=True, batch_first=True)
    output, state = bidirectional(torch.ones((8, 5, 64)))
    assert_shape(output.shape, (8, 5, 64))
    assert_shape(state.shape, (2, 8, 32))

    multilayer = nn.GRU(128, 64, num_layers=3, batch_first=True)
    output, state = multilayer(torch.ones((2, 20, 128)))
    assert_shape(output.shape, (2, 20, 64))
    assert_shape(state.shape, (3, 2, 64))


def test_gru_sequence_first_shapes() -> None:
    gru = nn.GRU(6, 4)
    output, state = gru(torch.ones((5, 2, 6)))
    assert_shape(output.shape, (5, 2, 4))
    assert_shape(state.shape, (1, 2, 4))


def test_gru_unbatched_shapes() -> None:
    gru = nn.GRU(6, 4)
    output, state = gru(torch.ones((5, 6)))
    assert_shape(output.shape, (5, 4))
    assert_shape(state.shape, (1, 4))


def test_gru_rejects_invalid_input_features() -> None:
    gru = nn.GRU(6, 4, batch_first=True)
    output, _ = gru(torch.ones((2, 5, 6)))
    assert_shape(output.shape, (2, 5, 4))

    with assert_raises(RuntimeError):
        # E: input feature size does not match input_size
        gru(torch.ones((2, 5, 7)))


if TYPE_CHECKING:

    def check_symbolic_gru[B: IntVar, T: IntVar, N: IntVar, H: IntVar](
        batch: Int[B], steps: Int[T], inputs: Int[N], hidden: Int[H]
    ) -> None:
        gru = nn.GRU(inputs, hidden, batch_first=True)
        output, state = gru(torch.ones((batch, steps, inputs)))
        assert_type(output, Tensor[[B, T, H]])
        assert_type(state, Tensor[[1, B, H]])
