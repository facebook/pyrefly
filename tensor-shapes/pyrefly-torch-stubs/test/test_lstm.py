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


def test_lstm_shapes() -> None:
    lstm = nn.LSTM(6, 4, batch_first=True)
    output, (hidden, cell) = lstm(torch.ones((2, 5, 6)))
    assert_shape(output.shape, (2, 5, 4))
    assert_shape(hidden.shape, (1, 2, 4))
    assert_shape(cell.shape, (1, 2, 4))

    bidirectional = nn.LSTM(6, 4, num_layers=3, bidirectional=True)
    output, (hidden, cell) = bidirectional(torch.ones((5, 2, 6)))
    assert_shape(output.shape, (5, 2, 8))
    assert_shape(hidden.shape, (6, 2, 4))
    assert_shape(cell.shape, (6, 2, 4))


def test_lstm_cell_shapes() -> None:
    hidden, cell = nn.LSTMCell(6, 4)(torch.ones((2, 6)))
    assert_shape(hidden.shape, (2, 4))
    assert_shape(cell.shape, (2, 4))


def test_lstm_rejects_invalid_input_features() -> None:
    lstm = nn.LSTM(6, 4, batch_first=True)
    with assert_raises(RuntimeError):
        # E: input feature size does not match input_size
        lstm(torch.ones((2, 5, 7)))


if TYPE_CHECKING:

    def check_lstm_return_structure() -> None:
        output, (hidden, cell) = nn.LSTM(6, 4, batch_first=True)(torch.ones((2, 5, 6)))
        assert_type(output, Tensor[[2, 5, 4]])
        assert_type(hidden, Tensor[[1, 2, 4]])
        assert_type(cell, Tensor[[1, 2, 4]])

    def check_symbolic_lstm[B: IntVar, T: IntVar, N: IntVar, H: IntVar](
        batch: Int[B], steps: Int[T], inputs: Int[N], hidden_size: Int[H]
    ) -> None:
        output, (hidden, cell) = nn.LSTM(inputs, hidden_size, batch_first=True)(
            torch.ones((batch, steps, inputs))
        )
        assert_type(output, Tensor[[B, T, H]])
        assert_type(hidden, Tensor[[1, B, H]])
        assert_type(cell, Tensor[[1, B, H]])
