# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_complex_fft_shapes() -> None:
    matrix = torch.randn((2, 3))
    assert_shape(torch.fft.fft(matrix).shape, (2, 3))
    assert_shape(torch.fft.ifft(matrix, n=None, dim=0, norm=None).shape, (2, 3))

    assert_shape(torch.fft.fft(matrix, n=5, dim=0).shape, (5, 3))
    assert_shape(torch.fft.ifft(matrix, n=4, dim=-1).shape, (2, 4))


def test_complex_fft_rejects_invalid_arguments() -> None:
    matrix = torch.randn((2, 3))
    assert_shape(torch.fft.fft(matrix).shape, (2, 3))

    with assert_raises(IndexError):
        torch.fft.fft(matrix, dim=2)  # E: FFT dimension out of range
    with assert_raises(IndexError):
        torch.fft.ifft(matrix, dim=-3)  # E: FFT dimension out of range

    # TODO: BUG: Reject nonpositive complex FFT lengths statically.
    with assert_raises(RuntimeError):
        torch.fft.fft(matrix, n=0)
    with assert_raises(RuntimeError):
        torch.fft.ifft(matrix, n=-1)

    scalar = torch.randn(())
    with assert_raises(IndexError):
        torch.fft.fft(scalar)  # E: FFT dimension out of range


if TYPE_CHECKING:

    def check_symbolic_complex_fft[N: IntVar, M: IntVar](
        input: Tensor[[N, M]], n: int, dim: int
    ) -> None:
        assert_type(torch.fft.fft(input), Tensor[[N, M]])
        assert_type(torch.fft.ifft(input, dim=0), Tensor[[N, M]])
        assert_type(torch.fft.fft(input, n=n, dim=0), Tensor[[int, M]])
        assert_type(torch.fft.ifft(input, n=n, dim=dim), Tensor[IntTuple])

    def check_gradual_complex_fft(input: Tensor) -> None:
        assert_type(torch.fft.fft(input), Tensor[IntTuple])
        assert_type(torch.fft.ifft(input), Tensor[IntTuple])
