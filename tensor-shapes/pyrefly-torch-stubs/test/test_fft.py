# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

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
    from shape_extensions import Int

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


def test_real_fft_shapes() -> None:
    tensor = torch.randn((4, 10, 6))
    assert_shape(torch.fft.rfft(tensor, dim=-2).shape, (4, 6, 6))
    assert_shape(torch.fft.rfft(tensor, n=8, dim=0).shape, (5, 10, 6))
    assert_shape(torch.fft.ihfft(tensor, n=8, dim=0).shape, (5, 10, 6))

    assert_shape(torch.fft.irfft(tensor, n=12, dim=1).shape, (4, 12, 6))
    assert_shape(torch.fft.hfft(tensor, n=12, dim=1).shape, (4, 12, 6))
    assert_shape(torch.fft.irfft(tensor, n=None, dim=1).shape, (4, 18, 6))
    assert_shape(torch.fft.hfft(tensor, dim=0).shape, (6, 10, 6))
    assert_shape(torch.fft.ihfft(tensor, n=None, dim=1).shape, (4, 6, 6))


def test_real_fft_rejects_invalid_arguments() -> None:
    matrix = torch.randn((3, 4))
    assert_shape(torch.fft.rfft(matrix).shape, (3, 3))

    with assert_raises(IndexError):
        torch.fft.rfft(matrix, dim=2)  # E: FFT dimension out of range
    with assert_raises(IndexError):
        torch.fft.irfft(matrix, dim=-3)  # E: FFT dimension out of range
    with assert_raises(IndexError):
        torch.fft.hfft(matrix, n=8, dim=2)  # E: FFT dimension out of range
    with assert_raises(IndexError):
        torch.fft.ihfft(matrix, n=8, dim=-3)  # E: FFT dimension out of range

    # TODO: BUG: Reject nonpositive real FFT lengths statically.
    with assert_raises(RuntimeError):
        torch.fft.rfft(matrix, n=0)
    with assert_raises(RuntimeError):
        torch.fft.irfft(matrix, n=-1)


if TYPE_CHECKING:

    def check_symbolic_real_fft[N: IntVar](input: Tensor[[3, 7]], n: Int[N]) -> None:
        assert_type(torch.fft.rfft(input, n=n, dim=0), Tensor[[N // 2 + 1, 7]])
        assert_type(torch.fft.ihfft(input, n=n, dim=0), Tensor[[N // 2 + 1, 7]])
        assert_type(torch.fft.irfft(input, n=n, dim=-1), Tensor[[3, N]])
        assert_type(torch.fft.hfft(input, n=n, dim=-1), Tensor[[3, N]])

    def check_gradual_real_fft(
        input: Tensor[[4, 10, 6]],
        bare: Tensor,
        n: int,
        optional_n: int | None,
        dim: int,
        value: Any,
    ) -> None:
        assert_type(torch.fft.rfft(input, n=n, dim=0), Tensor[[int, 10, 6]])
        assert_type(torch.fft.irfft(input, n=n, dim=1), Tensor[[4, int, 6]])
        assert_type(torch.fft.rfft(input, dim=dim), Tensor[IntTuple])
        assert_type(torch.fft.irfft(input, n=12, dim=dim), Tensor[IntTuple])
        assert_type(torch.fft.rfft(input, n=value), Tensor)
        assert_type(torch.fft.irfft(input, dim=value), Tensor)
        assert_type(torch.fft.rfft(bare), Tensor)
        # TODO: BUG: Preserve known axes for optional transform lengths.
        assert_type(torch.fft.hfft(input, n=optional_n, dim=1), Tensor)
