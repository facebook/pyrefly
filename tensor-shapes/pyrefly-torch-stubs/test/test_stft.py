# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, Int, IntVar
from torch import Tensor


def test_stft_shapes() -> None:
    signal = torch.randn(16)
    batch = torch.randn((3, 16))
    window = torch.hann_window(8)
    complex_signal = torch.view_as_complex(torch.randn((16, 2)))

    assert_shape(
        torch.stft(
            complex_signal,
            8,
            hop_length=4,
            window=window,
            center=False,
        ).shape,
        (int, int),
        runtime=(8, 3),
    )
    assert_shape(
        torch.stft(
            signal,
            8,
            hop_length=4,
            window=window,
            center=False,
            onesided=True,
            return_complex=True,
        ).shape,
        (5, int),
        runtime=(5, 3),
    )
    assert_shape(
        torch.stft(
            signal,
            8,
            hop_length=4,
            window=window,
            center=False,
            onesided=False,
            return_complex=False,
        ).shape,
        (8, int, 2),
        runtime=(8, 3, 2),
    )
    assert_shape(
        torch.stft(
            batch,
            8,
            hop_length=4,
            window=window,
            center=False,
            onesided=True,
            return_complex=True,
            align_to_window=False,
        ).shape,
        (3, 5, int),
        runtime=(3, 5, 3),
    )
    assert_shape(
        torch.stft(
            signal,
            8,
            hop_length=4,
            window=window,
            center=False,
            onesided=None,
            return_complex=True,
        ).shape,
        (int, int),
        runtime=(5, 3),
    )


if TYPE_CHECKING:

    def check_symbolic_stft[N: IntVar, F: IntVar](
        signal: Tensor[[N]], batch: Tensor[[3, N]], n_fft: Int[F]
    ) -> None:
        assert_type(
            torch.stft(signal, n_fft, onesided=True, return_complex=True),
            Tensor[[F // 2 + 1, int]],
        )
        assert_type(
            torch.stft(signal, n_fft, onesided=False, return_complex=False),
            Tensor[[F, int, 2]],
        )
        assert_type(
            torch.stft(batch, n_fft, onesided=True, return_complex=True),
            Tensor[[3, F // 2 + 1, int]],
        )
