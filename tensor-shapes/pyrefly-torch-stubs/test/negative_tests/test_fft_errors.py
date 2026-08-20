# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.fft
from torch import Tensor


def test_fft_axis_errors(x: Tensor[[3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: FFT dimension out of range
    torch.fft.rfft(x, dim=2)
    # E: Cannot evaluate type-level shape DSL call: FFT dimension out of range
    torch.fft.irfft(x, dim=-3)
