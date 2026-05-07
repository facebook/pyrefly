# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_complex_expression[HeadDim](n_elem: Dim[HeadDim], base: int = 10000) -> None:
    # This is the exact pattern from precompute_freqs_cis
    # If Issue 6 is still present, freqs will be Tensor[Any] instead of Tensor[HeadDim // 2]
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    assert_type(freqs, Tensor[HeadDim // 2])


def test_intermediate_steps[HeadDim](n_elem: Dim[HeadDim]) -> None:
    # Break down the complex expression to find where type info is lost
    step1 = torch.arange(0, n_elem, 2)
    assert_type(step1, Tensor[HeadDim // 2])

    step2 = step1[: (n_elem // 2)]
    assert_type(step2, Tensor[HeadDim // 2])

    step3 = step2.float()
    assert_type(step3, Tensor[HeadDim // 2])

    step4 = step3 / n_elem
    assert_type(step4, Tensor[HeadDim // 2])
