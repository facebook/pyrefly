# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def test_tensordot_dims_errors(a: Tensor[[2, 3]], b: Tensor[[4, 5]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: tensordot dims must be non-negative
    torch.tensordot(a, b, dims=-1)
    # E: Cannot evaluate type-level shape DSL call: tensordot dims exceeds input rank
    torch.tensordot(a, b, dims=3)
    # E: Cannot evaluate type-level shape DSL call: tensordot dims exceeds input rank
    torch.tensordot(a, b, dims=4)
