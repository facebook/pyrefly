# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import microtorch as torch
from microtorch import Tensor

features = torch.randn((1000, 3))
coefficients = torch.randn((3, 1))

predictions = features @ coefficients
gram_matrix = features.transpose() @ features
results = torch.concatenate((features, predictions), axis=1)

assert_type(features, Tensor[[1000, 3]])
assert_type(coefficients, Tensor[[3, 1]])
assert_type(predictions, Tensor[[1000, 1]])
assert_type(gram_matrix, Tensor[[3, 3]])
assert_type(results, Tensor[[1000, 4]])

if TYPE_CHECKING:
    wrong_coefficients = torch.randn((4, 1))
    features @ wrong_coefficients  # E: is not assignable
