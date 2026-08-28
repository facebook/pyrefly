# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

torch.moveaxis(x, 3, 0)  # E: source dimension out of range
x.movedim(0, -4)  # E: destination dimension out of range
torch.moveaxis(x, 0, (1,))  # E: No matching overload

scalar: Tensor[[]] = torch.tensor(1)

# Only 0 and -1 name the implicit axis of a rank-0 tensor; source is reported
# before destination when both are invalid.
torch.moveaxis(scalar, 1, 0)  # E: source dimension out of range
scalar.movedim(0, -2)  # E: destination dimension out of range
torch.movedim(scalar, 2, 2)  # E: source dimension out of range
