# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from shape_extensions import Int, IntVar
from torch import Tensor


x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

torch.moveaxis(x, 3, 0)  # E: source dimension out of range
x.movedim(0, -4)  # E: destination dimension out of range
torch.moveaxis(x, 0, (1,))  # E: No matching overload

torch.movedim(x, (0,), (1, 2))  # E: source and destination must have equal length
x.moveaxis((0, 0), (1, 2))  # E: source dimensions must be unique
torch.moveaxis(x, (0, -3), (1, 2))  # E: source dimensions must be unique
torch.movedim(x, (0, 1), (2, -1))  # E: destination dimensions must be unique
torch.moveaxis(x, (3,), (0,))  # E: source dimension out of range
x.movedim((0,), (-4,))  # E: destination dimension out of range
torch.movedim(x, (0,), 1)  # E: No matching overload
x.moveaxis(0, (1,))  # E: No matching overload

scalar: Tensor[[]] = torch.tensor(1)

# Only 0 and -1 name the implicit axis of a rank-0 tensor; source is reported
# before destination when both are invalid.
torch.moveaxis(scalar, 1, 0)  # E: source dimension out of range
scalar.movedim(0, -2)  # E: destination dimension out of range
torch.movedim(scalar, 2, 2)  # E: source dimension out of range

# The tuple overloads apply the same rank-0 rule. Length is checked before the
# axes, and because every legal rank-0 axis denotes the one implicit axis, a
# two-axis move always collides on the source.
torch.movedim(scalar, (0,), (1, 0))  # E: source and destination must have equal length
scalar.movedim((1,), (0,))  # E: source dimension out of range
torch.moveaxis(scalar, (0,), (-2,))  # E: destination dimension out of range
scalar.moveaxis((0, -1), (0, -1))  # E: source dimensions must be unique


def check_symbolic_axes[N: IntVar](axis: Int[N]) -> None:
    # An unknown source must not hide a concrete destination error, or vice versa.
    torch.movedim(x, (axis,), (3,))  # E: destination dimension out of range
    torch.movedim(x, (3,), (axis,))  # E: source dimension out of range
    torch.movedim(x, (axis, 1), (0, 0))  # E: destination dimensions must be unique
    torch.movedim(x, (0, 0), (axis, 1))  # E: source dimensions must be unique
    torch.movedim(scalar, (axis,), (2,))  # E: destination dimension out of range
    torch.movedim(scalar, (0, -1), (axis, axis))  # E: source dimensions must be unique
    torch.movedim(  # E: destination dimensions must be unique
        scalar, (axis, 0), (0, -1)
    )
