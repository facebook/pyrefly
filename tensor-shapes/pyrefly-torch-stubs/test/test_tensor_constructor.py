# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

from shape_extensions import IntTuple, RegularNestedList
from torch import Tensor


def test_regular_nested_list_constructor[Shape: IntTuple](
    data: RegularNestedList[Shape, int],
) -> None:
    assert_type(Tensor(data), Tensor[Shape])
