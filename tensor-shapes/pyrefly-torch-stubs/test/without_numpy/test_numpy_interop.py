# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, cast, reveal_type

import torch
from shape_extensions import assert_shape, IntTuple


def test_numpy_conversions_remain_available_without_numpy() -> None:
    tensor = torch.zeros((2, 3))
    reveal_type(tensor)  # E: revealed type: Tensor[[2, 3]]
    reveal_type(tensor.numpy())  # E: revealed type: Unknown
    assert_shape(tensor.shape, (2, 3))
    assert_shape(tensor.numpy().shape, IntTuple)

    unknown_array = cast(Any, None)
    converted = torch.from_numpy(unknown_array)
    reveal_type(converted)  # E: revealed type: Tensor
    assert_shape(converted.shape, IntTuple)
