# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from torch import Tensor
from torch.nn import init


def test_deprecated_init_aliases_preserve_shape() -> None:
    value = torch.zeros(2, 3)
    assert_type(init.constant(value, 0), Tensor[[2, 3]])
    assert_type(init.eye(value), Tensor[[2, 3]])
    assert_type(init.kaiming_normal(value), Tensor[[2, 3]])
    assert_type(init.kaiming_uniform(value), Tensor[[2, 3]])
    assert_type(init.normal(value), Tensor[[2, 3]])
    assert_type(init.orthogonal(value), Tensor[[2, 3]])
    assert_type(init.sparse(value, 0.5), Tensor[[2, 3]])
    assert_type(init.uniform(value), Tensor[[2, 3]])
    assert_type(init.xavier_normal(value), Tensor[[2, 3]])
    assert_type(init.xavier_uniform(value), Tensor[[2, 3]])
