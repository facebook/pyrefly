# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape
from torch import nn


def test_factory_kwargs_canonicalizes_options() -> None:
    assert nn.factory_kwargs(None) == {}
    options = nn.factory_kwargs(
        {"device": "cpu", "factory_kwargs": {"dtype": torch.float32}}
    )
    assert options == {"device": "cpu", "dtype": torch.float32}
    assert_shape(torch.empty((2, 3), **options).shape, (2, 3))

    with assert_raises(TypeError):
        nn.factory_kwargs(
            {"dtype": torch.float32, "factory_kwargs": {"dtype": torch.float64}}
        )
    with assert_raises(TypeError):
        nn.factory_kwargs({"unexpected": 1})


if TYPE_CHECKING:

    def check_factory_kwargs_type() -> None:
        assert_type(nn.factory_kwargs(None), dict[str, Any])
        assert_type(nn.factory_kwargs({"dtype": torch.float32}), dict[str, Any])
