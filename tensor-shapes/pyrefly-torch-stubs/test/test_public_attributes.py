# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape

if TYPE_CHECKING:

    def identity(value: int) -> int: ...


def test_public_attributes() -> None:
    assert_type(torch.complex64, torch.dtype)
    assert_type(torch.complex128, torch.dtype)
    assert_type(torch.strided, torch.layout)
    assert_type(torch.get_num_threads(), int)
    assert_type(torch.get_float32_matmul_precision(), str)
    assert_type(torch.are_deterministic_algorithms_enabled(), bool)
    assert_shape(torch.ones((2, 3), dtype=torch.complex128).shape, (2, 3))

    _ = (
        torch.compile,
        torch.set_grad_enabled,
        torch.set_default_dtype,
    )

    if TYPE_CHECKING:
        assert_type(torch.compile(identity)(1), int)
        assert_type(torch.compile()(identity)(1), int)
