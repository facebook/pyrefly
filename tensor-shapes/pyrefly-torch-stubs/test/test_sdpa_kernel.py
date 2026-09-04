# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

from shape_extensions import assert_raises
from torch.nn.attention import sdpa_kernel, SDPBackend, WARN_FOR_UNFUSED_KERNELS


def test_sdpa_kernel_context() -> None:
    with sdpa_kernel(SDPBackend.MATH) as state:
        assert state == {}

    with sdpa_kernel([SDPBackend.MATH], set_priority=True) as state:
        assert state == {}

    with assert_raises(AssertionError):
        # E: Argument `Literal['math']` is not assignable to parameter `backends`
        with sdpa_kernel("math"):
            pass


if TYPE_CHECKING:
    assert_type(WARN_FOR_UNFUSED_KERNELS, bool)

    with sdpa_kernel(SDPBackend.MATH) as state:
        assert_type(state, dict[object, object])
