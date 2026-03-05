# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.distributions.transforms."""

from typing import Any

from torch import Tensor

class Transform:
    """Base class for invertible transforms with computable log det Jacobians."""

    domain: Any
    codomain: Any
    bijective: bool
    sign: int

    def __init__(self, cache_size: int = 0) -> None: ...
    def __call__(self, x: Tensor) -> Tensor: ...
    def _call(self, x: Tensor) -> Tensor: ...
    def _inverse(self, y: Tensor) -> Tensor: ...
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor: ...
