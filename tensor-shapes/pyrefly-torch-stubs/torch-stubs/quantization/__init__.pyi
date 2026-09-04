# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stubs for torch.quantization APIs used in DLRM."""

from typing import Any

import torch.nn as nn

def quantize_dynamic(
    model: nn.Module,
    qconfig_spec: set[type] | dict[type, Any] | None = None,
    dtype: Any = ...,
    mapping: dict[type, type] | None = None,
    inplace: bool = False,
) -> nn.Module: ...
