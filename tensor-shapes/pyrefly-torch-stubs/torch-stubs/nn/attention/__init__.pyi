# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.nn.attention module."""

from typing import Any

from torch._C import _SDPBackend as SDPBackend
from torch.nn.attention._registry import (
    activate_flash_attention_impl as activate_flash_attention_impl,
    current_flash_attention_impl as current_flash_attention_impl,
    list_flash_attention_impls as list_flash_attention_impls,
    register_flash_attention_impl as register_flash_attention_impl,
    restore_flash_attention_impl as restore_flash_attention_impl,
)
from torch.nn.attention.flex_attention import (
    _mask_mod_signature as _mask_mod_signature,
    BlockMask as BlockMask,
    flex_attention as flex_attention,
)

# TODO: Add precise types for the remaining public API.
WARN_FOR_UNFUSED_KERNELS: Any
sdpa_kernel: Any
