# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.nn.attention.flex_attention module."""

from typing import Any, Callable

from torch import Tensor

# Type alias for mask modification functions
# Signature: (batch, head, query_idx, key_idx) -> bool
_mask_mod_signature = Callable[[int, int, int, int], bool]

class BlockMask:
    """Block mask for flex attention.

    Stores precomputed block-sparse attention mask for efficient attention computation.
    """

    mask_mod: _mask_mod_signature | None

    def __init__(
        self,
        mask_mod: _mask_mod_signature | None = None,
        B: int | None = None,
        H: int | None = None,
        Q_LEN: int | None = None,
        KV_LEN: int | None = None,
        device: Any = None,
        _compile: bool = False,
    ) -> None: ...

def flex_attention[B, H, H_kv, Tq, Tkv, D](
    query: Tensor[B, H, Tq, D],
    key: Tensor[B, H_kv, Tkv, D],
    value: Tensor[B, H_kv, Tkv, D],
    score_mod: Callable[..., Any] | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
) -> Tensor[B, H, Tq, D]:
    """Flexible attention with block-sparse masking.

    Args:
        query: Query tensor [B, H, Tq, D]
        key: Key tensor [B, H_kv, Tkv, D] (H_kv can differ from H when enable_gqa=True)
        value: Value tensor [B, H_kv, Tkv, D]
        score_mod: Optional score modification function
        block_mask: Optional block mask for sparse attention
        scale: Optional scaling factor (default: 1/sqrt(D))
        enable_gqa: Enable grouped query attention
        return_lse: Return log-sum-exp values

    Returns:
        Output tensor [B, H, Tq, D]
    """
    ...
