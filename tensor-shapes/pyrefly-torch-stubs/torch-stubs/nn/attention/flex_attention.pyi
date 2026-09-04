# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.nn.attention.flex_attention module."""

from typing import Any, Callable, Literal, NamedTuple, overload, TypedDict

from shape_extensions import Int, IntVar
from torch import Tensor

# Mask modifiers receive scalar tensor indices, including when vectorized by Torch.
_mask_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
_score_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]

class BlockMask:
    """Block mask for flex attention.

    Stores precomputed block-sparse attention mask for efficient attention computation.
    """

    seq_lengths: tuple[int, int]
    kv_num_blocks: Tensor[[int, int, int]]
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

def flex_attention[
    B: IntVar,
    H: IntVar,
    H_kv: IntVar,
    Tq: IntVar,
    Tkv: IntVar,
    D: IntVar,
](
    query: Tensor[[B, H, Tq, D]],
    key: Tensor[[B, H_kv, Tkv, D]],
    value: Tensor[[B, H_kv, Tkv, D]],
    score_mod: Callable[..., Any] | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
) -> Tensor[[B, H, Tq, D]]:
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

def noop_mask(
    batch: Tensor, head: Tensor, token_q: Tensor, token_kv: Tensor
) -> Tensor[[]]: ...
def and_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature: ...
def or_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature: ...
@overload
def create_mask[B: IntVar, H: IntVar, Q: IntVar, K: IntVar](
    mod_fn: _score_mod_signature | _mask_mod_signature,
    B: Int[B],
    H: Int[H],
    Q_LEN: Int[Q],
    KV_LEN: Int[K],
    device: Any = None,
) -> Tensor[[B, H, Q, K]]: ...
@overload
def create_mask[Q: IntVar, K: IntVar](
    mod_fn: _score_mod_signature | _mask_mod_signature,
    B: None,
    H: None,
    Q_LEN: Int[Q],
    KV_LEN: Int[K],
    device: Any = None,
) -> Tensor[[1, 1, Q, K]]: ...
@overload
def create_mask[Q: IntVar, K: IntVar](
    mod_fn: _score_mod_signature | _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: Int[Q],
    KV_LEN: Int[K],
    device: Any = None,
) -> Tensor[[int, int, Q, K]]: ...
def create_block_mask(
    mask_mod: _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: Any = None,
    BLOCK_SIZE: int | tuple[int, int] = 128,
    _compile: bool = False,
) -> BlockMask: ...

class AuxRequest(NamedTuple):
    lse: bool = False
    max_scores: bool = False

class AuxOutput(NamedTuple):
    lse: Tensor | None = None
    max_scores: Tensor | None = None

class FlexKernelOptions(TypedDict, total=False):
    """Optional tuning parameters for the FlexAttention kernels."""

    num_warps: int
    num_stages: int
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_M1: int
    BLOCK_N1: int
    BLOCK_M2: int
    BLOCK_N2: int
    PRESCALE_QK: bool
    ROWS_GUARANTEED_SAFE: bool
    BLOCKS_ARE_CONTIGUOUS: bool
    WRITE_DQ: bool
    FORCE_USE_FLEX_ATTENTION: bool
    USE_TMA: bool
    kpack: int
    matrix_instr_nonkdim: int
    waves_per_eu: int
    BACKEND: Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"]
