# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for variadic TypeVarTuple subset checking.

Two fixes are tested:

1. Tuple Unpacked-vs-Unpacked subset (subset.rs): passing tuple[*Cs] to a
   function expecting tuple[*Bs] should bind *Bs to *Cs. Previously the
   bidirectional check incorrectly compared *Ts ⊆ tuple[*Ts].

2. VarArg unpacked parameter matching (callable.rs): unpacking tuple[*Ts]
   into *args: *Ts should work. The parameter type (raw TypeVarTuple after
   stripping Unpack) must be wrapped in a tuple for comparison against the
   tuple built from call arguments.
"""

from __future__ import annotations

from typing import assert_type, Callable, TYPE_CHECKING

import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Fix 1: Tuple Unpacked subset
# ============================================================================


# Basic tuple: tuple[*Cs] passed to function expecting tuple[*Bs]
def identity[*Bs](x: tuple[*Bs]) -> tuple[*Bs]:
    return x


def test_basic_variadic[*Cs](x: tuple[*Cs]) -> None:
    y = identity(x)
    assert_type(y, tuple[*Cs])


# Tensor: variadic batch dims through multiple shape-preserving calls
class TwoReluCalls[D](nn.Module):
    def __init__(self, d: Dim[D]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, 256)
        self.fc2 = nn.Linear(256, 256)

    def forward[*Bs](self, x: Tensor[*Bs, D]) -> Tensor[*Bs, 256]:
        h = F.relu(self.fc1(x))
        assert_type(h, Tensor[*Bs, 256])
        return F.relu(self.fc2(h))


# Nested variadic-preserving calls (matches tacotron2 Prenet pattern)
class NestedVariadic[D](nn.Module):
    def __init__(self, d: Dim[D]) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 256)

    def forward[*Bs](self, x: Tensor[*Bs, D]) -> Tensor[*Bs, 256]:
        return F.dropout(F.relu(self.fc(x)), p=0.5, training=True)


# ============================================================================
# Fix 2: VarArg unpacked parameter matching
# ============================================================================


# asyncio pattern: callback and *args share the same TypeVarTuple.
# call_later ties _Ts to both the Callable param and *args.
# When forwarding, the unpacked tuple[*_Ts] must match *args: *_Ts.
def call_later[*_Ts](callback: Callable[[*_Ts], object], *args: *_Ts) -> None: ...


def test_forward_varargs[*_Ts](callback: Callable[[*_Ts], object], *args: *_Ts) -> None:
    # Direct call: callback takes *_Ts, args unpacks to *_Ts
    callback(*args)
    # Forwarding: call_later's *_Ts binds from callback, then *args must match
    call_later(callback, *args)
