# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from shape_extensions import CaptureNamedInts, Flag, IntTuple, NamedInts
from torch import Tensor
from torch.nn import Module

from .._shapes import rearrange_shape, reduce_shape

class Rearrange[Pattern: Flag[str], Axes: NamedInts](Module):
    def __init__(
        self, pattern: Pattern, **axes_lengths: CaptureNamedInts[Axes]
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[rearrange_shape(Pattern, Shape, Axes)]: ...

class Reduce[Pattern: Flag[str], Axes: NamedInts](Module):
    def __init__(
        self,
        pattern: Pattern,
        reduction: str,
        **axes_lengths: CaptureNamedInts[Axes],
    ) -> None: ...
    def forward[Shape: IntTuple](
        self, input: Tensor[Shape]
    ) -> Tensor[reduce_shape(Pattern, Shape, Axes)]: ...

class EinMix(Module):
    def __init__(
        self,
        pattern: str,
        weight_shape: str,
        bias_shape: str | None = None,
        **axes_lengths: int,
    ) -> None: ...
    def forward(self, input: Tensor[IntTuple]) -> Tensor[IntTuple]: ...
