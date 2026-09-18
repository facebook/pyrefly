# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, overload, Protocol

from shape_extensions import Flag, IntTuple
from torch import Tensor

from ._shapes import einsum_shape, rearrange_shape, reduce_shape, repeat_shape

__version__: str

class _Array[Shape: IntTuple = IntTuple](Protocol):
    @property
    def shape(self) -> Shape: ...

class EinopsError(RuntimeError): ...

def asnumpy[Shape: IntTuple](tensor: _Array[Shape]) -> _Array[Shape]: ...
def parse_shape(tensor: Tensor, pattern: str) -> dict[str, int]: ...

# TODO: Add shape primitives for the `*` packing pattern so these return precise shapes.
def pack(
    tensors: Sequence[Tensor], pattern: str
) -> tuple[Tensor[IntTuple], list[tuple[int, ...]]]: ...
def unpack(
    tensor: Tensor,
    packed_shapes: Sequence[Sequence[int]],
    pattern: str,
) -> list[Tensor[IntTuple]]: ...

# TODO: Replace these Torch-specific overloads with library-agnostic `MapShape`
# signatures that preserve each input array's nominal type.
@overload
def rearrange[Shape: IntTuple, Pattern: Flag[str]](
    tensor: Tensor[Shape], pattern: Pattern
) -> Tensor[rearrange_shape(Pattern, Shape)]: ...
@overload
def rearrange(
    tensor: Tensor, pattern: str, **axes_lengths: int
) -> Tensor[IntTuple]: ...
@overload
def rearrange(tensor: Any, pattern: str, **axes_lengths: int) -> Any: ...
@overload
def reduce[Shape: IntTuple, Pattern: Flag[str]](
    tensor: Tensor[Shape], pattern: Pattern, reduction: Any
) -> Tensor[reduce_shape(Pattern, Shape)]: ...
@overload
def reduce(
    tensor: Tensor, pattern: str, reduction: Any, **axes_lengths: int
) -> Tensor[IntTuple]: ...
@overload
def reduce(tensor: Any, pattern: str, reduction: Any, **axes_lengths: int) -> Any: ...
@overload
def repeat[Shape: IntTuple, Pattern: Flag[str]](
    tensor: Tensor[Shape], pattern: Pattern
) -> Tensor[repeat_shape(Pattern, Shape)]: ...
@overload
def repeat(tensor: Tensor, pattern: str, **axes_lengths: int) -> Tensor[IntTuple]: ...
@overload
def repeat(tensor: Any, pattern: str, **axes_lengths: int) -> Any: ...
@overload
def einsum[S1: IntTuple, Pattern: Flag[str]](
    tensor: Tensor[S1], pattern: Pattern, /
) -> Tensor[einsum_shape(Pattern, tuple[S1])]: ...
@overload
def einsum[S1: IntTuple, S2: IntTuple, Pattern: Flag[str]](
    tensor1: Tensor[S1], tensor2: Tensor[S2], pattern: Pattern, /
) -> Tensor[einsum_shape(Pattern, tuple[S1, S2])]: ...
@overload
def einsum[S1: IntTuple, S2: IntTuple, S3: IntTuple, Pattern: Flag[str]](
    tensor1: Tensor[S1],
    tensor2: Tensor[S2],
    tensor3: Tensor[S3],
    pattern: Pattern,
    /,
) -> Tensor[einsum_shape(Pattern, tuple[S1, S2, S3])]: ...
@overload
def einsum[
    S1: IntTuple,
    S2: IntTuple,
    S3: IntTuple,
    S4: IntTuple,
    Pattern: Flag[str],
](
    tensor1: Tensor[S1],
    tensor2: Tensor[S2],
    tensor3: Tensor[S3],
    tensor4: Tensor[S4],
    pattern: Pattern,
    /,
) -> Tensor[einsum_shape(Pattern, tuple[S1, S2, S3, S4])]: ...
@overload
def einsum(*tensors_and_pattern: Any) -> Any: ...
