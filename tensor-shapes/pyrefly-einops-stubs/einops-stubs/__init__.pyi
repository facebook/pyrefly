# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, overload, Protocol

from shape_extensions import Flag, IntTuple

from ._shapes import einsum_shape, rearrange_shape, reduce_shape, repeat_shape

# TODO: Replace these placeholders with precise signatures in follow-up diffs.
EinopsError: Any
asnumpy: Any
pack: Any
parse_shape: Any
unpack: Any

class _Array[Shape: IntTuple = IntTuple](Protocol):
    @property
    def shape(self) -> Shape: ...

@overload
def rearrange[Shape: IntTuple, Pattern: Flag[str]](
    tensor: _Array[Shape], pattern: Pattern
) -> _Array[rearrange_shape(Pattern, Shape)]: ...
@overload
def rearrange(tensor: _Array, pattern: str, **axes_lengths: int) -> _Array: ...
@overload
def rearrange(tensor: Any, pattern: str, **axes_lengths: int) -> Any: ...
@overload
def reduce[Shape: IntTuple, Pattern: Flag[str]](
    tensor: _Array[Shape], pattern: Pattern, reduction: Any
) -> _Array[reduce_shape(Pattern, Shape)]: ...
@overload
def reduce(
    tensor: _Array, pattern: str, reduction: Any, **axes_lengths: int
) -> _Array: ...
@overload
def reduce(tensor: Any, pattern: str, reduction: Any, **axes_lengths: int) -> Any: ...
@overload
def repeat[Shape: IntTuple, Pattern: Flag[str]](
    tensor: _Array[Shape], pattern: Pattern
) -> _Array[repeat_shape(Pattern, Shape)]: ...
@overload
def repeat(tensor: _Array, pattern: str, **axes_lengths: int) -> _Array: ...
@overload
def repeat(tensor: Any, pattern: str, **axes_lengths: int) -> Any: ...
@overload
def einsum[S1: IntTuple, Pattern: Flag[str]](
    tensor: _Array[S1], pattern: Pattern, /
) -> _Array[einsum_shape(Pattern, tuple[S1])]: ...
@overload
def einsum[S1: IntTuple, S2: IntTuple, Pattern: Flag[str]](
    tensor1: _Array[S1], tensor2: _Array[S2], pattern: Pattern, /
) -> _Array[einsum_shape(Pattern, tuple[S1, S2])]: ...
@overload
def einsum[S1: IntTuple, S2: IntTuple, S3: IntTuple, Pattern: Flag[str]](
    tensor1: _Array[S1],
    tensor2: _Array[S2],
    tensor3: _Array[S3],
    pattern: Pattern,
    /,
) -> _Array[einsum_shape(Pattern, tuple[S1, S2, S3])]: ...
@overload
def einsum[
    S1: IntTuple,
    S2: IntTuple,
    S3: IntTuple,
    S4: IntTuple,
    Pattern: Flag[str],
](
    tensor1: _Array[S1],
    tensor2: _Array[S2],
    tensor3: _Array[S3],
    tensor4: _Array[S4],
    pattern: Pattern,
    /,
) -> _Array[einsum_shape(Pattern, tuple[S1, S2, S3, S4])]: ...
@overload
def einsum(*tensors_and_pattern: Any) -> Any: ...
