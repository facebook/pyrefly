# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for repeat and expand with symbolic dimensions from size()

These operations must work with symbolic dimensions like Int[N] returned from x.size().
Previously failed when iter_shape_dims() filtered out Type::Quantified dimensions.
"""

from typing import assert_type, cast, TYPE_CHECKING

from shape_extensions import Elements, IntTuple, IntVar


if TYPE_CHECKING:
    from torch import Tensor


def test_repeat_symbolic[N: IntVar](x: Tensor[[N, 1]]):
    """Repeat with symbolic dimension from size()"""
    # Get symbolic dimension from size()
    n = x.size(0)  # Returns Int[N]

    # Repeat using symbolic dimension and literal
    # This previously failed with "repeat sizes length 1 doesn't match tensor rank 2"
    # because iter_shape_dims() filtered out the Quantified(N) type
    y = x.repeat(n, 3)

    # Should produce [N*N, 3]
    assert_type(y, Tensor[[N * N, 3]])


def test_expand_symbolic[N: IntVar](x: Tensor[[N, 1]]):
    """Expand with symbolic dimension from size()"""
    # Get symbolic dimension
    n = x.size(0)  # Returns Int[N]

    # Expand using symbolic dimension and literal
    # This previously failed with "expand target size length 1 doesn't match tensor rank 2"
    y = x.expand(n, 5)

    # Expands [N, 1] → [N, 5] (keeps dim 0, broadcasts dim 1)
    assert_type(y, Tensor[[N, 5]])


def test_expand_runtime_values[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    """Expand with multiple symbolic dimensions from size(), and with -1 targets"""
    n = x.size(0)
    m = x.size(1)

    assert_type(x.expand(n, m), Tensor[[N, M]])
    # -1 keeps the original dimension instead of naming it symbolically.
    assert_type(x.expand(-1, m), Tensor[[N, M]])


def test_expand_literal_tuple_and_vararg_parity():
    x = cast(Tensor[[2, 1, 4]], ...)
    assert_type(x.expand(2, 5, 4), Tensor[[2, 5, 4]])
    assert_type(x.expand((2, 5, 4)), Tensor[[2, 5, 4]])
    assert_type(x.expand(-1, -1, -1), Tensor[[2, 1, 4]])


def test_expand_leading_dimensions_and_scalar():
    x = cast(Tensor[[2, 3]], ...)
    scalar = cast(Tensor[[]], ...)
    assert_type(x.expand(4, -1, -1), Tensor[[4, 2, 3]])
    assert_type(scalar.expand(), Tensor[[]])
    assert_type(scalar.expand(2, 3), Tensor[[2, 3]])


def check_expand_gradual(
    concrete: Tensor[[2, 1]],
    open_rank: Tensor[IntTuple],
    bare: Tensor,
    broad_size: int,
    broad_tuple: tuple[int, ...],
) -> None:
    assert_type(concrete.expand(broad_size, 3), Tensor[[2, 3]])
    assert_type(concrete.expand(broad_tuple), Tensor[IntTuple])
    assert_type(open_rank.expand(2, 3), Tensor[IntTuple])
    assert_type(bare.expand(2, 3), Tensor)


def check_expand_partially_known_shape[Batch: IntTuple](
    x: Tensor[[*Elements[Batch], 2]],
) -> None:
    # Tuple unpacking currently loses the partial carrier before the DSL runs.
    assert_type(x.expand(x.size()), Tensor[IntTuple])
