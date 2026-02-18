# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_slice_with_typed_bounds():
    """Verify slices use TYPE information, not values"""
    x: Tensor[10, 20, 30] = torch.randn(10, 20, 30)

    # General int bound - should infer fresh var
    n: int = 5  # Type: int (general, not Literal[5])
    y1 = x[:n]

    assert_type(y1, Tensor[Any, 20, 30])

    # Literal[5] bound
    from typing import Literal

    m: Literal[5] = 5
    y2 = x[:m]
    assert_type(y2, Tensor[5, 20, 30])  # Should extract 5 from type


def test_slice_with_symbolic():
    """Verify Dim bounds extract the inner type"""
    x: Tensor[10, 20, 30] = torch.randn(10, 20, 30)

    # Dim[N] bound - use type variable
    def with_symbolic[N](n: Dim[N]):
        y = x[:n]
        assert_type(y, Tensor[N, 20, 30])


def test_tuple_variables():
    """Verify tuple indexing uses TYPE information, not AST"""
    x: Tensor[10, 20, 30] = torch.randn(10, 20, 30)

    # Literal tuple in code - works
    y1 = x[:, (0, 2, 4), :]
    assert_type(y1, Tensor[10, 3, 30])

    # Tuple variable - should also work (type-based)
    indices: tuple[int, int, int] = (0, 2, 4)
    y2 = x[:, indices, :]
    assert_type(y2, Tensor[10, 3, 30])  # Type has 3 elements

    # Different tuple length
    indices2: tuple[int, int] = (0, 1)
    y3 = x[:, indices2, :]
    assert_type(y3, Tensor[10, 2, 30])  # Type has 2 elements


def test_list_indices_not_supported():
    """Verify lists use Any for dimension (no compile-time length)"""
    x: Tensor[10, 20, 30] = torch.randn(10, 20, 30)

    # List variable - can't determine length at type level
    indices: list[int] = [0, 1, 2]
    y = x[:, indices, :]
    # List doesn't preserve length - dimension becomes Any
    assert_type(y, Tensor[10, Any, 30])  # Unknown dimension


def test_mixed_typed_indexing():
    """Mix different index types with typed variables"""
    x: Tensor[10, 20, 30, 40] = torch.randn(10, 20, 30, 40)

    from typing import Literal

    bound: Literal[5] = 5
    indices: tuple[int, int] = (1, 3)

    # slice[:bound], tuple variable, integer, slice
    y = x[:bound, indices, 0, :]
    assert_type(y, Tensor[5, 2, 40])
