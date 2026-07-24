# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test general unpacked tensor matching.

Tests that Tensor[[A, B, *Elements[Cs], D, E, F]] can match Tensor[[P, *Elements[Qs], R, S]]
where:
- A matches P (prefix)
- F matches S, E matches R (suffix from end)
- tuple[B, *Cs, D] matches Qs (middle)
"""

from typing import assert_type, cast

from shape_extensions import Elements, IntTuple, IntVar
from torch import Tensor


def accepts_prefix_middle_suffix[
    P: IntVar,
    Qs: IntTuple,
    R: IntVar,
    S: IntVar,
](
    x: Tensor[[P, *Elements[Qs], R, S]],
) -> Tensor[[P, *Elements[Qs], R, S]]:
    """Function that expects prefix P, middle *Qs, and suffix R, S."""
    return x


def test_general_unpacked_matching[
    A: IntVar,
    B: IntVar,
    Cs: IntTuple,
    D: IntVar,
    E: IntVar,
    F: IntVar,
]() -> None:
    """Test that Tensor[[A, B, *Elements[Cs], D, E, F]] matches Tensor[[P, *Elements[Qs], R, S]]."""
    # Create a tensor with more complex unpacked shape
    x = cast(Tensor[[A, B, *Elements[Cs], D, E, F]], ...)
    assert_type(x, Tensor[[A, B, *Elements[Cs], D, E, F]])

    # Pass to function expecting fewer prefix/suffix dims
    # This should match with:
    #   P = A
    #   Qs = tuple[B, *Cs, D]
    #   R = E
    #   S = F
    result = accepts_prefix_middle_suffix(x)
    assert_type(result, Tensor[[A, B, *Elements[Cs], D, E, F]])


def test_general_unpacked_matching_arith[
    A: IntVar,
    B: IntVar,
    Cs: IntTuple,
    D: IntVar,
    E: IntVar,
]() -> None:
    """Test that Tensor[[A+1, B*2, *Elements[Cs], D, E, F]] matches Tensor[[P, *Elements[Qs], R, S]]."""
    # Create a tensor with more complex unpacked shape
    x = cast(Tensor[[A + 1, B * 2, *Elements[Cs], D, E, E // 3]], ...)
    assert_type(x, Tensor[[A + 1, B * 2, *Elements[Cs], D, E, E // 3]])

    # Pass to function expecting fewer prefix/suffix dims
    # This should match with:
    #   P = A
    #   Qs = tuple[B, *Cs, D]
    #   R = E
    #   S = F
    result = accepts_prefix_middle_suffix(x)
    assert_type(result, Tensor[[A + 1, B * 2, *Elements[Cs], D, E, E // 3]])


def test_concrete_general_matching() -> None:
    """Test with concrete dimensions."""
    x = cast(Tensor[[1, 2, 3, 4, 5, 6]], ...)

    # Should match as: P=1, Qs=tuple[2,3,4], R=5, S=6
    result = accepts_prefix_middle_suffix(x)
    assert_type(result, Tensor[[1, 2, 3, 4, 5, 6]])
