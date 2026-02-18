# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test Tensor indexing operations.

Integer indexing reduces dimensionality by 1.
Slice indexing preserves rank: result dim = stop - start.
"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# Integer Indexing - Reduces Rank
# ============================================================================


def index_2d_to_1d(x: Tensor[10, 20]) -> Tensor[20]:
    """Integer index on 2D tensor gives 1D tensor"""
    return x[0]


def index_3d_to_2d(x: Tensor[5, 10, 15]) -> Tensor[10, 15]:
    """Integer index on 3D tensor gives 2D tensor"""
    return x[0]


def index_1d_to_scalar(x: Tensor[10]) -> Tensor[()]:
    """Integer index on 1D tensor gives scalar tensor"""
    return x[0]


# ============================================================================
# Slice Indexing - dim = stop - start
# ============================================================================


def slice_upper_only(x: Tensor[10, 20]) -> Tensor[5, 20]:
    """Slice [:5] → stop - start = 5 - 0 = 5"""
    return x[:5]


def slice_lower_only(x: Tensor[10, 20]) -> Tensor[7, 20]:
    """Slice [3:] → stop - start = 10 - 3 = 7"""
    return x[3:]


def slice_both(x: Tensor[10, 20]) -> Tensor[7, 20]:
    """Slice [3:10] → stop - start = 10 - 3 = 7"""
    return x[3:10]


def slice_both_middle(x: Tensor[10, 20]) -> Tensor[4, 20]:
    """Slice [2:6] → stop - start = 6 - 2 = 4"""
    return x[2:6]


def slice_no_bounds(x: Tensor[10, 20]) -> Tensor[10, 20]:
    """Slice [:] → stop - start = 10 - 0 = 10"""
    return x[:]


# ============================================================================
# Slice Indexing - Symbolic Dimensions
# ============================================================================


def slice_upper_symbolic[N, M](x: Tensor[N, M]) -> None:
    """Slice [:5] on symbolic → stop - start = 5 - 0 = 5"""
    assert_type(x[:5], Tensor[5, M])


def slice_lower_symbolic[N, M](x: Tensor[N, M]) -> Tensor[N - 3, M]:
    """Slice [3:] on symbolic → stop - start = N - 3"""
    return x[3:]


def slice_no_bounds_symbolic[N, M](x: Tensor[N, M]) -> None:
    """Slice [:] on symbolic → stop - start = N - 0 = N"""
    assert_type(x[:], Tensor[N, M])


# ============================================================================
# Slice Indexing - Negative Indices
# ============================================================================


def slice_neg_upper(x: Tensor[10, 20]) -> Tensor[9, 20]:
    """Slice [:-1] → dim0 + (-1) - 0 = 9"""
    return x[:-1]


def slice_neg_lower(x: Tensor[10, 20]) -> Tensor[2, 20]:
    """Slice [-2:] → dim0 - (dim0 + (-2)) = 2"""
    return x[-2:]


def slice_neg_both(x: Tensor[10, 20]) -> Tensor[2, 20]:
    """Slice [-3:-1] → (dim0 + (-1)) - (dim0 + (-3)) = 2"""
    return x[-3:-1]


# ============================================================================
# Integer Indexing - Symbolic Dimensions
# ============================================================================


def index_symbolic[N, M](x: Tensor[N, M]) -> Tensor[M]:
    """Integer index on symbolic tensor reduces rank"""
    return x[0]


def index_3d_symbolic[B, N, M](x: Tensor[B, N, M]) -> Tensor[N, M]:
    """Integer index on 3D symbolic tensor"""
    return x[0]


# ============================================================================
# Multidimensional Indexing - Concrete
# ============================================================================


def tuple_int_slice(x: Tensor[5, 10, 15]) -> None:
    """Int removes dim, slice keeps dim"""
    assert_type(x[0, :], Tensor[10, 15])


def tuple_slice_int(x: Tensor[5, 10, 15]) -> None:
    """Slice keeps dim, int removes dim"""
    assert_type(x[:, 0], Tensor[5, 15])


def tuple_slice_slice(x: Tensor[5, 10, 15]) -> None:
    """Two slices with bounds"""
    assert_type(x[:3, :7], Tensor[3, 7, 15])


def tuple_neg_slice(x: Tensor[10, 20]) -> None:
    """Negative slice in multidimensional"""
    assert_type(x[:-1, :5], Tensor[9, 5])


# ============================================================================
# Multidimensional Indexing - Unpacked (prefix only)
# ============================================================================


def tuple_unpacked_int_slice[B, D, *Ts, C](
    x: Tensor[B, D, *Ts, C],
) -> None:
    """Int + slice within prefix: remove B, keep D"""
    assert_type(x[0, :], Tensor[D, *Ts, C])


def tuple_unpacked_slice_int[B, D, *Ts, C](
    x: Tensor[B, D, *Ts, C],
) -> None:
    """Slice + int within prefix: slice B, remove D"""
    assert_type(x[:5, 0], Tensor[5, *Ts, C])


def tuple_unpacked_all_slices[B, D, *Ts, C](
    x: Tensor[B, D, *Ts, C],
) -> None:
    """All slices within prefix: shape preserved"""
    assert_type(x[:, :], Tensor[B, D, *Ts, C])


def tuple_unpacked_exceeds_prefix[B, *Ts, C](
    x: Tensor[B, *Ts, C],
) -> None:
    """Indices exceed prefix → hits middle → shapeless"""
    assert_type(x[0, 0], Tensor)


# ============================================================================
# Ellipsis Indexing - Concrete
# ============================================================================


def ellipsis_only(x: Tensor[5, 10, 15]) -> None:
    """Ellipsis alone preserves entire shape"""
    assert_type(x[...], Tensor[5, 10, 15])


def ellipsis_then_int(x: Tensor[5, 10, 15]) -> None:
    """Ellipsis then int: removes last dim"""
    assert_type(x[..., 0], Tensor[5, 10])


def int_then_ellipsis(x: Tensor[5, 10, 15]) -> None:
    """Int then ellipsis: removes first dim"""
    assert_type(x[0, ...], Tensor[10, 15])


def ellipsis_then_slice(x: Tensor[5, 10, 15]) -> None:
    """Ellipsis then slice: slices last dim"""
    assert_type(x[..., :7], Tensor[5, 10, 7])


def int_ellipsis_int(x: Tensor[5, 10, 15]) -> None:
    """Int, ellipsis, int: removes first and last dims"""
    assert_type(x[0, ..., 0], Tensor[10])


def slice_ellipsis_slice(x: Tensor[5, 10, 15]) -> None:
    """Slice, ellipsis, slice: slices first and last dims"""
    assert_type(x[:3, ..., :7], Tensor[3, 10, 7])


# ============================================================================
# Ellipsis Indexing - Unpacked
# ============================================================================


def ellipsis_unpacked_post_suffix[B, *Ts, C, D](
    x: Tensor[B, *Ts, C, D],
) -> None:
    """Ellipsis then int on suffix: removes last suffix dim"""
    assert_type(x[..., 0], Tensor[B, *Ts, C])


def ellipsis_unpacked_pre_prefix[B, D, *Ts, C](
    x: Tensor[B, D, *Ts, C],
) -> None:
    """Int then ellipsis on prefix: removes first prefix dim"""
    assert_type(x[0, ...], Tensor[D, *Ts, C])


def ellipsis_unpacked_both[B, D, *Ts, C, E](
    x: Tensor[B, D, *Ts, C, E],
) -> None:
    """Int, ellipsis, int: removes from prefix and suffix"""
    assert_type(x[0, ..., 0], Tensor[D, *Ts, C])


def ellipsis_unpacked_exceeds_suffix[B, *Ts, C](
    x: Tensor[B, *Ts, C],
) -> None:
    """Post-ellipsis indices exceed suffix → shapeless"""
    assert_type(x[..., 0, 0], Tensor)


# ============================================================================
# Index Type Errors
# ============================================================================


def index_wrong_result(x: Tensor[10, 20]) -> Tensor[10, 20]:
    """ERROR: Integer index reduces rank, can't return 2D"""
    return x[0]  # ERROR: Tensor[20] not assignable to Tensor[10, 20]


def slice_wrong_size(x: Tensor[10, 20]) -> Tensor[3, 20]:
    """ERROR: Slice [:5] gives 5 elements, not 3"""
    return x[:5]  # ERROR: Tensor[5, 20] not assignable to Tensor[3, 20]
