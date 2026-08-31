# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Focused tests for the shared `pool_shape` helper.

The six pooling module stubs keep scalar-only arguments, so the tuple side of the
shared surface is pinned here through a local stub that matches the functional
pooling surface.
"""

from typing import TYPE_CHECKING, assert_type, reveal_type

import torch
import torch.nn as nn
from shape_extensions import Flag, IntTuple, IntVar

if TYPE_CHECKING:
    from torch import Tensor
    from torch._shapes import pool_shape


def pool2d[
    Shape: IntTuple,
    KernelSize: Flag[int | tuple[int, ...]],
    Stride: Flag[int | tuple[int, ...] | None],
    Padding: Flag[int | tuple[int, ...]],
    Dilation: Flag[int | tuple[int, ...]],
](
    input: Tensor[Shape],
    kernel_size: KernelSize,
    stride: Stride = None,
    padding: Padding = 0,
    dilation: Dilation = 1,
) -> Tensor[pool_shape(Shape, 2, KernelSize, Stride, Padding, Dilation, False)]: ...


def test_scalar_arguments_normalize_to_every_axis():
    x: Tensor[[2, 3, 8, 12]] = torch.randn(2, 3, 8, 12)
    assert_type(pool2d(x, 2), Tensor[[2, 3, 4, 6]])
    assert_type(pool2d(x, 3, 1, 1), Tensor[[2, 3, 8, 12]])


def test_tuple_arguments_apply_per_axis():
    x: Tensor[[2, 3, 8, 9]] = torch.randn(2, 3, 8, 9)
    assert_type(pool2d(x, (2, 3)), Tensor[[2, 3, 4, 3]])
    assert_type(pool2d(x, (2, 3), (2, 1)), Tensor[[2, 3, 4, 7]])
    assert_type(pool2d(x, 2, None, (1, 0)), Tensor[[2, 3, 5, 4]])
    assert_type(pool2d(x, (2, 2), 2, 0, (2, 1)), Tensor[[2, 3, 3, 4]])


def test_omitted_stride_uses_the_normalized_kernel():
    x: Tensor[[2, 3, 8, 9]] = torch.randn(2, 3, 8, 9)
    # Adjacent windows of the kernel, per axis: 8 // 2 and 9 // 3.
    assert_type(pool2d(x, (2, 3)), Tensor[[2, 3, 4, 3]])
    assert_type(pool2d(x, (2, 3), None), Tensor[[2, 3, 4, 3]])


def test_pool_module_arguments_stay_scalar():
    # The module surfaces intentionally admit only scalar arguments; the tuple side
    # of the helper is reached through the stub above.
    x: Tensor[[2, 3, 8, 9]] = torch.randn(2, 3, 8, 9)
    assert_type(nn.MaxPool2d(2)(x), Tensor[[2, 3, 4, 4]])
    assert_type(nn.AvgPool2d(2)(x), Tensor[[2, 3, 4, 4]])


# ============================================================================
# Validation is not deferred past specialization
# ============================================================================
#
# An argument that is still generic cannot be range-checked, and the DSL is not
# re-evaluated once it is specialized. Each generic form below must therefore
# recover gradually, so that specializing it to an invalid value cannot resurrect
# arithmetic that no validation ever revisits.


def generic_kernel[KernelSize: Flag[int]](
    x: Tensor[[2, 3, 8]], kernel_size: KernelSize
):
    pooled = nn.MaxPool1d(kernel_size)(x)
    assert_type(pooled, Tensor)
    return pooled


def generic_stride[Stride: Flag[int]](x: Tensor[[2, 3, 8]], stride: Stride):
    pooled = nn.MaxPool1d(2, stride)(x)
    assert_type(pooled, Tensor)
    return pooled


def generic_padding[Padding: Flag[int]](x: Tensor[[2, 3, 8]], padding: Padding):
    pooled = nn.MaxPool1d(2, 2, padding)(x)
    assert_type(pooled, Tensor)
    return pooled


def generic_dilation[Dilation: Flag[int]](x: Tensor[[2, 3, 8]], dilation: Dilation):
    pooled = nn.MaxPool1d(2, 2, 0, dilation)(x)
    assert_type(pooled, Tensor)
    return pooled


def test_invalid_specializations_stay_gradual():
    x: Tensor[[2, 3, 8]] = torch.randn(2, 3, 8)
    assert_type(generic_kernel(x, 0), Tensor)
    assert_type(generic_stride(x, 0), Tensor)
    assert_type(generic_padding(x, 2), Tensor)
    assert_type(generic_padding(x, -1), Tensor)
    assert_type(generic_dilation(x, 0), Tensor)


# Unlike runtime controls, symbolic input extents retain their output formulas.
# Specialization can therefore produce a nonpositive dimension that the DSL cannot reject.
def generic_extent[L: IntVar](x: Tensor[[2, 3, L]]):
    pooled = nn.MaxPool1d(4)(x)
    assert_type(pooled, Tensor[[2, 3, (L - 4) // 4 + 1]])
    return pooled


def test_symbolic_extent_preserves_its_formula():
    too_small: Tensor[[2, 3, 2]] = torch.randn(2, 3, 2)
    reveal_type(generic_extent(too_small))  # revealed type: Tensor[[2, 3, 0]]


def check_undecidable_tuple_arguments_stay_gradual(
    unknown_arity: tuple[int, ...],
    unknown_elements: tuple[int, int],
) -> None:
    # A tuple argument is undecidable in two independent ways: its arity may be
    # unknown, so the per-axis rule cannot be zipped, or its arity may be fixed
    # while the entries are unknown, so no value predicate can be answered. Either
    # way the call must recover gradually rather than leave a check unmade.
    x: Tensor[[2, 3, 8, 12]] = torch.randn(2, 3, 8, 12)
    assert_type(pool2d(x, unknown_arity), Tensor)
    assert_type(pool2d(x, unknown_elements), Tensor)


def test_known_arguments_stay_exact():
    # Gradual recovery is limited to what cannot be decided: the same arguments are
    # exact when known, and invalid when known to be invalid (see
    # `negative_tests/test_module_argument_shapes.py`).
    x: Tensor[[2, 3, 8]] = torch.randn(2, 3, 8)
    assert_type(nn.MaxPool1d(2, 2)(x), Tensor[[2, 3, 4]])


def test_chained_symbolic_pools_stay_compact[
    B: IntVar,
    C: IntVar,
    H: IntVar,
    W: IntVar,
](x: Tensor[[B, C, H, W]]) -> None:
    # Chained ceil-mode pools are the growth case: the ceil correction names its own
    # window count three times, so a deferred expression would triple in size per
    # stage. Recovering gradually keeps every stage compact instead.
    once = nn.MaxPool2d(3, stride=2, ceil_mode=True)(x)
    assert_type(once, Tensor[[B, C, int, int]])
    twice = nn.MaxPool2d(3, stride=2, ceil_mode=True)(once)
    assert_type(twice, Tensor[[B, C, int, int]])
    assert_type(
        nn.MaxPool2d(3, stride=2, ceil_mode=True)(twice),
        Tensor[[B, C, int, int]],
    )
