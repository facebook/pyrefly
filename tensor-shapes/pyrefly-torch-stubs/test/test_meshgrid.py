# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_meshgrid_shapes() -> None:
    x = torch.arange(2)
    y = torch.arange(3)
    z = torch.arange(4)
    w = torch.arange(5)
    scalar = torch.tensor(0)

    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    assert_shape(grid_x.shape, (2, 3))
    assert_shape(grid_y.shape, (2, 3))

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="xy")
    assert_shape(grid_x.shape, (3, 2, 4))
    assert_shape(grid_y.shape, (3, 2, 4))
    assert_shape(grid_z.shape, (3, 2, 4))

    grid_x, grid_y, grid_z, grid_w = torch.meshgrid(x, y, z, w, indexing="xy")
    assert_shape(grid_x.shape, (3, 2, 4, 5))
    assert_shape(grid_y.shape, (3, 2, 4, 5))
    assert_shape(grid_z.shape, (3, 2, 4, 5))
    assert_shape(grid_w.shape, (3, 2, 4, 5))

    grid_scalar, grid_x = torch.meshgrid(scalar, x, indexing="ij")
    assert_shape(grid_scalar.shape, (1, 2))
    assert_shape(grid_x.shape, (1, 2))

    grid_x, grid_y = torch.meshgrid([x, y], indexing="ij")
    assert_shape(grid_x.shape, IntTuple, runtime=(2, 3))
    assert_shape(grid_y.shape, IntTuple, runtime=(2, 3))

    grid_x, grid_y = torch.meshgrid((x, y), indexing="xy")
    assert_shape(grid_x.shape, IntTuple, runtime=(3, 2))
    assert_shape(grid_y.shape, IntTuple, runtime=(3, 2))

    with assert_raises(RuntimeError):
        torch.meshgrid(indexing="ij")  # E: meshgrid expects at least one tensor
    with assert_raises(TypeError):
        torch.meshgrid(tensor1=x, indexing="ij")  # E: No matching overload
    with assert_raises(RuntimeError):
        # E: meshgrid expects scalar or 1D tensors
        torch.meshgrid(torch.ones((2, 2)), indexing="ij")
    with assert_raises(RuntimeError):
        # E: meshgrid indexing must be
        torch.meshgrid(x, indexing="invalid")


if TYPE_CHECKING:

    def check_container_meshgrid(
        tensors_list: list[Tensor], tensors_tuple: tuple[Tensor, ...]
    ) -> None:
        assert_type(
            torch.meshgrid(tensors_list, indexing="ij"),
            tuple[Tensor, ...],
        )
        assert_type(
            torch.meshgrid(tensors_tuple, indexing="xy"),
            tuple[Tensor, ...],
        )

    def check_dynamic_meshgrid[N: IntVar, M: IntVar](
        x: Tensor[[N]], y: Tensor[[M]], indexing: str
    ) -> None:
        assert_type(
            torch.meshgrid(x, y, indexing=indexing),
            tuple[Tensor, ...],
        )

    def check_symbolic_meshgrid[N: IntVar, M: IntVar, K: IntVar, L: IntVar](
        scalar: Tensor[[]],
        x: Tensor[[N]],
        y: Tensor[[M]],
        z: Tensor[[K]],
        w: Tensor[[L]],
    ) -> None:
        assert_type(torch.meshgrid(x), tuple[Tensor[[N]]])
        assert_type(
            torch.meshgrid(x, y, indexing=None),
            tuple[Tensor[[N, M]], Tensor[[N, M]]],
        )
        assert_type(
            torch.meshgrid(x, y, indexing="ij"),
            tuple[Tensor[[N, M]], Tensor[[N, M]]],
        )
        assert_type(
            torch.meshgrid(x, y, z, indexing="xy"),
            tuple[
                Tensor[[M, N, K]],
                Tensor[[M, N, K]],
                Tensor[[M, N, K]],
            ],
        )
        assert_type(
            torch.meshgrid(x, y, z, w, indexing="xy"),
            tuple[
                Tensor[[M, N, K, L]],
                Tensor[[M, N, K, L]],
                Tensor[[M, N, K, L]],
                Tensor[[M, N, K, L]],
            ],
        )
        assert_type(
            torch.meshgrid(scalar, x, indexing="ij"),
            tuple[Tensor[[1, N]], Tensor[[1, N]]],
        )
