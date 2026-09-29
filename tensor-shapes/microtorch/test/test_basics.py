# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, Literal, TYPE_CHECKING

import microtorch as torch
from microtorch import Tensor
from shape_extensions import Int, IntVar


def test_basic_shapes() -> None:
    left = torch.randn((2, 3))
    right = torch.ones((3, 5))

    assert_type(left + left, Tensor[[2, 3]])
    assert_type(left @ right, Tensor[[2, 5]])
    assert_type(left.transpose(), Tensor[[3, 2]])
    assert_type(torch.diagonal(torch.ones((4,)), offset=1), Tensor[[5, 5]])
    assert_type(
        torch.concatenate((torch.zeros((2, 3)), torch.ones((2, 4))), axis=1),
        Tensor[[2, 7]],
    )


if TYPE_CHECKING:

    def check_constant_slices(tensor: Tensor[[10]]) -> None:
        shaped_start: Int[1] = 1
        shaped_stop: Int[5] = 5
        shaped_step: Int[2] = 2
        shaped = slice(shaped_start, shaped_stop, shaped_step)
        literal_start: Literal[1] = 1
        literal_stop: Literal[5] = 5
        literal_step: Literal[2] = 2
        literal = slice(literal_start, literal_stop, literal_step)
        assert_type(shaped, slice[Int[1], Int[5], Int[2]])
        assert_type(literal, slice[Literal[1], Literal[5], Literal[2]])
        assert_type(tensor[shaped], Tensor[[2]])
        assert_type(tensor[literal], Tensor[[2]])

    def check_symbolic_slice[N: IntVar](tensor: Tensor[[N]], stop: Int[N]) -> None:
        section: slice[None, Int[N], None] = slice(None, stop)
        assert_type(tensor[section], Tensor[[N]])

    def check_shape_derived_slice[N: IntVar](tensor: Tensor[[N]]) -> None:
        stop = tensor.shape[0]
        assert_type(stop, Int[N])
        section: slice[None, Int[N], None] = slice(None, stop)
        assert_type(tensor[section], Tensor[[N]])
