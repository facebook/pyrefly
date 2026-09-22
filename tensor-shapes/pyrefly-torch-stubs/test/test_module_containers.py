# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING, TypedDict

import torch
import torch.nn as nn
from shape_extensions import assert_shape, Int, IntVar
from torch import Tensor


class Block[N: IntVar](nn.Module):
    def __init__(self, width: Int[N]) -> None:
        super().__init__()
        self.width = width

    def forward[B: IntVar, T: IntVar](
        self, tensor: Tensor[[B, T, N]]
    ) -> Tensor[[B, T, N]]:
        return tensor.relu()


class Modules(TypedDict):
    projection: nn.Linear[3, 4]
    embedding: nn.Embedding[10, 3]
    dropout: nn.Dropout


def test_module_list() -> None:
    modules = nn.ModuleList([Block(4), Block(4)])
    tensor = torch.ones((2, 3, 4))
    for block in modules:
        tensor = block(tensor)
    assert_shape(tensor.shape, (2, 3, 4))
    assert_type(modules[0], Block[4])
    assert_type(modules[:1], nn.ModuleList[Block[4]])
    assert len(modules) == 2


def test_parameter_list() -> None:
    parameters = nn.ParameterList(
        [nn.Parameter(torch.ones((3, 3))), nn.Parameter(torch.zeros((3, 3)))]
    )
    assert len(parameters) == 2
    assert_shape(parameters[0].shape, (3, 3))
    assert_shape(parameters[1].shape, (3, 3))
    assert_type(parameters[0], Tensor[[3, 3]])


def test_module_dict_typed_fields() -> None:
    modules: Modules = {
        "projection": nn.Linear(3, 4),
        "embedding": nn.Embedding(10, 3),
        "dropout": nn.Dropout(p=0.0),
    }
    module_dict = nn.ModuleDict(modules)

    assert_type(module_dict.projection, nn.Linear[3, 4])
    assert_type(module_dict["embedding"], nn.Embedding[10, 3])
    assert_type(module_dict.dropout, nn.Dropout)

    projected = module_dict.projection(torch.ones((2, 3)))
    assert_shape(projected.shape, (2, 4))
    embedded = module_dict["embedding"](torch.tensor([[1, 2], [3, 4]]))
    assert_shape(embedded.shape, (2, 2, 3))
    assert_shape(module_dict.dropout(projected).shape, (2, 4))


if TYPE_CHECKING:

    def check_symbolic_module_list[B: IntVar, T: IntVar, N: IntVar](
        modules: nn.ModuleList[Block[N]], tensor: Tensor[[B, T, N]]
    ) -> None:
        assert_type(modules[0](tensor), Tensor[[B, T, N]])
        for block in modules:
            tensor = block(tensor)
        assert_type(tensor, Tensor[[B, T, N]])
