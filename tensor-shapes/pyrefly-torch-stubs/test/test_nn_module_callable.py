# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test that nn.Module instances are callable (automatically redirect to forward method).
"""

from typing import assert_type, Protocol, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor


class SimpleModule[N: IntVar, M: IntVar](nn.Module):
    def forward(self, x: Tensor[[N, M]]) -> Tensor[[N, M * 2]]:
        return torch.cat([x, x], dim=1)


def test_module_callable[N: IntVar, M: IntVar](
    module: SimpleModule[N, M], x: Tensor[[N, M]]
):
    """Test that we can call module(x) instead of module.forward(x)"""
    # Should be able to call module directly
    result = module(x)
    assert_type(result, Tensor[[N, M * 2]])
    call_attr_result = module.__call__(x)
    assert_type(call_attr_result, Tensor[[N, M * 2]])


class GenericModule[N: IntVar, M: IntVar](nn.Module):
    def forward(self, x: Tensor[[N, M]]) -> Tensor[[N, M]]:
        return x


class ModuleCallback[N: IntVar, M: IntVar](Protocol):
    def __call__(self, x: Tensor[[N, M]]) -> Tensor[[N, M * 2]]: ...


def use_callback_protocol[N: IntVar, M: IntVar](
    callback: ModuleCallback[N, M], x: Tensor[[N, M]]
) -> Tensor[[N, M * 2]]:
    return callback(x)


def test_module_matches_callback_protocol[N: IntVar, M: IntVar](
    module: SimpleModule[N, M], x: Tensor[[N, M]]
):
    callback: ModuleCallback[N, M] = module
    result = use_callback_protocol(callback, x)
    assert_type(result, Tensor[[N, M * 2]])


def test_generic_module_callable[B: IntVar, C: IntVar](
    module: GenericModule[B, C], x: Tensor[[B, C]]
):
    """Test that generic modules are callable"""
    # Call module directly
    result = module(x)
    assert_type(result, Tensor[[B, C]])
    call_attr_result = module.__call__(x)
    assert_type(call_attr_result, Tensor[[B, C]])


# For now, test with concrete shapes by calling the generic functions
# Eventually these will be called with symbolic shapes from real model code
test_module_callable(SimpleModule(), torch.randn(5, 10))
test_module_matches_callback_protocol(SimpleModule(), torch.randn(5, 10))
test_generic_module_callable(GenericModule(), torch.randn(4, 8))
