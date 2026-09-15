# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test torch.no_grad preserves decorated function signatures.

The hand-written `no_grad` stub gives `__call__` a generic identity signature,
so context-manager and decorator usage keep precise types instead of degrading
to Any (which would silently disable shape checking for decorated functions).
"""

from typing import assert_type, TYPE_CHECKING

from shape_extensions import IntVar

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@torch.no_grad()
def decorated[N: IntVar](x: Tensor[[N]]) -> Tensor[[N]]:
    return x


def test_no_grad_decorator_preserves_signature[N: IntVar](x: Tensor[[N]]):
    y = decorated(x)
    assert_type(y, Tensor[[N]])


def test_no_grad_context_manager[N: IntVar](x: Tensor[[N]]):
    with torch.no_grad():
        y = x.relu()
    assert_type(y, Tensor[[N]])
