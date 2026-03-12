# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Debug test for Dim-bounded type parameter binding to expressions"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def simple_identity[D](x: Tensor[D]) -> Tensor[D]:
    """Simple identity function with Dim-bounded type parameter"""
    return x


def test_simple[N](x: Tensor[(N * 2)]):
    # Should bind D to (N * 2) and return Tensor[N * 2]
    y = simple_identity(x)
    assert_type(y, Tensor[(N * 2)])


def generic_func[D1, D2](
    x: Tensor[D1],
    y: Tensor[D2],
) -> Tensor[D1, D2]:
    """Function with two Dim-bounded type parameters"""
    ...


def test_two_params[N](
    a: Tensor[N],
    b: Tensor[(N * 2)],
):
    # Should bind D1 to N and D2 to (N * 2)
    result = generic_func(a, b)
    assert_type(result, Tensor[N, (N * 2)])
