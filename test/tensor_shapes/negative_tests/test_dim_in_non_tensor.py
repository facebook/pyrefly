# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test if Dim type variables work correctly in non-Tensor contexts"""

from typing import assert_type, Generic, reveal_type


# Test 1: Dim type variable in a regular generic class
class MyContainer[N]:
    def __init__(self, value: int):
        self.value = value


def process_container[N](c: MyContainer[N]) -> MyContainer[N]:
    return c


container: MyContainer[5] = MyContainer(42)
result = process_container(container)
reveal_type(result)  # Should be: MyContainer[5]


# Test 2: Dim type variable as a regular parameter (not in subscript)
def identity_dim[N](x: int) -> int:
    """Function with Dim type param but not used in subscript"""
    return x


result2 = identity_dim(42)
reveal_type(result2)  # Should be: int
