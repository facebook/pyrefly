# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test if Dim type variables work correctly in non-Tensor contexts"""

from typing import Any, assert_type


# Test 1: Dim type variable in a regular generic class
class MyContainer[N]:
    def __init__(self, value: int):
        self.value = value


def process_container[N](c: MyContainer[N]) -> MyContainer[N]:
    return c


# E: Expected a type form, got instance of `Literal[5]`
container: MyContainer[5] = MyContainer(42)
result = process_container(container)
assert_type(result, MyContainer[Any])


# Test 2: Dim type variable as a regular parameter (not in subscript)
def identity_dim[N](x: int) -> int:
    """Function with Dim type param but not used in subscript"""
    return x


result2 = identity_dim(42)
assert_type(result2, int)
