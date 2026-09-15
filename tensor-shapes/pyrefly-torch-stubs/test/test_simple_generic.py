# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test simple generic function - does substitution work?"""

from typing import assert_type


def identity[T](x: T) -> T:
    """Simple identity function"""
    return x


# Test with regular types
result1 = identity(5)
assert_type(result1, int)

result2 = identity("hello")
assert_type(result2, str)


# Test with class instances
class MyClass: ...


result3 = identity(MyClass())
assert_type(result3, MyClass)

# Test assert_type
assert_type(identity(5), int)
assert_type(identity("hello"), str)
assert_type(identity(MyClass()), MyClass)
