# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test from generic_basic.rs line 121-126 - generic function with type[T]"""

from typing import assert_type


class A: ...


class B: ...


def f[E](e: type[E]) -> E: ...


# Test what types we actually get
result_a = f(A)
assert_type(result_a, A)

result_b = f(B)
assert_type(result_b, B)

# These are the expected types from the test
assert_type(f(A), A)
assert_type(f(B), B)
