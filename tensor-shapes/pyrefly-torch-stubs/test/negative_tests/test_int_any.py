# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test Int type behavior with Any and implicit parameters.

This test documents that int literals can be assigned to Int types,
including bare Int, Int[Any], or passed to functions with type parameters.
"""

from typing import Any, assert_type, TYPE_CHECKING

from shape_extensions import IntVar

if TYPE_CHECKING:
    from shape_extensions import Int

int_implicit_any: Int = 4
assert_type(int_implicit_any, Int)
int_explicit_any: Int[Any] = 4
assert_type(int_explicit_any, Int[Any])


def accept_and_return_int[N: IntVar](s: Int[N]) -> Int[N]:
    return s


def test_accept_and_return_int():
    s = accept_and_return_int(4)
    assert_type(s, Int[4])
    n: int = 4
    s_n = accept_and_return_int(n)
    assert_type(s_n, Int)
    s_implicit_any = accept_and_return_int(int_implicit_any)
    assert_type(s_implicit_any, Int)
    s_explicit_any = accept_and_return_int(int_explicit_any)
    assert_type(s_explicit_any, Int[Any])
