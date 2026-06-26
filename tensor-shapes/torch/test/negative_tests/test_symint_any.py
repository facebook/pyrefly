# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test Dim type behavior with Any and implicit parameters.

This test documents that int literals can be assigned to Dim types,
including bare Dim, Dim[Any], or passed to functions with type parameters.
"""

from typing import Any, assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import Dim

symint_implicit_any: Dim = 4
assert_type(symint_implicit_any, Dim)
symint_explicit_any: Dim[Any] = 4
assert_type(symint_explicit_any, Dim[Any])


def accept_and_return_symint[N](s: Dim[N]) -> Dim[N]:
    return s


def test_accept_and_return_symint():
    s = accept_and_return_symint(4)
    assert_type(s, Dim[4])
    n: int = 4
    s_n = accept_and_return_symint(n)
    assert_type(s_n, Dim)
    s_implicit_any = accept_and_return_symint(symint_implicit_any)
    assert_type(s_implicit_any, Dim)
    s_explicit_any = accept_and_return_symint(symint_explicit_any)
    assert_type(s_explicit_any, Dim[Any])
