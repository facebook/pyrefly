# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for @type_check_only exclusion from coverage counting.
# Decorated functions and classes should not appear in the report.

from typing import type_check_only


@type_check_only
def helper(x: int) -> str:
    return ""


@type_check_only
class InternalProtocol:
    def method(self, value: str) -> bool:
        return True


def regular_function(a: int, b) -> str:
    return ""


class RegularClass:  # noqa: B903
    def __init__(self, name: str):
        self.name = name
