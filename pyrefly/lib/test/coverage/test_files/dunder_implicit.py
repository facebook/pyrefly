# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for dunder methods with implicit return types.
# __init__ → None, __bool__ → bool, __len__ → int, etc.
# These return slots should be IMPLICIT (0 slots).


class MyClass:
    def __init__(self):
        pass

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return 0

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    def regular_method(self) -> int:
        return 42
