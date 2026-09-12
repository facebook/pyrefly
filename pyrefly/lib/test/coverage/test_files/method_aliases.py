# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for class method aliases: __rand__ = __and__.
# Ported from typestats TestClassMethodAlias.


class Simple:
    def __and__(self, other: int) -> bool:
        return True

    __rand__ = __and__
