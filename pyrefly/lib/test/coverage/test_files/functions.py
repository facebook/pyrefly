# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def foo(x: int, y: str) -> bool:
    return True


def foo_unannotated(x, y):
    return True


class C:
    def bar(self, x: int, y: str) -> bool:
        return True

    class Inner:
        def baz(self, x: int, y: str) -> bool:
            return True
