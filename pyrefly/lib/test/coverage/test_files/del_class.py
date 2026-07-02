# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# gh-4021: methods and instance attrs of a module-scope `del`eted class must
# not appear in the report as dangling symbols.


class Gone:
    def method(self, x):
        pass

    def __init__(self):
        self.attr = 1

    alias = method


del Gone


class Kept:
    def method(self, x: int) -> int:
        return x

    def __init__(self) -> None:
        self.attr: int = 1
