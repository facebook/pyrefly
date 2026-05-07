# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __static__ import int64


class MyClass:
    def bar(self, x: int64) -> None:
        pass


obj = MyClass()
obj.bar(42)
