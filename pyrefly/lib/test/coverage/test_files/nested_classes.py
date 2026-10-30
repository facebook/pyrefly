# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Outer:
    def method(self, x: int) -> bool:
        return True

    class Inner:
        def inner_method(self, x):
            pass
