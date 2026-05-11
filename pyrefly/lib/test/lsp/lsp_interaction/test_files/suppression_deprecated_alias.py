# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class A:
    def f(self, x: int):
        pass


class B(A):
    def f(self, x1: int):  # pyrefly: ignore[bad-param-name-override]
        pass
