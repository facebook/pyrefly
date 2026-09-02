# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class C:
    def __init__(self, x):
        self.attr = x

    def method(self, a, b):
        return a


def only_in_py(a, b):
    return a
