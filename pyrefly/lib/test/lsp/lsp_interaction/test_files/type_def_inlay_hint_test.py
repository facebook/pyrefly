# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class MyClass:
    def __init__(self, value: int) -> None:
        self.value = value


def returns_my_class():
    return MyClass(42)


result = returns_my_class()
