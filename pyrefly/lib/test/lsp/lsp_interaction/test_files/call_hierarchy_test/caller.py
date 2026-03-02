# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from callee import my_function


def caller_one():
    my_function()


def caller_two():
    result = my_function()
    return result


class MyClass:
    def method_caller(self):
        my_function()
