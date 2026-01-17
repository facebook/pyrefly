# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypedDict


class MyTypedDict(TypedDict):
    name: str
    age: int


def returns_typed_dict():
    return MyTypedDict(name="Alice", age=30)
