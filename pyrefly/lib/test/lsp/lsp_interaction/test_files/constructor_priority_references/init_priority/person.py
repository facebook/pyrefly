# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Person:
    """Class that only overrides __init__, not __new__"""

    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age
