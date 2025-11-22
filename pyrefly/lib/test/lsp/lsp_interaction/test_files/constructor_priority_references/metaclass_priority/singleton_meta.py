# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Meta(type):
    def __call__(cls, a: int) -> None: ...


class Singleton(metaclass=Meta):
    def __init__(self):
        pass
