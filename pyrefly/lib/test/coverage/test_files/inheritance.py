# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Base:
    def base_method(self, x):
        pass


class Child(Base):
    def child_method(self, x: int) -> bool:
        return True
