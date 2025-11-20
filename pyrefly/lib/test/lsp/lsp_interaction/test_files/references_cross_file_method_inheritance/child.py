# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from base import Base


class Child(Base):
    def method(self) -> str:
        return "child"


child = Child()
child.method()
