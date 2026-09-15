# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

class Base:
    def m(self, x: int) -> None: ...

class Sub(Base): ...
