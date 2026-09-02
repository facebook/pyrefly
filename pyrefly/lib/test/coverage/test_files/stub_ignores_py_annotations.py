# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class C:
    def covered(self, x: int) -> str:
        return str(x)

    def __setstate__(self, state: tuple[int, bool]):
        pass


def typed_in_py(x: int) -> str:
    return str(x)


z: int = 5
