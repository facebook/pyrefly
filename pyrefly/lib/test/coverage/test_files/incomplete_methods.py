# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Complete:
    def method(self, x: int) -> bool:
        return True


class Incomplete:
    def method_unannotated(self, x):
        pass

    def method_partial(self, x: int):
        pass

    def method_complete(self, x: int) -> bool:
        return True
