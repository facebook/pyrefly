# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cython


@cython.cclass
class InitStub:
    def __cinit__(self, x):
        pass


@cython.cclass
class NewStub:
    def __cinit__(self, x):
        pass
