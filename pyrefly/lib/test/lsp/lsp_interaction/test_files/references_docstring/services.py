# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from a import A
from b import B


def use_a(x: A) -> A:
    return x


def use_b(x: B) -> B:
    return x
