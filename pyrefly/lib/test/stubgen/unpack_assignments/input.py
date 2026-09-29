# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from enum import Enum


A, B = 1, "module"
_PRIVATE, PUBLIC = 2, "public"
FIRST, *REST = os.environ["MY_VAR"].split(" ")
[LX, LY] = [10, 20]


class C:
    A, B = 3, "class"


class Color(Enum):
    RED, BLUE = 1, 2
