# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# attrs @define and @attr.s(auto_attribs=True) class fields are IMPLICIT (0 typable),
# matching typestats behavior.

import attr
from attrs import define


@define
class Point:
    x: float
    y: float


@attr.s(auto_attribs=True)
class Color:
    r: int
    g: int
    b: int
