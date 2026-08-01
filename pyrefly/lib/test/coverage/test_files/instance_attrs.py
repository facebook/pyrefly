# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for instance attribute detection via self.x in __init__.
# Ported from typestats TestInstanceAttrs.


class Untyped:
    def __init__(self):
        self.x = 1


class Typed:
    def __init__(self):
        self.x: int = 1


class ClassBodyWins:
    x: int

    def __init__(self):
        self.x = 1


class MultipleAttrs:
    def __init__(self):
        self.a = 1
        self.b: str = "hello"
        self.c = []
