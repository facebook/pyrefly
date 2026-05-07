# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for implicit class-body dunder attributes.
# __slots__, __doc__, __module__ etc. have types determined by the Python
# runtime and should be excluded from coverage counting (0 slots).


class WithDunderAttrs:
    """Class that assigns dunder attrs in __init__ — they should be excluded."""

    def __init__(self, x: int):
        self.x = x
        self.__doc__ = "instance doc"
        self.__module__ = "mymodule"


class WithAnnotatedDunder:
    """Class-body annotated dunder + init assignment — still excluded."""

    __module__: str
    regular_attr: int

    def __init__(self):
        self.__module__ = "mymodule"
        self.regular_attr = 42


class RegularOnly:  # noqa: B903
    """Class with only regular attrs — all counted."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value
