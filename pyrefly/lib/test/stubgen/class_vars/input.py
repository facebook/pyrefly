# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# https://github.com/facebook/pyrefly/issues/3890

from enum import Enum


class C:
    # Bare assignments in a class body are implicit class variables and must be
    # annotated `ClassVar[...]` so the stub doesn't mistype them as instance
    # attributes.
    FILEMAP = {"a": "adj"}
    DEFAULTS = [1, 2, 3]

    # An explicit annotation is preserved verbatim and stays an instance attribute.
    name: str

    def some_method(self) -> str:
        return f"<{self.name}>"

    # A method rebinding is also an implicit class variable.
    __repr__ = some_method


class Color(Enum):
    # Bare assignments in an enum body are enum members, not class variables, so
    # they must NOT be wrapped in `ClassVar[...]`.
    RED = 1
    GREEN = 2
    BLUE = 3
    COMPLEX = object()
