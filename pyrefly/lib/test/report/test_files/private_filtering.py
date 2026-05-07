# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests typestats filtering rules:
# - Non-public names (starting with _) are excluded
# - Dunder names (__x__) are kept EXCEPT excluded module dunders
# - Excluded module dunders: __all__, __dir__, __doc__, __getattr__

from typing import Any, List


def public_func(x: int) -> str:
    return str(x)


def _private_func(x: int) -> str:
    return str(x)


def __dunder_func__(x: int) -> str:
    return str(x)


public_var: int = 42
_private_var: str = "hidden"

# Excluded module dunders
__all__: List[str] = ["public_func"]
__doc__: str = "module doc"

# Non-excluded module dunder (should be included)
__version__: str = "1.0"


class MyClass:
    def __init__(self):
        self.public_attr: int = 1
        self._private_attr: str = "hidden"

    def public_method(self) -> int:
        return 1

    def _private_method(self) -> int:
        return 2

    def __dunder_method__(self) -> int:
        return 3
