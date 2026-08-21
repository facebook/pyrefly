# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeAlias, TypeAliasType, TypeVar

T = TypeVar("T")

# Implicit type alias (bare assignment with recognizable type RHS)
Alias1 = list[int]

# Explicit TypeAlias annotation (legacy form)
Alias2: TypeAlias = int | str

# TypeAliasType call (runtime equivalent of PEP 695)
Alias3 = TypeAliasType("Alias3", int)

# PEP 695 type alias
type Alias4 = float

# Regular typed variable for baseline comparison
x: int = 42
y: type[int] = int


def some_func() -> None:
    pass


class SomeClass:
    my_field = 42
